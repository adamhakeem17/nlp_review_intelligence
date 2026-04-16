"""
predictor.py
------------
Clean inference API combining sentiment classification,
topic extraction, and summarisation into one pipeline call.

Responsibilities:
  - ReviewPredictor: run the full pipeline on a single review or batch
  - CorpusAnalyser: run batch sentiment + topics + summary on a DataFrame
  - Both return typed result dataclasses

Designed for import by app.py (Streamlit) and cli.py.
No training. No evaluation metrics. No UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    BERT_MODEL_DIR, TFIDF_MODEL_PATH,
    SENTIMENT_EMOJI, InferenceConfig, inference_cfg,
)
from data_loader import clean_text
from summariser import ExtractiveSummariser
from topic_extractor import AspectAnalyser, KeyphraseExtractor, summarise_topics


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ReviewPrediction:
    text:          str
    label:         str
    label_idx:     int
    confidence:    float
    probabilities: dict
    flagged:       bool   # True if confidence < threshold
    model_used:    str

    @property
    def emoji(self) -> str:
        return SENTIMENT_EMOJI.get(self.label, "⚪")

    def __str__(self) -> str:
        flag = " ⚠️ LOW CONFIDENCE" if self.flagged else ""
        return f"{self.emoji} {self.label.upper()} ({self.confidence:.1%}){flag}"


@dataclass
class CorpusAnalysis:
    df:              pd.DataFrame   # original df + predicted_label + confidence
    summary:         str            # extractive summary across all reviews
    by_sentiment:    dict           # {label: summary string}
    topic_summary:   object         # CorpusTopicSummary
    label_counts:    dict           # {label: count}
    avg_confidence:  float


# ── Single review predictor ───────────────────────────────────────────────────

class ReviewPredictor:
    """
    Classifies a single review using either DistilBERT or TF-IDF.

    Loads the model lazily on first call. Falls back to TF-IDF if
    BERT weights are not available.

    Usage:
        predictor = ReviewPredictor.auto()
        result    = predictor.predict("This product is fantastic!")
    """

    def __init__(self, model, model_name: str, cfg: InferenceConfig = inference_cfg) -> None:
        self._model     = model
        self._name      = model_name
        self.cfg        = cfg

    @classmethod
    def from_bert(cls, directory: Path | str = BERT_MODEL_DIR, cfg: InferenceConfig = inference_cfg) -> "ReviewPredictor":
        from bert_classifier import BERTSentimentClassifier
        model = BERTSentimentClassifier.load(directory)
        return cls(model=model, model_name="DistilBERT", cfg=cfg)

    @classmethod
    def from_tfidf(cls, path: Path | str = TFIDF_MODEL_PATH, cfg: InferenceConfig = inference_cfg) -> "ReviewPredictor":
        from tfidf_classifier import TFIDFSentimentClassifier
        model = TFIDFSentimentClassifier.load(path)
        return cls(model=model, model_name="TF-IDF + LogReg", cfg=cfg)

    @classmethod
    def auto(cls, cfg: InferenceConfig = inference_cfg) -> "ReviewPredictor":
        """Load BERT if available, otherwise fall back to TF-IDF."""
        bert_dir = Path(BERT_MODEL_DIR)
        tfidf_p  = Path(TFIDF_MODEL_PATH)
        if bert_dir.exists() and any(bert_dir.iterdir()):
            return cls.from_bert(bert_dir, cfg)
        elif tfidf_p.exists():
            return cls.from_tfidf(tfidf_p, cfg)
        else:
            raise FileNotFoundError(
                "No trained model found. Run: python train.py --model tfidf"
            )

    def predict(self, text: str) -> ReviewPrediction:
        """Classify a single review string."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[ReviewPrediction]:
        """Classify a list of review strings."""
        cleaned  = [clean_text(t) for t in texts]
        raw_preds = self._model.predict_batch(cleaned)
        return [
            ReviewPrediction(
                text=         texts[i],
                label=        p["label"],
                label_idx=    p["label_idx"],
                confidence=   p["confidence"],
                probabilities=p["probabilities"],
                flagged=      p["confidence"] < self.cfg.confidence_threshold,
                model_used=   self._name,
            )
            for i, p in enumerate(raw_preds)
        ]


# ── Corpus analyser ───────────────────────────────────────────────────────────

class CorpusAnalyser:
    """
    Runs the full NLP pipeline on a DataFrame of reviews:
      1. Sentiment classification (BERT or TF-IDF)
      2. Extractive summarisation per sentiment class
      3. Topic extraction (keyphrases, entities, aspects)

    Usage:
        analyser = CorpusAnalyser(predictor)
        result   = analyser.analyse(df)
    """

    def __init__(
        self,
        predictor:  ReviewPredictor,
        summariser: Optional[ExtractiveSummariser] = None,
    ) -> None:
        self.predictor  = predictor
        self.summariser = summariser or ExtractiveSummariser()

    def analyse(self, df: pd.DataFrame, batch_size: int = 32) -> CorpusAnalysis:
        """
        Run the full pipeline on a DataFrame.

        Args:
            df:         DataFrame with a "text" column.
            batch_size: Number of reviews per prediction batch.

        Returns:
            CorpusAnalysis with enriched DataFrame and all NLP outputs.
        """
        texts = df["text"].tolist()

        # ── Sentiment in batches ───────────────────────────────────────────
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_preds.extend(self.predictor.predict_batch(batch))

        df = df.copy()
        df["predicted_label"] = [p.label      for p in all_preds]
        df["confidence"]      = [p.confidence  for p in all_preds]
        df["flagged"]         = [p.flagged      for p in all_preds]

        # ── Extractive summaries ───────────────────────────────────────────
        overall_summary  = self.summariser.summarise(texts, n=5)
        by_sentiment     = self.summariser.summarise_by_sentiment(df)

        # ── Topic extraction ───────────────────────────────────────────────
        topic_summary = _safe_topic_summary(df)

        label_counts   = df["predicted_label"].value_counts().to_dict()
        avg_confidence = float(df["confidence"].mean())

        return CorpusAnalysis(
            df=df,
            summary=overall_summary,
            by_sentiment=by_sentiment,
            topic_summary=topic_summary,
            label_counts=label_counts,
            avg_confidence=avg_confidence,
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_topic_summary(df: pd.DataFrame):
    """Run topic extraction, returning None if spaCy is unavailable."""
    try:
        return summarise_topics(df, sentiment_col="predicted_label")
    except (OSError, ImportError):
        return None
