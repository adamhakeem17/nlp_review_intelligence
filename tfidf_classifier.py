"""
tfidf_classifier.py
-------------------
TF-IDF + Logistic Regression baseline classifier.

Purpose:
  - Fast to train (seconds vs minutes for BERT)
  - Fully interpretable (top weighted features per class)
  - Strong baseline to compare transformer performance against
  - Works offline with no HuggingFace downloads needed

Pipeline:
  raw text → TfidfVectorizer → LogisticRegression → label + probabilities

Persisted with joblib. No PyTorch required.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import SENTIMENT_LABELS, TFIDF_MODEL_PATH, TFIDFConfig, tfidf_cfg


class TFIDFSentimentClassifier:
    """
    Sklearn pipeline: TF-IDF vectoriser → Logistic Regression.

    Usage:
        clf = TFIDFSentimentClassifier()
        clf.fit(train_df["text"], train_df["label"])
        result = clf.predict("This is great!")
    """

    def __init__(self, cfg: TFIDFConfig = tfidf_cfg) -> None:
        self.cfg      = cfg
        self.pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            max_features = self.cfg.max_features,
            ngram_range  = self.cfg.ngram_range,
            sublinear_tf = self.cfg.sublinear_tf,
            min_df       = self.cfg.min_df,
            max_df       = self.cfg.max_df,
            strip_accents = "unicode",
            analyzer     = "word",
            token_pattern = r"\b[a-zA-Z][a-zA-Z]+\b",   # alpha tokens ≥ 2 chars
        )
        classifier = LogisticRegression(
            C            = self.cfg.C,
            max_iter     = self.cfg.max_iter,
            solver       = self.cfg.solver,
            class_weight = self.cfg.class_weight,
            random_state = 42,
        )
        return Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFSentimentClassifier":
        """
        Fit the TF-IDF + LR pipeline on training data.

        Args:
            texts:  List of raw review strings.
            labels: List of label strings ("negative", "neutral", "positive").
        """
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(texts, labels)
        print(f"[TFIDFSentimentClassifier] Fitted on {len(texts)} samples")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """Classify a single review. Returns label, confidence, probabilities."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Classify a list of reviews in one pipeline call."""
        if self.pipeline is None:
            raise RuntimeError("Model not fitted. Call .fit() or .load() first.")
        probs      = self.pipeline.predict_proba(texts)
        classes    = self.pipeline.classes_
        results    = []
        for prob_row in probs:
            label_idx = int(prob_row.argmax())
            label     = classes[label_idx]
            results.append({
                "label":         label,
                "label_idx":     SENTIMENT_LABELS.index(label),
                "confidence":    float(prob_row[label_idx]),
                "probabilities": {cls: float(p) for cls, p in zip(classes, prob_row)},
            })
        return results

    # ── Interpretability ──────────────────────────────────────────────────────

    def top_features(self, label: str, n: int = 15) -> pd.DataFrame:
        """
        Return the n most influential TF-IDF features for a given class.

        Args:
            label: One of "negative", "neutral", "positive".
            n:     Number of top features to return.

        Returns:
            DataFrame with columns: feature, weight.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not fitted.")

        vectorizer: TfidfVectorizer    = self.pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = self.pipeline.named_steps["clf"]
        feature_names = vectorizer.get_feature_names_out()

        class_list = list(classifier.classes_)
        if label not in class_list:
            raise ValueError(f"Unknown label '{label}'. Valid: {class_list}")

        class_idx = class_list.index(label)
        coef      = classifier.coef_[class_idx]
        top_idx   = np.argsort(coef)[::-1][:n]

        return pd.DataFrame({
            "feature": feature_names[top_idx],
            "weight":  coef[top_idx],
        })

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path | str = TFIDF_MODEL_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"[TFIDFSentimentClassifier] Saved to {path}")

    @classmethod
    def load(cls, path: Path | str = TFIDF_MODEL_PATH, cfg: Optional[TFIDFConfig] = None) -> "TFIDFSentimentClassifier":
        instance = cls(cfg=cfg or tfidf_cfg)
        instance.pipeline = joblib.load(path)
        print(f"[TFIDFSentimentClassifier] Loaded from {path}")
        return instance
