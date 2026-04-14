"""
Extractive and optional abstractive summarisation utilities for review corpora.
"""


from __future__ import annotations

from typing import List, Optional

import numpy as np

from config import SummaryConfig, summary_cfg


# ── Extractive summariser ─────────────────────────────────────────────────────

class ExtractiveSummariser:
    """
    Ranks review sentences by TF-IDF score and returns the top N.

    This is a simple but effective approach for creating a representative
    summary of many short reviews. No model loading required.

    Usage:
        summariser = ExtractiveSummariser()
        summary    = summariser.summarise(["Great product!", "Terrible quality."])
    """

    def __init__(self, cfg: SummaryConfig = summary_cfg) -> None:
        self.cfg = cfg

    def summarise(self, texts: List[str], n: Optional[int] = None) -> str:
        """
        Select the most representative sentences from a list of reviews.

        Args:
            texts: List of review strings.
            n:     Number of sentences to include (default: cfg.extractive_n_sentences).

        Returns:
            A paragraph of the top N most representative sentences.
        """
        n = n or self.cfg.extractive_n_sentences
        sentences = _split_sentences(texts)
        if not sentences:
            return "No reviews to summarise."
        if len(sentences) <= n:
            return " ".join(sentences)

        scores = _tfidf_sentence_scores(sentences)
        top_idx = np.argsort(scores)[::-1][:n]
        # Return in original order for readability
        selected = [sentences[i] for i in sorted(top_idx)]
        return " ".join(selected)

    def summarise_by_sentiment(
        self,
        df,
        sentiment_col: str = "predicted_label",
        n_per_group: int = 2,
    ) -> dict:
        """
        Produce a separate extractive summary per sentiment class.

        Returns:
            dict mapping sentiment label → summary string.
        """
        summaries = {}
        for label in df[sentiment_col].unique():
            subset  = df[df[sentiment_col] == label]["text"].tolist()
            summaries[label] = self.summarise(subset, n=n_per_group)
        return summaries


# ── Abstractive summariser ────────────────────────────────────────────────────

class AbstractiveSummariser:
    """
    Generates a fluent summary paragraph using DistilBART.

    Downloads ~300MB on first use. Subsequent runs use cached weights.
    CPU inference is slow for long inputs — use ExtractiveSummariser for
    large corpora and AbstractiveSummariser for short highlight sets.

    Usage:
        summariser = AbstractiveSummariser()
        summary    = summariser.summarise(["Good product", "Fast shipping"])
    """

    def __init__(self, cfg: SummaryConfig = summary_cfg) -> None:
        self.cfg       = cfg
        self._pipeline = None   # lazy load

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            print(f"[AbstractiveSummariser] Loading {self.cfg.abstractive_model}…")
            self._pipeline = pipeline(
                "summarization",
                model=self.cfg.abstractive_model,
                device=-1,   # CPU
            )
        return self._pipeline

    def summarise(self, texts: List[str]) -> str:
        """
        Generate an abstractive summary from a list of reviews.

        Args:
            texts: List of review strings.

        Returns:
            A generated summary paragraph.
        """
        combined = " ".join(texts)
        # Truncate to max_input_tokens words (rough proxy for tokens)
        words    = combined.split()
        if len(words) > self.cfg.max_input_tokens:
            combined = " ".join(words[: self.cfg.max_input_tokens])

        pipe   = self._get_pipeline()
        output = pipe(
            combined,
            max_length=self.cfg.max_output_tokens,
            min_length=self.cfg.min_output_tokens,
            num_beams=self.cfg.num_beams,
            do_sample=False,
        )
        return output[0]["summary_text"]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_sentences(texts: List[str]) -> List[str]:
    """Naive sentence splitter using punctuation — avoids NLTK dependency."""
    import re
    sentences = []
    for text in texts:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
        if not parts:
            continue

        long_entries = [(idx, p) for idx, p in enumerate(parts) if len(p) > 10]
        short_entries = [(idx, p) for idx, p in enumerate(parts) if len(p) <= 10]

        if len(parts) == 1 and not long_entries:
            continue

        selected = long_entries[:]
        if len(parts) >= 2:
            selected.extend(short_entries)

        if not selected:
            continue

        selected.sort(key=lambda item: item[0])
        sentences.extend(p for _, p in selected)
    return sentences


def _tfidf_sentence_scores(sentences: List[str]) -> np.ndarray:
    """Score sentences by average TF-IDF weight of their tokens."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    try:
        vec    = TfidfVectorizer(stop_words="english", min_df=1)
        matrix = vec.fit_transform(sentences)
        # Mean TF-IDF weight per sentence
        return np.asarray(matrix.mean(axis=1)).flatten()
    except ValueError:
        # Fallback: score by sentence length (longer = more informative)
        return np.array([len(s.split()) for s in sentences], dtype=float)
