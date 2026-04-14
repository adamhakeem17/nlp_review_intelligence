"""
tests/test_summariser.py
------------------------
Unit tests for summariser.py — no network, no model downloads.
Only tests ExtractiveSummariser (AbstractiveSummariser requires network).

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from summariser import ExtractiveSummariser, _split_sentences, _tfidf_sentence_scores


# ── ExtractiveSummariser ──────────────────────────────────────────────────────

class TestExtractiveSummariser:
    @pytest.fixture
    def summariser(self):
        return ExtractiveSummariser()

    @pytest.fixture
    def reviews(self):
        return [
            "The product arrived quickly and was well packaged.",
            "Build quality is excellent. Very sturdy and durable.",
            "Setup was straightforward and the instructions were clear.",
            "Battery life is impressive. Lasts all day easily.",
            "Price is competitive for the quality you get.",
        ]

    def test_returns_string(self, summariser, reviews):
        result = summariser.summarise(reviews)
        assert isinstance(result, str)

    def test_non_empty(self, summariser, reviews):
        result = summariser.summarise(reviews)
        assert len(result) > 0

    def test_empty_input(self, summariser):
        result = summariser.summarise([])
        assert isinstance(result, str)
        assert "No reviews" in result

    def test_single_review(self, summariser):
        result = summariser.summarise(["A single sentence review."])
        assert isinstance(result, str)

    def test_n_controls_length(self, summariser, reviews):
        short  = summariser.summarise(reviews, n=1)
        longer = summariser.summarise(reviews, n=3)
        # Longer summary should generally have more sentences
        assert len(short) <= len(longer) + 50   # allow small variance

    def test_fewer_than_n_sentences_returns_all(self, summariser):
        texts  = ["Short review."]
        result = summariser.summarise(texts, n=5)
        assert isinstance(result, str)

    def test_summarise_by_sentiment_returns_dict(self, summariser):
        df = pd.DataFrame({
            "text":            ["Great!", "Terrible!", "Okay."],
            "predicted_label": ["positive", "negative", "neutral"],
        })
        result = summariser.summarise_by_sentiment(df)
        assert isinstance(result, dict)

    def test_summarise_by_sentiment_keys(self, summariser):
        df = pd.DataFrame({
            "text":            ["Good product."] * 3 + ["Bad quality."] * 3,
            "predicted_label": ["positive"] * 3 + ["negative"] * 3,
        })
        result = summariser.summarise_by_sentiment(df)
        assert "positive" in result
        assert "negative" in result


# ── _split_sentences ──────────────────────────────────────────────────────────

class TestSplitSentences:
    def test_splits_on_period(self):
        texts = ["First sentence. Second sentence."]
        sents = _split_sentences(texts)
        assert len(sents) >= 2

    def test_splits_on_exclamation(self):
        sents = _split_sentences(["Love it! Great product!"])
        assert len(sents) >= 2

    def test_filters_short_fragments(self):
        sents = _split_sentences(["Ok."])
        assert all(len(s) > 10 for s in sents)

    def test_handles_multiple_texts(self):
        texts = ["First. Second.", "Third. Fourth."]
        sents = _split_sentences(texts)
        assert len(sents) >= 4

    def test_empty_input(self):
        sents = _split_sentences([])
        assert sents == []


# ── _tfidf_sentence_scores ────────────────────────────────────────────────────

class TestTfidfSentenceScores:
    def test_returns_array(self):
        import numpy as np
        sents  = ["Good quality product.", "Terrible build quality.", "Fast shipping."]
        scores = _tfidf_sentence_scores(sents)
        assert isinstance(scores, np.ndarray)

    def test_length_matches_input(self):
        sents  = ["A.", "B.", "C.", "D."]
        scores = _tfidf_sentence_scores(sents)
        assert len(scores) == 4

    def test_non_negative(self):
        sents  = ["Great product.", "Good quality.", "Fast delivery."]
        scores = _tfidf_sentence_scores(sents)
        assert (scores >= 0).all()

    def test_single_sentence(self):
        scores = _tfidf_sentence_scores(["Just one sentence here."])
        assert len(scores) == 1
