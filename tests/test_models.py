"""
tests/test_models.py
--------------------
Unit tests for tfidf_classifier.py and bert_classifier.py.

TF-IDF tests: full fit + predict cycle (no network needed).
BERT tests:   interface validation only — building the full model
              requires a HuggingFace download, so we mock where needed.

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from config import SENTIMENT_LABELS, TFIDFConfig
from tfidf_classifier import TFIDFSentimentClassifier


# ── TF-IDF Classifier ─────────────────────────────────────────────────────────

class TestTFIDFSentimentClassifier:

    @pytest.fixture
    def fitted_clf(self):
        clf = TFIDFSentimentClassifier()
        texts  = (
            ["Great product! Very happy with it."] * 10
            + ["Terrible quality, broke immediately."] * 10
            + ["Okay, nothing special, average."] * 10
        )
        labels = ["positive"] * 10 + ["negative"] * 10 + ["neutral"] * 10
        clf.fit(texts, labels)
        return clf

    def test_fit_returns_self(self):
        clf    = TFIDFSentimentClassifier()
        result = clf.fit(["test text"] * 5, ["positive"] * 3 + ["negative"] * 2)
        assert result is clf

    def test_predict_returns_dict(self, fitted_clf):
        result = fitted_clf.predict("This is amazing!")
        assert isinstance(result, dict)
        assert "label" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_predict_label_valid(self, fitted_clf):
        result = fitted_clf.predict("Excellent product!")
        assert result["label"] in SENTIMENT_LABELS

    def test_confidence_in_range(self, fitted_clf):
        result = fitted_clf.predict("Great item.")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, fitted_clf):
        result = fitted_clf.predict("Decent product.")
        total  = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5

    def test_predict_batch_length(self, fitted_clf):
        texts   = ["Good!", "Bad!", "Okay."]
        results = fitted_clf.predict_batch(texts)
        assert len(results) == 3

    def test_predict_batch_all_valid_labels(self, fitted_clf):
        texts   = ["Love it", "Hate it", "It's fine"]
        results = fitted_clf.predict_batch(texts)
        for r in results:
            assert r["label"] in SENTIMENT_LABELS

    def test_unfitted_predict_raises(self):
        clf = TFIDFSentimentClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict("test")

    def test_top_features_returns_dataframe(self, fitted_clf):
        df = fitted_clf.top_features("positive", n=5)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "weight" in df.columns
        assert len(df) == 5

    def test_top_features_invalid_label(self, fitted_clf):
        with pytest.raises(ValueError, match="Unknown label"):
            fitted_clf.top_features("unknown_label")

    def test_top_features_negative(self, fitted_clf):
        df = fitted_clf.top_features("negative", n=3)
        assert len(df) == 3

    def test_save_and_load(self, fitted_clf):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            fitted_clf.save(path)
            assert path.exists()
            loaded = TFIDFSentimentClassifier.load(path)
            result = loaded.predict("Fantastic quality!")
            assert result["label"] in SENTIMENT_LABELS

    def test_positive_review_classified_correctly(self, fitted_clf):
        # Strong positive signal
        result = fitted_clf.predict("Great product! Very happy with it.")
        assert result["label"] == "positive"

    def test_negative_review_classified_correctly(self, fitted_clf):
        result = fitted_clf.predict("Terrible quality, broke immediately.")
        assert result["label"] == "negative"

    def test_custom_config(self):
        cfg = TFIDFConfig(max_features=500, ngram_range=(1, 1))
        clf = TFIDFSentimentClassifier(cfg=cfg)
        clf.fit(["positive text"] * 5, ["positive"] * 3 + ["negative"] * 2)
        assert clf.pipeline is not None


# ── BERT Classifier (interface only) ─────────────────────────────────────────

class TestBERTClassifierInterface:
    """
    Tests the BERTSentimentClassifier interface without downloading the model.
    Uses mocking to avoid HuggingFace Hub network calls.
    """

    def test_instantiation(self):
        from bert_classifier import BERTSentimentClassifier
        clf = BERTSentimentClassifier()
        assert clf.model is None
        assert clf.tokenizer is None

    def test_predict_before_build_raises(self):
        from bert_classifier import BERTSentimentClassifier
        clf = BERTSentimentClassifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict("test")

    def test_predict_batch_before_build_raises(self):
        from bert_classifier import BERTSentimentClassifier
        clf = BERTSentimentClassifier()
        with pytest.raises(RuntimeError, match="not loaded"):
            clf.predict_batch(["test"])

    def test_count_parameters_before_build(self):
        from bert_classifier import BERTSentimentClassifier
        clf = BERTSentimentClassifier()
        result = clf.count_parameters()
        assert result == {}

    def test_predict_batch_with_mocked_model(self):
        """Validate the output schema of predict_batch without a real model."""
        import torch
        import numpy as np
        from bert_classifier import BERTSentimentClassifier

        clf = BERTSentimentClassifier()

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids":      torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value.logits = torch.tensor([[0.1, 0.3, 0.6]])
        clf.tokenizer = mock_tokenizer
        clf.model     = mock_model

        results = clf.predict_batch(["test review"])
        assert len(results) == 1
        r = results[0]
        assert "label"         in r
        assert "confidence"    in r
        assert "probabilities" in r
        assert r["label"] in SENTIMENT_LABELS
        assert 0.0 <= r["confidence"] <= 1.0
        assert abs(sum(r["probabilities"].values()) - 1.0) < 1e-4
