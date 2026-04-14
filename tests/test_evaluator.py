"""
tests/test_evaluator.py
-----------------------
Unit tests for evaluator.py — pure metric computation, no models.

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from evaluator import (
    EvalResult,
    compare_models,
    evaluate,
    plot_confidence_histogram,
    plot_keyphrase_bar,
    plot_sentiment_distribution,
)


# ── evaluate() ────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_perfect_accuracy(self):
        labels = ["positive", "negative", "neutral"]
        result = evaluate(labels, labels, "Test")
        assert result.accuracy == pytest.approx(1.0)

    def test_zero_accuracy(self):
        true  = ["positive", "positive"]
        pred  = ["negative", "negative"]
        result = evaluate(true, pred, "Test")
        assert result.accuracy == pytest.approx(0.0)

    def test_accuracy_partial(self):
        true  = ["positive", "negative", "positive", "neutral"]
        pred  = ["positive", "negative", "negative", "neutral"]
        result = evaluate(true, pred, "Test")
        assert result.accuracy == pytest.approx(0.75)

    def test_model_name_stored(self):
        result = evaluate(["positive"], ["positive"], model_name="MyModel")
        assert result.model_name == "MyModel"

    def test_conf_matrix_shape(self):
        labels = ["positive", "negative", "neutral"] * 3
        result = evaluate(labels, labels, "Test")
        assert result.conf_matrix.shape == (3, 3)

    def test_conf_matrix_diagonal_sums_to_correct(self):
        labels = ["positive", "positive", "negative"]
        result = evaluate(labels, labels, "Test")
        assert result.conf_matrix.diagonal().sum() == 3

    def test_precision_in_range(self):
        true  = ["positive"] * 5 + ["negative"] * 5
        pred  = ["positive"] * 4 + ["negative"] + ["negative"] * 5
        result = evaluate(true, pred, "Test")
        for cls in result.class_names:
            if cls in result.precision:
                assert 0.0 <= result.precision[cls] <= 1.0

    def test_recall_in_range(self):
        true  = ["positive"] * 5 + ["negative"] * 5
        pred  = ["positive"] * 5 + ["negative"] * 5
        result = evaluate(true, pred, "Test")
        for cls in result.class_names:
            if cls in result.recall:
                assert 0.0 <= result.recall[cls] <= 1.0

    def test_macro_f1_in_range(self):
        true  = ["positive", "negative", "neutral"] * 4
        pred  = ["positive", "positive", "neutral"] * 4
        result = evaluate(true, pred, "Test")
        assert 0.0 <= result.macro_f1 <= 1.0

    def test_to_dataframe_has_macro_avg(self):
        true  = ["positive", "negative"]
        pred  = ["positive", "negative"]
        df    = evaluate(true, pred, "Test").to_dataframe()
        assert "macro avg" in df["class"].values

    def test_str_representation(self):
        result = evaluate(["positive"], ["positive"], "TestModel")
        s = str(result)
        assert "TestModel" in s
        assert "Accuracy" in s


# ── compare_models() ──────────────────────────────────────────────────────────

class TestCompareModels:
    def test_returns_dataframe(self):
        r1 = evaluate(["positive", "negative"], ["positive", "negative"], "M1")
        r2 = evaluate(["positive", "negative"], ["positive", "positive"], "M2")
        df = compare_models(r1, r2)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_model_names_in_column(self):
        r1 = evaluate(["positive"], ["positive"], "BERT")
        r2 = evaluate(["positive"], ["positive"], "TF-IDF")
        df = compare_models(r1, r2)
        assert "BERT"   in df["model"].values
        assert "TF-IDF" in df["model"].values

    def test_accuracy_column_present(self):
        r = evaluate(["positive"], ["positive"], "M")
        df = compare_models(r)
        assert "accuracy" in df.columns

    def test_macro_f1_column_present(self):
        r = evaluate(["positive"], ["positive"], "M")
        df = compare_models(r)
        assert "macro_f1" in df.columns


# ── Plotly chart builders ─────────────────────────────────────────────────────

class TestChartBuilders:
    def test_sentiment_distribution_chart(self):
        df  = pd.DataFrame({"label": ["positive", "negative", "positive", "neutral"]})
        fig = plot_sentiment_distribution(df, "label")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_keyphrase_bar_chart(self):
        df  = pd.DataFrame({"phrase": ["good quality", "fast shipping"], "frequency": [10, 5]})
        fig = plot_keyphrase_bar(df)
        assert fig is not None

    def test_keyphrase_bar_empty(self):
        df  = pd.DataFrame(columns=["phrase", "frequency"])
        fig = plot_keyphrase_bar(df)
        assert fig is not None   # should return a "no data" figure, not crash

    def test_confidence_histogram(self):
        preds = [
            {"label": "positive", "confidence": 0.9},
            {"label": "negative", "confidence": 0.6},
            {"label": "neutral",  "confidence": 0.7},
        ]
        fig = plot_confidence_histogram(preds)
        assert fig is not None


# ── EvalResult edge cases ─────────────────────────────────────────────────────

class TestEvalResultEdgeCases:
    def test_empty_inputs(self):
        result = evaluate([], [], "Empty")
        assert result.accuracy == 0.0

    def test_single_class_only(self):
        true  = ["positive"] * 10
        pred  = ["positive"] * 10
        result = evaluate(true, pred, "SingleClass")
        assert result.accuracy == pytest.approx(1.0)

    def test_unknown_labels_handled(self):
        # Labels not in class_names should be ignored gracefully
        true  = ["positive", "unknown"]
        pred  = ["positive", "positive"]
        result = evaluate(true, pred, "Test")
        assert isinstance(result.accuracy, float)
