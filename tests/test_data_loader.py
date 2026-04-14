"""
tests/test_data_loader.py
--------------------------
Unit tests for data_loader.py — no model calls, no network.

Run:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    clean_text,
    generate_sample_data,
    label_distribution,
    load_csv,
    split_dataset,
)


# ── clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_html_tags(self):
        assert "<b>" not in clean_text("<b>Bold text</b>")
        assert "Bold text" in clean_text("<b>Bold text</b>")

    def test_decodes_html_entities(self):
        result = clean_text("AT&amp;T is great")
        assert "&amp;" not in result
        assert "AT&T" in result

    def test_strips_urls(self):
        result = clean_text("Visit https://example.com for more info")
        assert "https://" not in result
        assert "for more info" in result

    def test_collapses_whitespace(self):
        result = clean_text("too    many    spaces")
        assert "  " not in result

    def test_strips_leading_trailing(self):
        assert clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_punctuation(self):
        # Punctuation carries sentiment — must not be stripped
        result = clean_text("Great product! Highly recommended.")
        assert "!" in result
        assert "." in result

    def test_none_converted(self):
        # Should handle None gracefully via str() conversion
        assert isinstance(clean_text(None), str)


# ── load_csv ──────────────────────────────────────────────────────────────────

class TestLoadCSV:
    @pytest.fixture
    def csv_path(self, tmp_path):
        df = pd.DataFrame({
            "text":  ["Great product!", "Terrible quality.", "It's okay."],
            "label": ["positive", "negative", "neutral"],
        })
        p = tmp_path / "reviews.csv"
        df.to_csv(p, index=False)
        return p

    def test_loads_rows(self, csv_path):
        df = load_csv(csv_path)
        assert len(df) == 3

    def test_expected_columns(self, csv_path):
        df = load_csv(csv_path)
        assert "text" in df.columns
        assert "label" in df.columns
        assert "label_idx" in df.columns

    def test_label_idx_range(self, csv_path):
        df = load_csv(csv_path)
        assert df["label_idx"].between(0, 2).all()

    def test_invalid_label_raises(self, tmp_path):
        df = pd.DataFrame({"text": ["test"], "label": ["unknown"]})
        p  = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="Invalid labels"):
            load_csv(p)

    def test_missing_column_raises(self, tmp_path):
        df = pd.DataFrame({"review": ["test"], "label": ["positive"]})
        p  = tmp_path / "missing.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="Missing columns"):
            load_csv(p, text_col="text")

    def test_max_rows(self, csv_path):
        df = load_csv(csv_path, max_rows=2)
        assert len(df) <= 2

    def test_custom_column_names(self, tmp_path):
        df = pd.DataFrame({"review": ["Nice!"], "sentiment": ["positive"]})
        p  = tmp_path / "custom.csv"
        df.to_csv(p, index=False)
        result = load_csv(p, text_col="review", label_col="sentiment")
        assert len(result) == 1


# ── split_dataset ─────────────────────────────────────────────────────────────

class TestSplitDataset:
    @pytest.fixture
    def df(self):
        n   = 200
        rng = np.random.default_rng(0)
        labels = rng.choice(["positive", "negative", "neutral"], n)
        return pd.DataFrame({
            "text":      [f"review {i}" for i in range(n)],
            "label":     labels,
            "label_idx": [{"positive": 2, "negative": 0, "neutral": 1}[l] for l in labels],
        })

    def test_sizes_sum_to_total(self, df):
        train, val, test = split_dataset(df, val_frac=0.15, test_frac=0.15)
        assert len(train) + len(val) + len(test) == len(df)

    def test_no_overlap(self, df):
        train, val, test = split_dataset(df)
        train_idx = set(train.index)
        val_idx   = set(val.index)
        test_idx  = set(test.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)

    def test_reproducible_with_seed(self, df):
        t1, v1, te1 = split_dataset(df, seed=42)
        t2, v2, te2 = split_dataset(df, seed=42)
        assert list(t1["text"]) == list(t2["text"])

    def test_stratification_preserves_distribution(self, df):
        train, _, _ = split_dataset(df)
        original_dist = df["label"].value_counts(normalize=True)
        train_dist    = train["label"].value_counts(normalize=True)
        for label in original_dist.index:
            assert abs(original_dist[label] - train_dist.get(label, 0)) < 0.10


# ── generate_sample_data ──────────────────────────────────────────────────────

class TestGenerateSampleData:
    def test_returns_dataframe(self, tmp_path):
        path = tmp_path / "reviews.csv"
        df   = generate_sample_data(path, n=50, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self, tmp_path):
        path = tmp_path / "reviews.csv"
        df   = generate_sample_data(path, n=100, seed=0)
        assert len(df) == 100

    def test_saves_to_disk(self, tmp_path):
        path = tmp_path / "reviews.csv"
        generate_sample_data(path, n=20, seed=0)
        assert path.exists()

    def test_has_required_columns(self, tmp_path):
        path = tmp_path / "reviews.csv"
        df   = generate_sample_data(path, n=30, seed=0)
        assert "text" in df.columns
        assert "label" in df.columns

    def test_valid_labels(self, tmp_path):
        path = tmp_path / "reviews.csv"
        df   = generate_sample_data(path, n=50, seed=0)
        assert set(df["label"].unique()).issubset({"positive", "negative", "neutral"})

    def test_all_three_labels_present(self, tmp_path):
        path = tmp_path / "reviews.csv"
        df   = generate_sample_data(path, n=100, seed=0)
        assert len(df["label"].unique()) == 3

    def test_reproducible(self, tmp_path):
        p1 = tmp_path / "r1.csv"
        p2 = tmp_path / "r2.csv"
        d1 = generate_sample_data(p1, n=50, seed=42)
        d2 = generate_sample_data(p2, n=50, seed=42)
        assert list(d1["label"]) == list(d2["label"])


# ── label_distribution ────────────────────────────────────────────────────────

class TestLabelDistribution:
    def test_returns_dict(self):
        df = pd.DataFrame({"label": ["positive", "negative", "positive"]})
        d  = label_distribution(df)
        assert isinstance(d, dict)

    def test_counts_correct(self):
        df = pd.DataFrame({"label": ["positive", "positive", "negative"]})
        d  = label_distribution(df)
        assert d["positive"] == 2
        assert d["negative"] == 1
