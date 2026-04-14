"""
Dataset loading, cleaning, splitting, and sample-generation helpers for the review pipeline.
"""


from __future__ import annotations

import re
import html
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    SAMPLE_DATA_PATH, SENTIMENT_LABELS, SENTIMENT_TO_IDX,
    bert_cfg, BERTConfig,
)


# ── Review dataclass ──────────────────────────────────────────────────────────

@dataclass
class Review:
    text:      str
    label:     str    # "negative" | "neutral" | "positive"
    label_idx: int
    source:    str = ""
    rating:    Optional[float] = None


# ── Loading ───────────────────────────────────────────────────────────────────

def load_csv(
    path:          Path | str,
    text_col:      str = "text",
    label_col:     str = "label",
    rating_col:    Optional[str] = None,
    source_col:    Optional[str] = None,
    max_rows:      Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a review CSV into a cleaned, validated DataFrame.

    Expects at minimum a text column and a label column.
    Labels must be in {"negative", "neutral", "positive"}.

    Args:
        path:       Path to the CSV file.
        text_col:   Name of the text column.
        label_col:  Name of the label column.
        rating_col: Optional numeric rating column (e.g. 1–5 stars).
        source_col: Optional source/platform column.
        max_rows:   Limit rows loaded (useful for quick testing).

    Returns:
        DataFrame with columns: text, label, label_idx, [rating], [source].

    Raises:
        ValueError: If required columns are missing or labels are invalid.
    """
    df = pd.read_csv(path, nrows=max_rows)

    _require_columns(df, [text_col, label_col], path)

    df = df.rename(columns={text_col: "text", label_col: "label"})
    if rating_col and rating_col in df.columns:
        df = df.rename(columns={rating_col: "rating"})
    if source_col and source_col in df.columns:
        df = df.rename(columns={source_col: "source"})

    # Validate labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    invalid = set(df["label"].unique()) - set(SENTIMENT_LABELS)
    if invalid:
        raise ValueError(
            f"Invalid labels found: {invalid}. "
            f"Expected one of: {SENTIMENT_LABELS}"
        )

    df["label_idx"] = df["label"].map(SENTIMENT_TO_IDX)
    df["text"]      = df["text"].astype(str).apply(clean_text)
    df              = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df              = df[df["text"].str.len() > 3]   # drop overly short reviews (e.g. "Ok.")

    return df


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise raw review text for NLP processing.

    Steps (order matters):
      1. Decode HTML entities  (e.g. &amp; → &)
      2. Strip HTML tags
      3. Normalise whitespace
      4. Strip leading/trailing whitespace

    Intentionally does NOT lowercase or remove punctuation —
    transformers handle casing internally, and punctuation
    carries sentiment signal.
    """
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\s+", " ", text)               # collapse whitespace
    return text.strip()


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_dataset(
    df:         pd.DataFrame,
    val_frac:   float = 0.15,
    test_frac:  float = 0.15,
    seed:       int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split preserving label proportions.

    Args:
        df:         Full cleaned DataFrame from load_csv().
        val_frac:   Fraction of data for validation.
        test_frac:  Fraction of data for test.
        seed:       Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    train_df, temp_df = train_test_split(
        df, test_size=val_frac + test_frac,
        stratify=df["label"], random_state=seed,
    )
    relative_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=seed,
    )
    return train_df, val_df, test_df


# ── PyTorch Dataset for DistilBERT ────────────────────────────────────────────

class ReviewDataset:
    """
    Tokenises a DataFrame of reviews for DistilBERT fine-tuning.

    Lazy tokenisation — encodings are computed on first access so
    the tokenizer can be swapped without reloading data.

    Usage:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        ds = ReviewDataset(train_df, tokenizer)
        sample = ds[0]   # {"input_ids": ..., "attention_mask": ..., "labels": ...}
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        tokenizer,
        cfg:        BERTConfig = bert_cfg,
    ) -> None:
        self.texts      = df["text"].tolist()
        self.labels     = df["label_idx"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = cfg.max_length
        self._encodings = None   # computed lazily on first __getitem__

    def _encode(self):
        self._encodings = self.tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        import torch
        if self._encodings is None:
            self._encode()
        return {
            "input_ids":      self._encodings["input_ids"][idx],
            "attention_mask": self._encodings["attention_mask"][idx],
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Label distribution helper ─────────────────────────────────────────────────

def label_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return count per sentiment label."""
    return df["label"].value_counts().to_dict()


# ── Sample data generator ─────────────────────────────────────────────────────

def generate_sample_data(
    path: Path | str = SAMPLE_DATA_PATH,
    n:    int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic review dataset for demo purposes.

    Returns a DataFrame and saves it to `path`.
    """
    rng = np.random.default_rng(seed)

    positive_templates = [
        "Absolutely love this product! {adj} quality and {adv} easy to use.",
        "Great {noun}. Works exactly as described. {bonus}",
        "Exceeded my expectations. {adj} and {adj2}. Would definitely recommend.",
        "Five stars! {bonus} The {noun} is top notch.",
        "Really happy with this purchase. {adj} build quality and fast shipping.",
        "Best {noun} I've ever bought. {adv} impressed with the quality.",
        "Fantastic product. Does exactly what it promises. {bonus}",
        "Solid {noun}, great value for money. {adv} satisfied.",
        "Works perfectly. Setup was quick and the {noun} feels premium.",
        "Very happy customer here. {adj} and reliable. Shipping was fast too.",
    ]
    neutral_templates = [
        "It's okay. Does the job but nothing special about the {noun}.",
        "Average product. {adj} in some ways but could be better.",
        "Works as expected. Not amazing but not bad either.",
        "Decent {noun} for the price. A few minor issues but usable.",
        "Mixed feelings. Good {noun} overall but the {part} needs improvement.",
        "It's fine. Does what it says but the quality is just average.",
        "Not bad, not great. The {noun} works but feels cheap.",
        "Acceptable for the price point. Some features are missing.",
        "The {noun} arrived on time and works. Nothing more to say.",
        "Mediocre. Expected more based on the reviews.",
    ]
    negative_templates = [
        "Terrible quality. The {noun} broke after {days} days. Very disappointed.",
        "Do not buy this! Complete waste of money. {complaint}.",
        "Awful product. Stopped working after one week. {complaint}.",
        "Very poor quality. The {noun} looks nothing like the photos.",
        "Disappointed. {complaint} and customer service was unhelpful.",
        "Worst purchase ever. {noun} arrived damaged and {complaint}.",
        "Horrible experience. {complaint}. Would give zero stars if I could.",
        "Returned immediately. The {noun} didn't work out of the box.",
        "Complete junk. Fell apart after {days} days. Avoid at all costs.",
        "Not as described. {complaint} and the shipping took forever.",
    ]

    adj_pool    = ["excellent", "outstanding", "impressive", "solid", "great", "premium"]
    adj2_pool   = ["reliable", "durable", "well-made", "sturdy", "lightweight", "compact"]
    adv_pool    = ["incredibly", "surprisingly", "genuinely", "extremely", "very"]
    noun_pool   = ["product", "item", "device", "unit", "gadget", "tool", "kit"]
    part_pool   = ["packaging", "manual", "cable", "connector", "finish", "design"]
    bonus_pool  = ["Shipping was fast.", "Packaging was great.", "Will buy again.", "Highly recommend."]
    complaint_p = ["packaging was terrible", "it stopped working", "it's not as advertised",
                   "quality control is nonexistent", "instructions were useless"]
    days_pool   = [2, 3, 5, 7, 10, 14]

    def fill(template: str) -> str:
        return template.format(
            adj       = rng.choice(adj_pool),
            adj2      = rng.choice(adj2_pool),
            adv       = rng.choice(adv_pool),
            noun      = rng.choice(noun_pool),
            part      = rng.choice(part_pool),
            bonus     = rng.choice(bonus_pool),
            complaint = rng.choice(complaint_p),
            days      = rng.choice(days_pool),
        )

    # Generate with realistic distribution: 50% positive, 25% neutral, 25% negative
    n_pos = int(n * 0.50)
    n_neu = int(n * 0.25)
    n_neg = n - n_pos - n_neu

    rows = []
    for _ in range(n_pos):
        rows.append({"text": fill(rng.choice(positive_templates)), "label": "positive",
                     "rating": float(rng.choice([4, 5]))})
    for _ in range(n_neu):
        rows.append({"text": fill(rng.choice(neutral_templates)),  "label": "neutral",
                     "rating": float(rng.choice([3]))})
    for _ in range(n_neg):
        rows.append({"text": fill(rng.choice(negative_templates)), "label": "negative",
                     "rating": float(rng.choice([1, 2]))})

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"✅ Generated {len(df)} sample reviews → {path}")
    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_columns(df: pd.DataFrame, cols: List[str], path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {path}. "
            f"Available: {list(df.columns)}"
        )
