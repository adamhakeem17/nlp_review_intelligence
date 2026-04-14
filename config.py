"""
config.py
---------
Central configuration for NLP Review Intelligence.

All hyperparameters, model names, paths, and label definitions live here.
No module in the codebase uses magic strings or numbers directly —
they all import from this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
LOGS_DIR    = ROOT / "logs"

# Saved artefacts
BERT_MODEL_DIR   = MODELS_DIR / "bert_sentiment"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_sentiment.joblib"
LABEL_MAP_PATH   = MODELS_DIR / "label_map.json"
SAMPLE_DATA_PATH = DATA_DIR   / "sample" / "reviews.csv"


# ── Sentiment labels ──────────────────────────────────────────────────────────

SENTIMENT_LABELS: List[str] = ["negative", "neutral", "positive"]
SENTIMENT_TO_IDX: Dict[str, int] = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
IDX_TO_SENTIMENT: Dict[int, str] = {i: l for i, l in enumerate(SENTIMENT_LABELS)}

SENTIMENT_EMOJI: Dict[str, str] = {
    "positive": "🟢",
    "neutral":  "🟡",
    "negative": "🔴",
}


# ── DistilBERT transformer config ─────────────────────────────────────────────

@dataclass
class BERTConfig:
    """
    DistilBERT is ~40% smaller and 60% faster than BERT-base while
    retaining 97% of its NLU performance — ideal for CPU inference.
    """
    model_name:       str   = "distilbert-base-uncased"
    num_labels:       int   = 3           # negative / neutral / positive
    max_length:       int   = 128         # token limit; 128 is sufficient for reviews
    batch_size:       int   = 8           # small batch for CPU memory
    num_epochs:       int   = 3           # fine-tuning converges fast
    learning_rate:    float = 2e-5        # standard fine-tune LR
    weight_decay:     float = 0.01
    warmup_ratio:     float = 0.1         # fraction of steps for LR warm-up
    patience:         int   = 2           # early stopping patience
    fp16:             bool  = False       # always False on CPU
    device:           str   = "cpu"
    save_steps:       int   = 100
    eval_steps:       int   = 100
    logging_steps:    int   = 50


# ── TF-IDF + LogisticRegression baseline config ───────────────────────────────

@dataclass
class TFIDFConfig:
    """
    Classical NLP baseline — fast to train, interpretable,
    and a strong comparison point for the transformer.
    """
    max_features:     int   = 10_000
    ngram_range:      tuple = (1, 2)      # unigrams + bigrams
    sublinear_tf:     bool  = True        # log-scale TF
    min_df:           int   = 1           # minimum document frequency (1 keeps everything)
    max_df:           float = 1.0         # document frequency upper bound (1.0 keeps everything)
    # Logistic Regression
    C:                float = 1.0
    max_iter:         int   = 500
    solver:           str   = "lbfgs"
    class_weight:     str   = "balanced"


# ── Topic extraction config ───────────────────────────────────────────────────

@dataclass
class TopicConfig:
    """Settings for spaCy-based topic and entity extraction."""
    spacy_model:      str   = "en_core_web_sm"   # small CPU model
    top_n_topics:     int   = 10
    min_phrase_freq:  int   = 2
    # Aspect categories to look for in review text
    aspect_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "quality":   ["quality", "build", "durable", "material", "solid", "cheap", "flimsy"],
        "price":     ["price", "cost", "value", "expensive", "cheap", "worth", "affordable"],
        "shipping":  ["shipping", "delivery", "arrived", "package", "fast", "slow", "damaged"],
        "service":   ["service", "support", "staff", "helpful", "rude", "response", "team"],
        "usability": ["easy", "difficult", "intuitive", "confusing", "setup", "install", "use"],
    })


# ── Summarisation config ──────────────────────────────────────────────────────

@dataclass
class SummaryConfig:
    """Settings for extractive and abstractive summarisation."""
    extractive_n_sentences: int = 3        # sentences to extract
    # For abstractive: facebook/bart-large-cnn is strong but large.
    # We default to a smaller model suitable for CPU.
    abstractive_model:  str = "sshleifer/distilbart-cnn-6-6"
    max_input_tokens:   int = 512
    max_output_tokens:  int = 128
    min_output_tokens:  int = 30
    num_beams:          int = 2            # beam search width (lower = faster on CPU)


# ── Inference config ──────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    confidence_threshold:   float = 0.70   # below this → flag as uncertain
    batch_size:             int   = 16


# ── Default instances ─────────────────────────────────────────────────────────

bert_cfg      = BERTConfig()
tfidf_cfg     = TFIDFConfig()
topic_cfg     = TopicConfig()
summary_cfg   = SummaryConfig()
inference_cfg = InferenceConfig()
