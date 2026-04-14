"""
models/bert_classifier.py
--------------------------
DistilBERT-based sentiment classifier.

Why DistilBERT?
  - 40% smaller than BERT-base, 60% faster, 97% of the performance
  - Designed for CPU inference — practical on a laptop
  - Fine-tuning 3 epochs on 500 reviews takes ~10 minutes on CPU

Architecture:
  DistilBERT encoder → [CLS] token → Dropout → Linear(768 → num_labels)

Uses HuggingFace Transformers + a thin training wrapper.
No UI. No data loading. No evaluation metrics (those are in evaluator.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
)

from config import (
    BERT_MODEL_DIR, LABEL_MAP_PATH, SENTIMENT_LABELS,
    BERTConfig, bert_cfg,
)


class BERTSentimentClassifier:
    """
    Wraps HuggingFace DistilBERT for 3-class sentiment classification.

    Provides a clean predict() / predict_batch() API that hides all
    tokenisation and tensor management from the caller.

    Usage:
        clf = BERTSentimentClassifier()
        clf.load("models/bert_sentiment/")
        result = clf.predict("This product is excellent!")
    """

    def __init__(self, cfg: BERTConfig = bert_cfg) -> None:
        self.cfg       = cfg
        self.device    = torch.device(cfg.device)
        self.tokenizer = None
        self.model     = None

    # ── Build (for training) ──────────────────────────────────────────────────

    def build(self) -> "BERTSentimentClassifier":
        """Initialise tokenizer and model from HuggingFace Hub."""
        print(f"[BERTSentimentClassifier] Loading {self.cfg.model_name}…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name,
            num_labels=self.cfg.num_labels,
        )
        self.model.to(self.device)
        return self

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, directory: Path | str = BERT_MODEL_DIR) -> None:
        """Save model, tokenizer, and label map to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        # Save label map
        label_map = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
        (directory / "label_map.json").write_text(json.dumps(label_map, indent=2))
        print(f"[BERTSentimentClassifier] Saved to {directory}")

    @classmethod
    def load(cls, directory: Path | str = BERT_MODEL_DIR, cfg: Optional[BERTConfig] = None) -> "BERTSentimentClassifier":
        """Load a fine-tuned model from directory."""
        directory = Path(directory)
        instance  = cls(cfg=cfg or bert_cfg)
        instance.tokenizer = AutoTokenizer.from_pretrained(directory)
        instance.model     = AutoModelForSequenceClassification.from_pretrained(directory)
        instance.model.to(instance.device)
        instance.model.eval()
        print(f"[BERTSentimentClassifier] Loaded from {directory}")
        return instance

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single review text.

        Returns:
            dict with keys: label (str), label_idx (int),
                            confidence (float), probabilities (dict[str, float])
        """
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Classify a list of review texts in one forward pass.

        Returns:
            List of dicts, one per input text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .build() or .load() first.")

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            logits = self.model(**encodings).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for prob_row in probs:
            label_idx  = int(prob_row.argmax())
            results.append({
                "label":         SENTIMENT_LABELS[label_idx],
                "label_idx":     label_idx,
                "confidence":    float(prob_row[label_idx]),
                "probabilities": {
                    lbl: float(prob_row[i])
                    for i, lbl in enumerate(SENTIMENT_LABELS)
                },
            })
        return results

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        if self.model is None:
            return {}
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
