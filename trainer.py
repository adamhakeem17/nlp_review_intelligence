"""
trainer.py
----------
Training orchestration for the BERT and TF-IDF classifiers.

Responsibilities:
  - BERTTrainer: fine-tune DistilBERT using HuggingFace Trainer API
  - TFIDFTrainer: fit the sklearn pipeline and log metrics
  - Both trainers save the best checkpoint and return an EvalResult

No model definition. No data loading. No UI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    BERT_MODEL_DIR, TFIDF_MODEL_PATH,
    BERTConfig, TFIDFConfig,
    bert_cfg, tfidf_cfg,
)
from data_loader import ReviewDataset, label_distribution
from evaluator import EvalResult, evaluate
from models.bert_classifier import BERTSentimentClassifier
from models.tfidf_classifier import TFIDFSentimentClassifier


# ── TF-IDF trainer ────────────────────────────────────────────────────────────

class TFIDFTrainer:
    """
    Fits the TF-IDF + Logistic Regression pipeline and evaluates it.

    Usage:
        trainer = TFIDFTrainer(train_df, val_df)
        model, result = trainer.train()
    """

    def __init__(
        self,
        train_df:  pd.DataFrame,
        val_df:    pd.DataFrame,
        cfg:       TFIDFConfig = tfidf_cfg,
        save_path: Path = TFIDF_MODEL_PATH,
    ) -> None:
        self.train_df  = train_df
        self.val_df    = val_df
        self.cfg       = cfg
        self.save_path = save_path

    def train(self) -> tuple[TFIDFSentimentClassifier, EvalResult]:
        """
        Fit the pipeline, evaluate on val_df, save model, and return results.
        """
        print(f"\n{'='*60}")
        print("Training TF-IDF + Logistic Regression baseline")
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)}")
        print(f"Label distribution: {label_distribution(self.train_df)}")
        print(f"{'='*60}")

        t0  = time.time()
        clf = TFIDFSentimentClassifier(cfg=self.cfg)
        clf.fit(self.train_df["text"].tolist(), self.train_df["label"].tolist())

        # Evaluate on validation set
        preds   = clf.predict_batch(self.val_df["text"].tolist())
        pred_labels = [p["label"] for p in preds]
        true_labels = self.val_df["label"].tolist()

        result = evaluate(true_labels, pred_labels, model_name="TF-IDF + LogReg")
        elapsed = time.time() - t0

        print(result)
        print(f"Training time: {elapsed:.1f}s")

        clf.save(self.save_path)
        return clf, result


# ── BERT trainer ──────────────────────────────────────────────────────────────

class BERTTrainer:
    """
    Fine-tunes DistilBERT using HuggingFace Trainer.

    Uses the Trainer API for clean integration with:
    - Early stopping via EarlyStoppingCallback
    - Automatic best-model checkpoint saving
    - Evaluation at each epoch

    Usage:
        trainer = BERTTrainer(train_df, val_df)
        model, result = trainer.train()
    """

    def __init__(
        self,
        train_df:   pd.DataFrame,
        val_df:     pd.DataFrame,
        cfg:        BERTConfig = bert_cfg,
        save_dir:   Path = BERT_MODEL_DIR,
    ) -> None:
        self.train_df = train_df
        self.val_df   = val_df
        self.cfg      = cfg
        self.save_dir = Path(save_dir)

    def train(self) -> tuple[BERTSentimentClassifier, EvalResult]:
        """
        Fine-tune DistilBERT and return (model, EvalResult on val set).
        """
        from transformers import (
            TrainingArguments, Trainer,
            EarlyStoppingCallback,
        )

        print(f"\n{'='*60}")
        print(f"Fine-tuning {self.cfg.model_name}")
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)}")
        print(f"Epochs: {self.cfg.num_epochs}  |  Batch: {self.cfg.batch_size}")
        print(f"{'='*60}")

        clf = BERTSentimentClassifier(cfg=self.cfg).build()

        train_ds = ReviewDataset(self.train_df, clf.tokenizer, self.cfg)
        val_ds   = ReviewDataset(self.val_df,   clf.tokenizer, self.cfg)

        training_args = TrainingArguments(
            output_dir=             str(self.save_dir / "checkpoints"),
            num_train_epochs=       self.cfg.num_epochs,
            per_device_train_batch_size= self.cfg.batch_size,
            per_device_eval_batch_size=  self.cfg.batch_size * 2,
            learning_rate=          self.cfg.learning_rate,
            weight_decay=           self.cfg.weight_decay,
            warmup_ratio=           self.cfg.warmup_ratio,
            evaluation_strategy=    "epoch",
            save_strategy=          "epoch",
            load_best_model_at_end= True,
            metric_for_best_model=  "eval_loss",
            fp16=                   self.cfg.fp16,
            logging_steps=          self.cfg.logging_steps,
            report_to=              "none",   # disable wandb/mlflow
            no_cuda=                True,
        )

        trainer = Trainer(
            model=          clf.model,
            args=           training_args,
            train_dataset=  train_ds,
            eval_dataset=   val_ds,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.cfg.patience)],
        )

        trainer.train()
        clf.model = trainer.model   # best checkpoint

        # Final evaluation
        preds       = clf.predict_batch(self.val_df["text"].tolist())
        pred_labels = [p["label"] for p in preds]
        true_labels = self.val_df["label"].tolist()
        result      = evaluate(true_labels, pred_labels, model_name="DistilBERT")

        print(result)
        clf.save(self.save_dir)
        return clf, result

    @staticmethod
    def _compute_metrics(eval_pred) -> dict:
        """HuggingFace Trainer metric callback."""
        logits, labels = eval_pred
        preds   = np.argmax(logits, axis=-1)
        correct = (preds == labels).sum()
        return {"accuracy": correct / len(labels)}
