"""
train.py
--------
Command-line training script for TF-IDF and DistilBERT classifiers.

Usage:
    # Train TF-IDF baseline (fast, no downloads)
    python train.py --model tfidf

    # Fine-tune DistilBERT (~300MB download on first run)
    python train.py --model bert

    # Train both and compare
    python train.py --model both

    # Use your own CSV
    python train.py --model tfidf --data path/to/reviews.csv \\
        --text-col review_text --label-col sentiment

    # Generate sample data first
    python train.py --model both --generate-data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    BERT_MODEL_DIR, DATA_DIR, MODELS_DIR, SAMPLE_DATA_PATH,
    TFIDF_MODEL_PATH, BERTConfig, TFIDFConfig, bert_cfg, tfidf_cfg,
)
from data_loader import generate_sample_data, load_csv, split_dataset
from evaluator import compare_models
from trainer import BERTTrainer, TFIDFTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train NLP sentiment classifiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model tfidf --generate-data
  python train.py --model bert  --epochs 3
  python train.py --model both  --data reviews.csv --text-col text --label-col sentiment
        """,
    )
    p.add_argument("--model",         choices=["tfidf", "bert", "both"], default="tfidf")
    p.add_argument("--data",          default=None,       help="Path to CSV (uses sample data if omitted)")
    p.add_argument("--text-col",      default="text",     help="Text column name (default: text)")
    p.add_argument("--label-col",     default="label",    help="Label column name (default: label)")
    p.add_argument("--max-rows",      type=int, default=None)
    p.add_argument("--epochs",        type=int, default=None,  help="Override num_epochs")
    p.add_argument("--batch-size",    type=int, default=None,  help="Override batch_size")
    p.add_argument("--generate-data", action="store_true",     help="Generate 500 synthetic reviews before training")
    p.add_argument("--n-samples",     type=int, default=500,   help="Rows when --generate-data is used")
    p.add_argument("--val-frac",      type=float, default=0.15)
    p.add_argument("--test-frac",     type=float, default=0.15)
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def load_data(args: argparse.Namespace):
    data_path = Path(args.data) if args.data else SAMPLE_DATA_PATH

    if not data_path.exists():
        if args.generate_data or not args.data:
            print("Generating sample data…")
            generate_sample_data(SAMPLE_DATA_PATH, n=args.n_samples, seed=args.seed)
        else:
            raise FileNotFoundError(
                f"Data file not found: {data_path}. "
                "Use --generate-data or provide a valid --data path."
            )

    df = load_csv(data_path, text_col=args.text_col, label_col=args.label_col, max_rows=args.max_rows)
    print(f"Loaded {len(df)} reviews from {data_path}")
    return split_dataset(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)


def train_tfidf(train_df, val_df, args) -> tuple:
    cfg = TFIDFConfig()
    return TFIDFTrainer(train_df, val_df, cfg=cfg).train()


def train_bert(train_df, val_df, args) -> tuple:
    cfg = BERTConfig(
        num_epochs=args.epochs    or bert_cfg.num_epochs,
        batch_size=args.batch_size or bert_cfg.batch_size,
    )
    return BERTTrainer(train_df, val_df, cfg=cfg).train()


def main() -> None:
    args = parse_args()

    if args.generate_data:
        print("Generating synthetic sample data…")
        generate_sample_data(SAMPLE_DATA_PATH, n=args.n_samples, seed=args.seed)

    train_df, val_df, test_df = load_data(args)

    results = []
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("tfidf", "both"):
        print("\n" + "=" * 60)
        print("TRAINING TF-IDF BASELINE")
        print("=" * 60)
        _, result_tfidf = train_tfidf(train_df, val_df, args)
        results.append(result_tfidf)

    if args.model in ("bert", "both"):
        print("\n" + "=" * 60)
        print("FINE-TUNING DISTILBERT")
        print("=" * 60)
        _, result_bert = train_bert(train_df, val_df, args)
        results.append(result_bert)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        comparison = compare_models(*results)
        print(comparison.to_string(index=False))
        comp_path = MODELS_DIR / "model_comparison.json"
        comp_path.write_text(comparison.to_json(orient="records", indent=2))
        print(f"\nComparison saved to {comp_path}")

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
