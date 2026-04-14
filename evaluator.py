"""
evaluator.py
------------
Evaluation metrics and Plotly visualisations for NLP models.

Responsibilities:
  - evaluate(): compute accuracy, precision, recall, F1 per class
  - compare_models(): side-by-side BERT vs TF-IDF metrics table
  - plot_confusion_matrix(): Plotly heatmap
  - plot_sentiment_distribution(): pie/bar chart of label counts
  - plot_keyphrase_bar(): horizontal bar of top keyphrases
  - plot_aspect_radar(): radar chart of aspect sentiment scores
  - plot_model_comparison(): grouped bar comparing two models

No model loading. No training. No UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import SENTIMENT_LABELS

PLOTLY_TEMPLATE = "plotly_dark"

# Colour palette consistent across all charts
SENTIMENT_COLOURS = {
    "positive": "#6dfabd",
    "neutral":  "#fad96d",
    "negative": "#fa6d6d",
}


# ── Evaluation result ─────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    model_name:  str
    accuracy:    float
    precision:   Dict[str, float]
    recall:      Dict[str, float]
    f1:          Dict[str, float]
    macro_f1:    float
    conf_matrix: np.ndarray
    class_names: List[str]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for cls in self.class_names:
            rows.append({
                "class":     cls,
                "precision": round(self.precision[cls], 4),
                "recall":    round(self.recall[cls],    4),
                "f1":        round(self.f1[cls],        4),
            })
        rows.append({
            "class":     "macro avg",
            "precision": round(np.mean(list(self.precision.values())), 4),
            "recall":    round(np.mean(list(self.recall.values())),    4),
            "f1":        round(self.macro_f1, 4),
        })
        return pd.DataFrame(rows)

    def __str__(self) -> str:
        lines = [
            f"Model: {self.model_name}",
            f"Accuracy: {self.accuracy:.4f}  |  Macro F1: {self.macro_f1:.4f}",
            "",
        ]
        for cls in self.class_names:
            lines.append(
                f"  {cls:10s} — P={self.precision[cls]:.3f}  "
                f"R={self.recall[cls]:.3f}  F1={self.f1[cls]:.3f}"
            )
        return "\n".join(lines)


# ── Core evaluation function ──────────────────────────────────────────────────

def evaluate(
    true_labels: List[str],
    pred_labels: List[str],
    model_name:  str = "Model",
    class_names: List[str] = SENTIMENT_LABELS,
) -> EvalResult:
    """
    Compute classification metrics from ground-truth and predicted label lists.

    Args:
        true_labels: Ground-truth sentiment strings.
        pred_labels: Predicted sentiment strings.
        model_name:  Display name for the model.
        class_names: Ordered list of class labels.

    Returns:
        EvalResult with accuracy, per-class P/R/F1, and confusion matrix.
    """
    n   = len(class_names)
    cm  = np.zeros((n, n), dtype=int)
    idx = {c: i for i, c in enumerate(class_names)}

    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    accuracy   = cm.diagonal().sum() / cm.sum() if cm.sum() > 0 else 0.0
    precision  = {}
    recall     = {}
    f1         = {}

    for i, cls in enumerate(class_names):
        tp  = cm[i, i]
        fp  = cm[:, i].sum() - tp
        fn  = cm[i, :].sum() - tp
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precision[cls] = p
        recall[cls]    = r
        f1[cls]        = f

    macro_f1 = float(np.mean(list(f1.values())))
    return EvalResult(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=macro_f1,
        conf_matrix=cm,
        class_names=class_names,
    )


def compare_models(*results: EvalResult) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame for multiple EvalResults.

    Returns:
        DataFrame with rows: model | accuracy | macro_f1 | per-class F1s.
    """
    rows = []
    for r in results:
        row = {"model": r.model_name, "accuracy": r.accuracy, "macro_f1": r.macro_f1}
        for cls in r.class_names:
            row[f"f1_{cls}"] = r.f1[cls]
        rows.append(row)
    return pd.DataFrame(rows).round(4)


# ── Plotly chart builders ─────────────────────────────────────────────────────

def plot_confusion_matrix(result: EvalResult) -> go.Figure:
    """Heatmap of the confusion matrix."""
    fig = px.imshow(
        result.conf_matrix,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=result.class_names,
        y=result.class_names,
        color_continuous_scale="Blues",
        title=f"Confusion Matrix — {result.model_name}",
        text_auto=True,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def plot_sentiment_distribution(df: pd.DataFrame, label_col: str = "label") -> go.Figure:
    """Pie chart of sentiment label distribution."""
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["label", "count"]
    colours = [SENTIMENT_COLOURS.get(l, "#a0a0c0") for l in counts["label"]]
    fig = px.pie(
        counts, names="label", values="count",
        title="Sentiment Distribution",
        color="label",
        color_discrete_map=SENTIMENT_COLOURS,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig


def plot_keyphrase_bar(phrase_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top keyphrases by frequency."""
    if phrase_df.empty:
        return go.Figure().update_layout(title="No keyphrases found", template=PLOTLY_TEMPLATE)
    fig = px.bar(
        phrase_df.head(15),
        x="frequency", y="phrase",
        orientation="h",
        title="Top Keyphrases",
        color="frequency",
        color_continuous_scale="Blues",
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    return fig


def plot_aspect_sentiment(aspect_df: pd.DataFrame) -> go.Figure:
    """Bar chart: aspect mentions coloured by average sentiment score."""
    if aspect_df.empty:
        return go.Figure().update_layout(title="No aspect data", template=PLOTLY_TEMPLATE)

    colours = [
        SENTIMENT_COLOURS["positive"] if s > 0.1
        else SENTIMENT_COLOURS["negative"] if s < -0.1
        else SENTIMENT_COLOURS["neutral"]
        for s in aspect_df["avg_sentiment"]
    ]
    fig = go.Figure(go.Bar(
        x=aspect_df["aspect"],
        y=aspect_df["mention_count"],
        marker_color=colours,
        text=aspect_df["avg_sentiment"].round(2),
        textposition="outside",
    ))
    fig.update_layout(
        title="Aspect Mentions & Sentiment",
        xaxis_title="Aspect",
        yaxis_title="Mention Count",
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Grouped bar: accuracy and macro F1 per model."""
    fig = go.Figure()
    metrics = ["accuracy", "macro_f1"]
    colours = ["#7c6dfa", "#6dfabd"]
    for metric, colour in zip(metrics, colours):
        fig.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=comparison_df["model"],
            y=comparison_df[metric],
            marker_color=colour,
            text=comparison_df[metric].round(3),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Model Comparison",
        yaxis=dict(range=[0, 1.1]),
        template=PLOTLY_TEMPLATE,
    )
    return fig


def plot_confidence_histogram(predictions: List[dict]) -> go.Figure:
    """Histogram of prediction confidence scores."""
    confidences = [p["confidence"] for p in predictions]
    labels      = [p["label"] for p in predictions]
    df = pd.DataFrame({"confidence": confidences, "label": labels})
    fig = px.histogram(
        df, x="confidence", color="label",
        nbins=20, barmode="overlay",
        title="Prediction Confidence Distribution",
        color_discrete_map=SENTIMENT_COLOURS,
        opacity=0.75,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE)
    return fig
