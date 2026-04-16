"""
Streamlit UI entry point for the NLP Review Intelligence workflow.

All model, data handling, and summarisation utilities live in the modules below; this file only wires them into Streamlit.

Helpers referenced here:

    config.py           -> hyperparameters, paths, and label metadata
    data_loader.py      -> CSV loading, cleaning, sampling, ReviewDataset
    bert_classifier.py  -> DistilBERT classifier wrapper
    tfidf_classifier.py -> TF-IDF + LR classifier wrapper
    trainer.py          -> training loops, early stopping, evaluation helpers
    evaluator.py        -> metrics, model comparison, Plotly chart builders
    topic_extractor.py  -> spaCy-based keyphrase, entity, and aspect extraction
    summariser.py       -> extractive and optional abstractive summarisation
    predictor.py        -> ReviewPredictor and CorpusAnalyser API

Run `streamlit run app.py` after training (for example `python train.py --model tfidf --generate-data`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BERT_MODEL_DIR, TFIDF_MODEL_PATH, SENTIMENT_EMOJI,
    SAMPLE_DATA_PATH,
)
from data_loader import clean_text, generate_sample_data, load_csv
from evaluator import (
    plot_aspect_sentiment, plot_confidence_histogram,
    plot_keyphrase_bar, plot_sentiment_distribution,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Review Intelligence",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background: #0d0d14; color: #e0e0f0; }
    .sentiment-positive { background: rgba(109,250,189,0.1); border: 1px solid #6dfabd;
                          border-radius: 8px; padding: 14px; text-align: center; }
    .sentiment-negative { background: rgba(250,109,109,0.1); border: 1px solid #fa6d6d;
                          border-radius: 8px; padding: 14px; text-align: center; }
    .sentiment-neutral  { background: rgba(250,217,109,0.1); border: 1px solid #fad96d;
                          border-radius: 8px; padding: 14px; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_predictor(model_choice: str):
    from predictor import ReviewPredictor
    try:
        if model_choice == "DistilBERT" and Path(BERT_MODEL_DIR).exists():
            return ReviewPredictor.from_bert()
        elif Path(TFIDF_MODEL_PATH).exists():
            return ReviewPredictor.from_tfidf()
        return None
    except Exception as e:
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.title("💬 NLP Review Intelligence")
st.caption(
    "**DistilBERT** fine-tuned sentiment classifier · "
    "**TF-IDF + LogReg** interpretable baseline · "
    "**spaCy** topic & entity extraction · "
    "Extractive summarisation · 100% CPU"
)

tab_single, tab_batch, tab_train, tab_about = st.tabs([
    "💬 Single Review",
    "📊 Batch Analysis",
    "🏋️ Train Models",
    "ℹ️ About",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single review
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Analyse a Review")
        model_choice = st.selectbox("Model", ["TF-IDF + LogReg (fast)", "DistilBERT (accurate)"])
        use_bert     = "DistilBERT" in model_choice

        sample_reviews = {
            "Positive": "Absolutely love this product! Excellent quality and incredibly easy to use. Will definitely buy again.",
            "Negative": "Terrible quality. Broke after 3 days. Complete waste of money — avoid at all costs.",
            "Neutral":  "It's okay. Does the job but nothing special. Average build quality for the price.",
        }
        sample_key  = st.selectbox("Load a sample", ["— write your own —"] + list(sample_reviews))
        review_text = st.text_area(
            "Review text",
            value=sample_reviews.get(sample_key, ""),
            height=140,
            placeholder="Type or paste a customer review here…",
        )
        analyse_btn = st.button("🔍 Analyse", type="primary")

    with col_right:
        model_key = "bert" if use_bert else "tfidf"
        predictor = load_predictor(model_key)

        if not predictor:
            st.warning("⚠️ No trained model found. Go to the **Train Models** tab first.")

        if analyse_btn and review_text.strip() and predictor:
            from predictor import ReviewPredictor
            with st.spinner("Analysing…"):
                pred = predictor.predict(review_text)

            css   = f"sentiment-{pred.label}"
            emoji = pred.emoji
            st.markdown(
                f'<div class="{css}">'
                f'<h2>{emoji} {pred.label.upper()}</h2>'
                f'<p>Confidence: <strong>{pred.confidence:.1%}</strong></p>'
                f'{"<p>⚠️ Low confidence — treat with caution</p>" if pred.flagged else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.divider()
            st.subheader("Class Probabilities")
            for label, prob in pred.probabilities.items():
                colour = {"positive": "green", "neutral": "orange", "negative": "red"}[label]
                st.progress(prob, text=f"{SENTIMENT_EMOJI[label]} {label}: {prob:.1%}")

            # Topic extraction for single review
            try:
                from topic_extractor import KeyphraseExtractor, AspectAnalyser
                kpe     = KeyphraseExtractor()
                phrases = kpe.extract(review_text)
                if phrases:
                    st.divider()
                    st.subheader("🔑 Key Phrases")
                    st.write(" · ".join(f"`{p}`" for p in phrases[:8]))
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📊 Batch Review Analysis")
    st.caption("Upload a CSV or use the built-in sample dataset.")

    col_ctrl, col_main = st.columns([1, 2])

    with col_ctrl:
        use_sample = st.toggle("Use sample dataset", value=True)
        if not use_sample:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            text_col = st.text_input("Text column", "text")
            label_col_in = st.text_input("Label column (optional)", "label")
        model_b_choice = st.selectbox("Model", ["TF-IDF + LogReg", "DistilBERT"], key="model_b")
        max_rows       = st.slider("Max rows to process", 50, 2000, 200, 50)
        run_btn        = st.button("🚀 Run Analysis", type="primary")

    if run_btn:
        predictor_b = load_predictor("bert" if "DistilBERT" in model_b_choice else "tfidf")
        if not predictor_b:
            with col_main:
                st.warning("⚠️ No trained model found. Go to the **Train Models** tab.")
        else:
            with col_main:
                with st.spinner("Loading data…"):
                    if use_sample:
                        if not SAMPLE_DATA_PATH.exists():
                            generate_sample_data(SAMPLE_DATA_PATH, n=500)
                        df = load_csv(SAMPLE_DATA_PATH, max_rows=max_rows)
                    else:
                        if uploaded:
                            df = load_csv(uploaded, text_col=text_col, max_rows=max_rows)
                        else:
                            st.warning("Upload a CSV or enable the sample dataset.")
                            st.stop()

                with st.spinner(f"Classifying {len(df)} reviews…"):
                    from predictor import CorpusAnalyser
                    analyser = CorpusAnalyser(predictor_b)
                    analysis = analyser.analyse(df)

                result_df = analysis.df

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Reviews",         len(result_df))
                k2.metric("Positive",        analysis.label_counts.get("positive", 0))
                k3.metric("Negative",        analysis.label_counts.get("negative", 0))
                k4.metric("Avg Confidence",  f"{analysis.avg_confidence:.1%}")

                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        plot_sentiment_distribution(result_df, "predicted_label"),
                        use_container_width=True,
                    )
                with c2:
                    preds_list = result_df[["predicted_label","confidence"]].rename(
                        columns={"predicted_label":"label"}).to_dict("records")
                    st.plotly_chart(
                        plot_confidence_histogram(preds_list),
                        use_container_width=True,
                    )

                # Topic charts
                if analysis.topic_summary:
                    ts = analysis.topic_summary
                    c3, c4 = st.columns(2)
                    with c3:
                        st.plotly_chart(plot_keyphrase_bar(ts.top_keyphrases), use_container_width=True)
                    with c4:
                        st.plotly_chart(plot_aspect_sentiment(ts.aspect_sentiment), use_container_width=True)

                # Summary
                st.divider()
                st.subheader("📝 Extractive Summary")
                st.info(analysis.summary)

                with st.expander("Summary by Sentiment"):
                    for lbl, summ in analysis.by_sentiment.items():
                        st.markdown(f"**{SENTIMENT_EMOJI[lbl]} {lbl.title()}:** {summ}")

                # Results table
                st.divider()
                st.subheader("Review Results")
                display = result_df[["text","predicted_label","confidence","flagged"]].head(100)
                st.dataframe(display, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 Download Results CSV",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_results.csv", mime="text/csv",
                )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Train
# ═══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.subheader("🏋️ Train Models")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**TF-IDF Baseline** *(trains in seconds)*")
        n_samples  = st.slider("Sample reviews to generate", 100, 2000, 500, 100)
        train_tfidf_btn = st.button("Train TF-IDF", type="primary")

    with col_b:
        st.markdown("**DistilBERT** *(~10 min on CPU, ~300MB download)*")
        bert_epochs = st.slider("Fine-tuning epochs", 1, 5, 2)
        train_bert_btn  = st.button("Fine-tune DistilBERT")

    if train_tfidf_btn:
        import subprocess, sys
        progress = st.progress(0, "Generating data…")
        subprocess.run([sys.executable, "train.py", "--model", "tfidf",
                        "--generate-data", "--n-samples", str(n_samples)])
        progress.progress(100, "✅ TF-IDF trained!")
        st.cache_resource.clear()
        st.success("TF-IDF model trained. Refresh other tabs to use it.")

    if train_bert_btn:
        import subprocess, sys
        progress = st.progress(0, "Fine-tuning DistilBERT (this takes a few minutes)…")
        subprocess.run([sys.executable, "train.py", "--model", "bert",
                        "--generate-data", "--epochs", str(bert_epochs)])
        progress.progress(100, "✅ DistilBERT fine-tuned!")
        st.cache_resource.clear()
        st.success("DistilBERT fine-tuned. Refresh other tabs to use it.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About this project")
    st.markdown("""
    ### Architecture

    **Model 1 — DistilBERT Sentiment Classifier**
    - Pre-trained: `distilbert-base-uncased` (HuggingFace)
    - Fine-tuned with 3-class head: negative / neutral / positive
    - 40% smaller and 60% faster than BERT-base, 97% of performance
    - ~66M parameters · < 500ms inference per review on CPU

    **Model 2 — TF-IDF + Logistic Regression Baseline**
    - 10,000 features, unigrams + bigrams, balanced class weights
    - Fully interpretable: `top_features("negative")` shows key words
    - Trains in seconds · No internet connection required

    **Topic Extraction (spaCy)**
    - Noun-phrase keyphrase extraction across the review corpus
    - Named entity recognition: brands, products, locations
    - Aspect-based analysis: quality / price / shipping / service / usability

    **Summarisation**
    - Extractive: TF-IDF sentence ranking — fast, no model needed
    - Abstractive: DistilBART (optional) — fluent generated summaries

    ### Module Structure
    ```
    config.py          — All hyperparameters, paths, label maps
    data_loader.py     — CSV loading, text cleaning, splits, sample generator
    models/
      bert_classifier.py  — DistilBERT wrapper (build, predict, save/load)
      tfidf_classifier.py — TF-IDF + LR pipeline (fit, predict, top_features)
    trainer.py         — BERTTrainer + TFIDFTrainer + metrics callbacks
    evaluator.py       — evaluate(), compare_models(), Plotly chart builders
    topic_extractor.py — KeyphraseExtractor, EntityExtractor, AspectAnalyser
    summariser.py      — ExtractiveSummariser + AbstractiveSummariser
    predictor.py       — ReviewPredictor + CorpusAnalyser (full pipeline)
    train.py           — CLI: --model tfidf|bert|both + all flags
    app.py             — Streamlit UI (no business logic)
    ```

    ### Business Use Case
    - **E-commerce:** Automatically tag thousands of product reviews by sentiment
    - **SaaS:** Monitor customer feedback for churn signals and feature requests
    - **Hospitality:** Identify what guests praise or complain about (aspects)
    - **Any domain:** Drop in your own CSV and get instant NLP insights
    """)
