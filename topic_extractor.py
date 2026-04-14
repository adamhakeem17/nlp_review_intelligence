"""
topic_extractor.py
------------------
Topic and entity extraction from review text using spaCy.

Responsibilities:
  - KeyphraseExtractor: noun-phrase frequency analysis
  - EntityExtractor: named entity recognition (products, brands, locations)
  - AspectAnalyser: maps review sentences to predefined business aspects
    (quality, price, shipping, service, usability)
  - summarise_topics(): aggregate topic insights across a review corpus

No sentiment logic. No model training. No UI.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import TopicConfig, topic_cfg


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class TopicResult:
    keyphrases:      List[Tuple[str, int]]   # (phrase, frequency)
    entities:        List[Tuple[str, str]]   # (text, entity_type)
    aspects:         Dict[str, List[str]]    # aspect_name → matching sentences


@dataclass
class CorpusTopicSummary:
    top_keyphrases:     pd.DataFrame   # phrase | frequency | sentiment_score
    top_entities:       pd.DataFrame   # entity | type | count
    aspect_sentiment:   pd.DataFrame   # aspect | mention_count | avg_sentiment


# ── spaCy loader (lazy — only import when called) ─────────────────────────────

_nlp = None   # module-level cache

def _get_nlp(model_name: str = topic_cfg.spacy_model):
    """Load spaCy model once and cache it."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load(model_name)
        except OSError:
            raise OSError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )
    return _nlp


# ── Keyphrase extractor ───────────────────────────────────────────────────────

class KeyphraseExtractor:
    """
    Extracts noun phrases from review text using spaCy's dependency parser.

    Noun phrases (e.g. "battery life", "easy setup", "poor quality") carry
    the key topics customers discuss.
    """

    def __init__(self, cfg: TopicConfig = topic_cfg) -> None:
        self.cfg = cfg

    def extract(self, text: str) -> List[str]:
        """Return a list of noun phrases from a single text."""
        nlp = _get_nlp(self.cfg.spacy_model)
        doc = nlp(text)
        return [
            chunk.text.lower().strip()
            for chunk in doc.noun_chunks
            if len(chunk.text.split()) >= 1 and len(chunk.text) > 2
        ]

    def extract_corpus(self, texts: List[str]) -> List[Tuple[str, int]]:
        """
        Extract and count keyphrases across a corpus.

        Returns:
            List of (phrase, count) sorted by frequency descending.
        """
        nlp    = _get_nlp(self.cfg.spacy_model)
        counts: Counter = Counter()

        for doc in nlp.pipe(texts, batch_size=32):
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()
                if 2 <= len(phrase) <= 40:
                    counts[phrase] += 1

        return [
            (phrase, count)
            for phrase, count in counts.most_common(self.cfg.top_n_topics * 3)
            if count >= self.cfg.min_phrase_freq
        ][: self.cfg.top_n_topics]


# ── Entity extractor ──────────────────────────────────────────────────────────

class EntityExtractor:
    """
    Extracts named entities (ORG, PRODUCT, GPE, PERSON) from reviews.

    Useful for identifying brand mentions, product names, and locations.
    """

    ENTITY_TYPES = {"ORG", "PRODUCT", "GPE", "PERSON", "FAC", "EVENT"}

    def __init__(self, cfg: TopicConfig = topic_cfg) -> None:
        self.cfg = cfg

    def extract(self, text: str) -> List[Tuple[str, str]]:
        """Return list of (entity_text, entity_type) from a single text."""
        nlp = _get_nlp(self.cfg.spacy_model)
        doc = nlp(text)
        return [
            (ent.text, ent.label_)
            for ent in doc.ents
            if ent.label_ in self.ENTITY_TYPES
        ]

    def extract_corpus(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract and count entities across a corpus.

        Returns:
            DataFrame with columns: entity, type, count.
        """
        nlp    = _get_nlp(self.cfg.spacy_model)
        counts: Counter = Counter()
        types:  Dict[str, str] = {}

        for doc in nlp.pipe(texts, batch_size=32):
            for ent in doc.ents:
                if ent.label_ in self.ENTITY_TYPES:
                    key        = ent.text.strip()
                    counts[key] += 1
                    types[key]   = ent.label_

        if not counts:
            return pd.DataFrame(columns=["entity", "type", "count"])

        return pd.DataFrame([
            {"entity": k, "type": types[k], "count": v}
            for k, v in counts.most_common(20)
        ])


# ── Aspect analyser ───────────────────────────────────────────────────────────

class AspectAnalyser:
    """
    Maps review sentences to predefined business aspects using keyword matching.

    Aspects: quality, price, shipping, service, usability
    Each sentence is assigned to the first matching aspect.

    This is a lightweight rule-based approach — for production, consider
    training an aspect classifier on labelled sentence data.
    """

    def __init__(self, cfg: TopicConfig = topic_cfg) -> None:
        self.cfg = cfg

    def analyse(self, text: str) -> Dict[str, List[str]]:
        """
        Split text into sentences and assign each to an aspect.

        Returns:
            Dict mapping aspect_name → list of sentences mentioning that aspect.
        """
        nlp     = _get_nlp(self.cfg.spacy_model)
        doc     = nlp(text)
        result  = {aspect: [] for aspect in self.cfg.aspect_keywords}

        for sent in doc.sents:
            sent_lower = sent.text.lower()
            for aspect, keywords in self.cfg.aspect_keywords.items():
                if any(kw in sent_lower for kw in keywords):
                    result[aspect].append(sent.text.strip())
                    break   # assign to first matching aspect only

        return {k: v for k, v in result.items() if v}   # drop empty aspects

    def analyse_corpus(
        self,
        df: pd.DataFrame,
        sentiment_col: str = "predicted_label",
    ) -> pd.DataFrame:
        """
        Aggregate aspect mentions and average sentiment across a corpus.

        Args:
            df:            DataFrame with "text" and sentiment_col columns.
            sentiment_col: Name of the predicted sentiment column.

        Returns:
            DataFrame: aspect | mention_count | avg_sentiment_score
        """
        sentiment_map = {"negative": -1, "neutral": 0, "positive": 1}

        rows: List[dict] = []
        for aspect in self.cfg.aspect_keywords:
            mentions  = 0
            sentiment = 0.0
            for _, row in df.iterrows():
                assigned = self.analyse(str(row["text"]))
                if aspect in assigned:
                    mentions  += len(assigned[aspect])
                    sentiment += sentiment_map.get(str(row.get(sentiment_col, "neutral")), 0)
            if mentions > 0:
                rows.append({
                    "aspect":         aspect,
                    "mention_count":  mentions,
                    "avg_sentiment":  round(sentiment / mentions, 2),
                })

        return pd.DataFrame(rows).sort_values("mention_count", ascending=False).reset_index(drop=True)


# ── Corpus summariser ─────────────────────────────────────────────────────────

def summarise_topics(
    df:             pd.DataFrame,
    sentiment_col:  str = "predicted_label",
    cfg:            TopicConfig = topic_cfg,
) -> CorpusTopicSummary:
    """
    Run full topic analysis across a DataFrame of reviews.

    Args:
        df:            DataFrame with "text" and sentiment_col.
        sentiment_col: Column name for predicted sentiment labels.
        cfg:           TopicConfig settings.

    Returns:
        CorpusTopicSummary with keyphrases, entities, and aspect sentiment.
    """
    texts = df["text"].tolist()

    kpe    = KeyphraseExtractor(cfg)
    ee     = EntityExtractor(cfg)
    aa     = AspectAnalyser(cfg)

    phrases  = kpe.extract_corpus(texts)
    entities = ee.extract_corpus(texts)
    aspects  = aa.analyse_corpus(df, sentiment_col)

    phrase_df = pd.DataFrame(phrases, columns=["phrase", "frequency"])

    return CorpusTopicSummary(
        top_keyphrases=phrase_df,
        top_entities=entities,
        aspect_sentiment=aspects,
    )
