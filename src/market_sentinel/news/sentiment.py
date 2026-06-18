"""Analyse de sentiment des actualités financières.

Deux backends :
  - "lexicon" (par défaut) : analyse lexicale légère, sans dépendance lourde,
    basée sur un dictionnaire financier extensible. Idéal pour démarrer.
  - "finbert" (optionnel) : modèle NLP FinBERT via `transformers`. Beaucoup
    plus précis mais nécessite transformers + torch (voir requirements-optional).

Chaque texte reçoit : un score [-1, 1], un label (5 niveaux), une confiance
[0, 1] et un score d'impact [0, 1].
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Lexique financier minimal mais extensible (poids dans [-1, 1]).
POSITIVE = {
    "beat": 0.8, "beats": 0.8, "surge": 0.9, "surges": 0.9, "soar": 0.9,
    "record": 0.7, "growth": 0.6, "profit": 0.6, "upgrade": 0.8,
    "outperform": 0.8, "bullish": 0.8, "strong": 0.6, "gain": 0.6,
    "gains": 0.6, "rally": 0.8, "rise": 0.5, "rises": 0.5, "expansion": 0.6,
    "approval": 0.7, "approved": 0.7, "partnership": 0.5, "dividend": 0.4,
    "buyback": 0.6, "raised": 0.6, "exceeds": 0.8, "robust": 0.6, "boom": 0.7,
}
NEGATIVE = {
    "miss": -0.8, "misses": -0.8, "plunge": -0.9, "plunges": -0.9,
    "crash": -1.0, "loss": -0.7, "losses": -0.7, "decline": -0.6,
    "downgrade": -0.8, "underperform": -0.8, "bearish": -0.8, "weak": -0.6,
    "fall": -0.6, "falls": -0.6, "drop": -0.6, "drops": -0.6, "lawsuit": -0.7,
    "probe": -0.6, "fraud": -1.0, "bankruptcy": -1.0, "recall": -0.7,
    "warning": -0.6, "warns": -0.6, "cut": -0.5, "cuts": -0.5, "layoff": -0.7,
    "layoffs": -0.7, "slump": -0.8, "fine": -0.5, "investigation": -0.6,
}

_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_finbert_pipeline = None  # initialisé paresseusement


@dataclass
class SentimentResult:
    """Résultat d'analyse de sentiment d'un texte."""

    score: float        # [-1, 1]
    label: str          # Très positif / Positif / Neutre / Négatif / Très négatif
    confidence: float   # [0, 1]
    impact: float       # [0, 1]


def _label(score: float) -> str:
    if score >= 0.5:
        return "Très positif"
    if score >= 0.15:
        return "Positif"
    if score > -0.15:
        return "Neutre"
    if score > -0.5:
        return "Négatif"
    return "Très négatif"


def analyze_text(text: str, backend: str = "lexicon") -> SentimentResult:
    """Point d'entrée : choisit le backend, avec repli automatique sur lexicon."""
    if backend == "finbert":
        try:
            return _analyze_finbert(text)
        except Exception:  # noqa: BLE001 - repli silencieux si transformers absent
            pass
    return analyze_text_lexicon(text)


def analyze_text_lexicon(text: str) -> SentimentResult:
    """Analyse lexicale : moyenne des mots porteurs de sentiment détectés."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    if not tokens:
        return SentimentResult(0.0, "Neutre", 0.0, 0.0)

    hits = [POSITIVE[t] for t in tokens if t in POSITIVE]
    hits += [NEGATIVE[t] for t in tokens if t in NEGATIVE]
    if not hits:
        return SentimentResult(0.0, "Neutre", 0.2, 0.1)

    score = max(-1.0, min(1.0, sum(hits) / len(hits)))
    confidence = min(1.0, 0.4 + 0.1 * len(hits))
    impact = min(1.0, abs(score) * (0.5 + 0.1 * len(hits)))
    return SentimentResult(
        round(score, 3), _label(score), round(confidence, 2), round(impact, 2)
    )


def _analyze_finbert(text: str) -> SentimentResult:
    """Backend FinBERT (transformers). Chargé une seule fois (paresseux)."""
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline  # import local : dépendance optionnelle

        _finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    result = _finbert_pipeline((text or "")[:512])[0]
    label = result["label"].lower()
    confidence = float(result["score"])
    signed = confidence if label == "positive" else (
        -confidence if label == "negative" else 0.0
    )
    return SentimentResult(
        round(signed, 3), _label(signed), round(confidence, 2), round(abs(signed), 2)
    )
