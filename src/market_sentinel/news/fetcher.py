"""Récupération des actualités liées à une valeur et scoring de leur sentiment.

Source par défaut : `yfinance.Ticker(...).news` (gratuit, sans clé). Le schéma
renvoyé par Yahoo a changé au fil des versions : on gère les deux formats.

Le module agrège ensuite le sentiment des titres en un indicateur unique
[-1, 1] pondéré par la confiance et l'impact de chaque article.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List

import yfinance as yf

from ..logging_setup import get_logger
from .sentiment import SentimentResult, analyze_text

log = get_logger(__name__)


@dataclass
class NewsItem:
    """Une actualité analysée."""

    ticker: str
    title: str
    publisher: str
    link: str
    published: str
    sentiment: SentimentResult


def fetch_news(
    ticker: str, max_articles: int = 20, backend: str = "lexicon"
) -> List[NewsItem]:
    """Récupère et analyse les dernières actualités d'une valeur."""
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001
        log.warning("Actualités indisponibles pour %s : %s", ticker, exc)
        raw = []

    items: List[NewsItem] = []
    for article in raw[:max_articles]:
        # Nouveau schéma : données sous "content" ; ancien : à la racine.
        content = article.get("content", article) if isinstance(article, dict) else {}
        title = content.get("title") or article.get("title") or ""
        if not title:
            continue
        publisher = (
            (content.get("provider") or {}).get("displayName")
            or article.get("publisher")
            or ""
        )
        link = (content.get("canonicalUrl") or {}).get("url") or article.get("link", "")
        published = _published(article, content)

        sentiment = analyze_text(title, backend=backend)
        items.append(NewsItem(ticker, title, publisher, link, published, sentiment))
    return items


def aggregate_sentiment(items: List[NewsItem]) -> float:
    """Sentiment moyen pondéré par (confiance x impact). Retour borné [-1, 1]."""
    if not items:
        return 0.0
    numerator = denominator = 0.0
    for item in items:
        weight = max(0.1, item.sentiment.confidence * (0.5 + item.sentiment.impact))
        numerator += item.sentiment.score * weight
        denominator += weight
    return round(numerator / denominator, 3) if denominator else 0.0


def _published(article: dict, content: dict) -> str:
    ts = article.get("providerPublishTime")
    if ts:
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return content.get("pubDate", "") or ""
