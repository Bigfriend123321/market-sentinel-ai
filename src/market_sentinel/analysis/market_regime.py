"""Régime de marché — l'IA "suit le cours du marché" global.

On résume l'orientation générale du marché (par défaut le S&P 500) en un
score unique dans [-1, 1] :
    +1  marché nettement haussier   /   -1  marché nettement baissier.

Ce score "tilte" les scénarios probabilistes : dans un marché porteur, les
hausses individuelles sont un peu plus probables, et inversement. Le résultat
est mis en cache pendant une heure pour éviter de retélécharger l'indice à
chaque valeur analysée.
"""

from __future__ import annotations

import time
from typing import Optional

from ..data.market_data import get_history
from ..logging_setup import get_logger

log = get_logger(__name__)

_CACHE: dict[str, tuple[float, float]] = {}   # index -> (timestamp, score)
_TTL_SECONDS = 3600


def market_trend_score(index_symbol: str = "^GSPC") -> float:
    """Score d'orientation du marché dans [-1, 1] (0 = neutre)."""
    now = time.time()
    cached = _CACHE.get(index_symbol)
    if cached and (now - cached[0]) < _TTL_SECONDS:
        return cached[1]

    score = 0.0
    try:
        df = get_history(index_symbol, period="1y", interval="1d")
        if df is not None and not df.empty and len(df) >= 60:
            close = df["Close"].astype(float)
            price = float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1])
            sma200 = (
                float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
            )
            mom_60 = price / float(close.iloc[-60]) - 1.0

            parts = []
            parts.append(1.0 if price > sma50 else -1.0)
            parts.append(1.0 if sma50 > sma200 else -1.0)
            parts.append(max(-1.0, min(1.0, mom_60 * 5)))  # ±20% -> ±1
            score = max(-1.0, min(1.0, sum(parts) / len(parts)))
    except Exception as exc:  # noqa: BLE001
        log.debug("Régime de marché indisponible (%s) : %s", index_symbol, exc)
        score = 0.0

    _CACHE[index_symbol] = (now, score)
    return score


def regime_label(score: float) -> str:
    if score >= 0.5:
        return "Marché haussier"
    if score >= 0.15:
        return "Marché plutôt porteur"
    if score > -0.15:
        return "Marché neutre"
    if score > -0.5:
        return "Marché prudent"
    return "Marché baissier"
