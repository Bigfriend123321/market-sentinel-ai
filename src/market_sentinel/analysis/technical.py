"""Analyse technique approfondie : RSI, moyennes mobiles, MACD (+ histogramme),
bandes de Bollinger, momentum, position dans le canal 52 semaines, volatilité.

Toutes les fonctions sont pures (pandas/numpy) et testables sans réseau.
Le module produit un sous-score technique /100 réutilisé par le moteur de
score global et par le moteur prédictif.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TechnicalSignals:
    """Synthèse des indicateurs techniques d'une valeur."""

    rsi: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None        # histogramme MACD (macd - signal)
    bollinger_pct: Optional[float] = None    # %b : position dans les bandes (0=bas,1=haut)
    momentum_20: Optional[float] = None      # variation sur 20 séances (fraction)
    week52_position: Optional[float] = None  # position dans le canal 52 sem (0..1)
    trend: str = "neutre"                    # "haussier" | "baissier" | "neutre"
    volatility: Optional[float] = None       # volatilité annualisée (fraction)
    score: float = 50.0                      # sous-score technique /100


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder simplifié)."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_technical(df: pd.DataFrame) -> TechnicalSignals:
    """Calcule l'ensemble des signaux techniques à partir d'un DataFrame OHLCV."""
    if df is None or df.empty or "Close" not in df:
        return TechnicalSignals()

    close = df["Close"].dropna()
    if len(close) < 30:  # historique insuffisant pour des indicateurs fiables
        return TechnicalSignals()

    price = float(close.iloc[-1])

    rsi_series = _rsi(close)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.dropna().empty else None

    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_v = float(macd_line.iloc[-1])
    signal_v = float(signal_line.iloc[-1])
    macd_hist = macd_v - signal_v

    # Bandes de Bollinger (20, 2σ) -> %b
    bollinger_pct = None
    if len(close) >= 20:
        ma20 = close.rolling(20).mean()
        sd20 = close.rolling(20).std()
        upper = ma20.iloc[-1] + 2 * sd20.iloc[-1]
        lower = ma20.iloc[-1] - 2 * sd20.iloc[-1]
        if upper != lower:
            bollinger_pct = float((price - lower) / (upper - lower))

    # Momentum sur 20 séances
    momentum_20 = None
    if len(close) >= 21:
        past = float(close.iloc[-21])
        if past:
            momentum_20 = price / past - 1.0

    # Position dans le canal 52 semaines (≈ 252 séances)
    week52_position = None
    window = close.tail(252)
    lo, hi = float(window.min()), float(window.max())
    if hi != lo:
        week52_position = (price - lo) / (hi - lo)

    returns = close.pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252)) if not returns.empty else None

    trend = _trend(price, sma50, sma200)
    score = _technical_score(trend, macd_hist, rsi, bollinger_pct, momentum_20)

    return TechnicalSignals(
        rsi=rsi,
        sma_50=sma50,
        sma_200=sma200,
        macd=macd_v,
        macd_signal=signal_v,
        macd_hist=macd_hist,
        bollinger_pct=bollinger_pct,
        momentum_20=momentum_20,
        week52_position=week52_position,
        trend=trend,
        volatility=volatility,
        score=score,
    )


def _trend(price: float, sma50: Optional[float], sma200: Optional[float]) -> str:
    if sma50 is not None and sma200 is not None:
        if price > sma50 > sma200:
            return "haussier"
        if price < sma50 < sma200:
            return "baissier"
        return "neutre"
    if sma50 is not None:
        return "haussier" if price > sma50 else "baissier"
    return "neutre"


def _technical_score(
    trend: str,
    macd_hist: float,
    rsi: Optional[float],
    bollinger_pct: Optional[float],
    momentum_20: Optional[float],
) -> float:
    """Combine les signaux en un sous-score borné [0, 100]."""
    score = 50.0
    if trend == "haussier":
        score += 18
    elif trend == "baissier":
        score -= 18

    score += 8 if macd_hist > 0 else -8  # croisement MACD

    if rsi is not None:
        if rsi < 30:        # survendu -> potentiel de rebond
            score += 8
        elif rsi > 70:      # suracheté -> risque de correction
            score -= 8

    if bollinger_pct is not None:
        if bollinger_pct < 0.1:
            score += 6      # proche de la bande basse
        elif bollinger_pct > 0.9:
            score -= 6      # proche de la bande haute

    if momentum_20 is not None:
        score += max(-8, min(8, momentum_20 * 40))  # ±20% -> ±8 pts

    return max(0.0, min(100.0, score))
