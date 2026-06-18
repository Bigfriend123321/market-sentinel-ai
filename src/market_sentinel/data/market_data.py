"""Récupération des cours boursiers et détection de signaux de marché.

Source : yfinance (gratuit, sans clé d'API).

NOTE IMPORTANTE sur le "temps réel" : l'API publique Yahoo Finance fournit
des données souvent **différées** (~15 min) et non un flux tick par tick.
Pour du véritable temps réel professionnel, il faut un fournisseur payant
(Polygon.io, Alpaca, IEX Cloud, flux du courtier...). L'architecture du
module isole cette source : `get_quote` / `get_history` peuvent être
réimplémentées sur un autre fournisseur sans toucher au reste du code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class Quote:
    """Cotation instantanée d'une valeur."""

    ticker: str
    price: Optional[float]
    previous_close: Optional[float]
    change_pct: Optional[float]
    volume: Optional[float]
    currency: Optional[str]
    market_cap: Optional[float]
    timestamp: str


def get_quote(ticker: str) -> Quote:
    """Cotation quasi temps réel (données potentiellement différées)."""
    t = yf.Ticker(ticker)
    price = prev = vol = mcap = None
    currency = None

    # `fast_info` est l'accès rapide recommandé ; il peut échouer selon le ticker.
    try:
        fi = t.fast_info
        price = _as_float(fi.get("last_price"))
        prev = _as_float(fi.get("previous_close"))
        vol = _as_float(fi.get("last_volume"))
        mcap = _as_float(fi.get("market_cap"))
        currency = fi.get("currency")
    except Exception as exc:  # noqa: BLE001 - robustesse réseau volontaire
        log.debug("fast_info indisponible pour %s : %s", ticker, exc)

    # Repli sur l'historique récent si aucun prix n'a été obtenu.
    if price is None:
        try:
            hist = t.history(period="5d", interval="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
                vol = float(hist["Volume"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            log.warning("Historique indisponible pour %s : %s", ticker, exc)

    change_pct = None
    if price is not None and prev:
        change_pct = round((price - prev) / prev * 100, 2)

    return Quote(
        ticker=ticker,
        price=price,
        previous_close=prev,
        change_pct=change_pct,
        volume=vol,
        currency=currency,
        market_cap=mcap,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def get_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Télécharge l'historique OHLCV. Retourne un DataFrame (vide si échec)."""
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval)
    except Exception as exc:  # noqa: BLE001
        log.warning("history() a échoué pour %s : %s", ticker, exc)
        return pd.DataFrame()


def detect_price_move(quote: Quote, threshold_pct: float) -> bool:
    """Vrai si la variation absolue du jour atteint le seuil donné."""
    return quote.change_pct is not None and abs(quote.change_pct) >= threshold_pct


def detect_volume_spike(ticker: str, factor: float = 2.0, lookback: int = 20) -> bool:
    """Vrai si le dernier volume dépasse `factor` x la moyenne récente."""
    df = get_history(ticker, period="3mo", interval="1d")
    if df.empty or len(df) < lookback + 1:
        return False
    recent = df["Volume"].tail(lookback + 1)
    avg = recent.iloc[:-1].mean()
    last = recent.iloc[-1]
    return bool(avg > 0 and last >= factor * avg)


def _as_float(value) -> Optional[float]:
    """Conversion défensive vers float."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
