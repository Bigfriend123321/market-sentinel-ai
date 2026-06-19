"""Tests de l'analyse technique (compute_technical) sur données synthétiques."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_sentinel.analysis.technical import compute_technical  # noqa: E402


def _ohlcv(n: int, drift: float) -> pd.DataFrame:
    """Série de prix synthétique avec une dérive donnée (drift>0 = hausse)."""
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(n, drift))), index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": 1_000_000.0},
        index=idx,
    )


def test_short_history_returns_default():
    sig = compute_technical(_ohlcv(20, 0.001))
    assert sig.trend == "neutre" and sig.score == 50.0


def test_uptrend_is_bullish():
    sig = compute_technical(_ohlcv(300, 0.0015))
    assert sig.trend == "haussier"
    assert sig.score > 50.0


def test_downtrend_is_bearish():
    sig = compute_technical(_ohlcv(300, -0.0015))
    assert sig.trend == "baissier"
    assert sig.score < 50.0


def test_score_bounds_and_fields():
    sig = compute_technical(_ohlcv(300, 0.0005))
    assert 0.0 <= sig.score <= 100.0
    assert sig.rsi is None or 0.0 <= sig.rsi <= 100.0
    assert sig.volatility is None or sig.volatility >= 0.0


def test_empty_dataframe_is_safe():
    sig = compute_technical(pd.DataFrame())
    assert sig.trend == "neutre"
