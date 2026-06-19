"""Tests des métriques de backtest (fonction pure, sans réseau)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from market_sentinel.analysis.backtest import _metrics  # noqa: E402


def test_metrics_positive_serie():
    r = pd.Series([0.01] * 24)               # +1% par mois, 24 mois
    equity = (1.0 + r).cumprod()
    m = _metrics(r, equity)
    assert m["months"] == 24
    assert m["hit_rate"] == 1.0
    assert m["cagr"] > 0
    assert m["max_drawdown"] <= 0.0          # drawdown toujours <= 0


def test_metrics_drawdown_detected():
    r = pd.Series([0.10, 0.10, -0.30, 0.05, 0.05])
    equity = (1.0 + r).cumprod()
    m = _metrics(r, equity)
    assert m["max_drawdown"] < 0.0           # une perte a bien eu lieu


def test_metrics_too_short_returns_empty():
    r = pd.Series([0.01])
    equity = (1.0 + r).cumprod()
    assert _metrics(r, equity) == {}


def test_sharpe_is_finite_for_volatile_serie():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.01, 0.04, 36))
    equity = (1.0 + r).cumprod()
    m = _metrics(r, equity)
    assert np.isfinite(m["sharpe"])
