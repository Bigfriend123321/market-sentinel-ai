"""Tests du moteur prédictif probabiliste (predict_scenarios). Sans réseau."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_sentinel.ai.predictor import predict_scenarios  # noqa: E402


def test_scenarios_sum_to_100():
    s = predict_scenarios(75, 65, 0.3, 0.25)
    assert abs((s.up + s.stable + s.down) - 100.0) < 0.5


def test_confidence_in_bounds():
    for args in [(90, 80, 0.5, 0.1), (35, 30, -0.6, 0.9), (60, 50, 0.0, 0.3)]:
        s = predict_scenarios(*args)
        assert 0.30 <= s.confidence <= 0.95


def test_high_score_favours_upside():
    s = predict_scenarios(90, 80, 0.5, 0.2)
    assert s.up > s.down


def test_low_score_favours_downside():
    s = predict_scenarios(35, 30, -0.5, 0.2)
    assert s.down > s.up


def test_volatility_flattens_distribution():
    # Même signal haussier, mais volatilité plus forte -> hausse moins tranchée.
    low_vol = predict_scenarios(90, 80, 0.5, 0.10)
    high_vol = predict_scenarios(90, 80, 0.5, 0.90)
    assert low_vol.up > high_vol.up


def test_ai_signal_shifts_probabilities():
    bullish_ai = predict_scenarios(70, 60, 0.2, 0.3, ai_up_proba=0.9)
    bearish_ai = predict_scenarios(70, 60, 0.2, 0.3, ai_up_proba=0.1)
    assert bullish_ai.up > bearish_ai.up
