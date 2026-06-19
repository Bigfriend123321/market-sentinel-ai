"""Backtesting HONNÊTE d'une stratégie de sélection (point-in-time / walk-forward).

⚠️ On NE backteste PAS le score fondamental : yfinance ne fournit pas de
fondamentaux historiques datés, donc l'utiliser rétrospectivement créerait une
FUITE de données (look-ahead). On backteste donc ce qui est honnêtement
reconstituable à chaque date passée : une stratégie **momentum + tendance**
basée uniquement sur les prix.

Principe : chaque mois, on classe les valeurs avec les SEULES données connues à
cette date, on « achète » le TOP N à parts égales, et on mesure le rendement
réalisé le mois suivant. On compare au benchmark (S&P 500 par défaut).

Biais du survivant assumé : l'univers testé = des valeurs encore cotées
aujourd'hui (les délistings/faillites passés sont absents → résultats flattés).
Les performances passées ne préjugent pas des performances futures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from ..data.market_data import get_history
from ..logging_setup import get_logger

log = get_logger(__name__)


@dataclass
class BacktestResult:
    """Résultat d'un backtest : courbes d'équité + métriques."""

    equity: pd.Series           # courbe d'équité de la stratégie (base 1.0)
    benchmark: pd.Series        # courbe d'équité du benchmark (base 1.0)
    monthly_returns: pd.Series  # rendements mensuels de la stratégie
    metrics: dict
    benchmark_metrics: dict
    n_tickers: int


def _normalized_close(df: pd.DataFrame) -> Optional[pd.Series]:
    """Série de clôtures indexée par date naïve normalisée."""
    if df is None or df.empty or "Close" not in df:
        return None
    s = df["Close"].astype(float).copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s


def _prices(tickers: Iterable[str], period: str) -> pd.DataFrame:
    """Construit un tableau prix (colonnes = tickers) aligné par date."""
    cols = {}
    for ticker in tickers:
        s = _normalized_close(get_history(ticker, period=period, interval="1d"))
        if s is not None and not s.empty:
            cols[ticker] = s
    return pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()


def _metrics(returns: pd.Series, equity: pd.Series, ppy: int = 12, rf: float = 0.0) -> dict:
    """Métriques standard à partir des rendements périodiques et de l'équité."""
    r = returns.dropna()
    if len(r) < 2 or equity.empty:
        return {}
    years = len(r) / ppy
    final = float(equity.iloc[-1])
    cagr = final ** (1 / years) - 1.0 if years > 0 and final > 0 else float("nan")
    vol = float(r.std() * np.sqrt(ppy))
    sharpe = ((r.mean() * ppy) - rf) / vol if vol > 0 else float("nan")
    max_dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "total_return": final - 1.0,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": float((r > 0).mean()),
        "months": int(len(r)),
    }


def run_backtest(
    tickers: Iterable[str],
    period: str = "5y",
    top_n: int = 10,
    benchmark: str = "^GSPC",
    momentum_months: int = 6,
    trend_months: int = 10,
    rf: float = 0.0,
) -> BacktestResult:
    """Backtest walk-forward d'une stratégie momentum+tendance, rééquilibrée mensuellement.

    À chaque fin de mois t : on retient les valeurs au-dessus de leur moyenne
    mobile de tendance, on garde les `top_n` plus forts momentums (connus à t),
    et on encaisse leur rendement moyen équipondéré sur le mois t -> t+1.
    """
    prices = _prices(list(tickers), period)
    if prices.empty:
        raise ValueError("Aucune donnée de prix pour le backtest.")

    bench_close = _normalized_close(get_history(benchmark, period=period, interval="1d"))

    # Passage en mensuel (fin de mois) — décisions et rendements alignés.
    monthly = prices.resample("ME").last()
    m_ret = monthly.pct_change()
    momentum = monthly / monthly.shift(momentum_months) - 1.0      # connu à t
    trend_ma = monthly.rolling(trend_months).mean()
    above_trend = monthly > trend_ma                               # connu à t

    bench_ret = None
    if bench_close is not None:
        bench_ret = bench_close.resample("ME").last().pct_change()

    dates = monthly.index
    warmup = max(momentum_months, trend_months)
    rows = []
    for i in range(warmup, len(dates) - 1):
        t, nxt = dates[i], dates[i + 1]
        scores = momentum.loc[t].where(above_trend.loc[t])
        picks = scores.dropna().sort_values(ascending=False).head(top_n).index

        if len(picks) == 0:
            strat_r = 0.0                                          # rien d'éligible -> cash
        else:
            strat_r = float(m_ret.loc[nxt, picks].mean(skipna=True))
            if np.isnan(strat_r):
                strat_r = 0.0

        bench_r = 0.0
        if bench_ret is not None and nxt in bench_ret.index and not pd.isna(bench_ret.loc[nxt]):
            bench_r = float(bench_ret.loc[nxt])
        rows.append((nxt, strat_r, bench_r))

    if not rows:
        raise ValueError("Historique insuffisant pour le backtest.")

    bdf = pd.DataFrame(rows, columns=["date", "strat", "bench"]).set_index("date")
    equity = (1.0 + bdf["strat"]).cumprod()
    bench_equity = (1.0 + bdf["bench"]).cumprod()

    metrics = _metrics(bdf["strat"], equity, rf=rf)
    bench_metrics = _metrics(bdf["bench"], bench_equity, rf=rf)
    metrics["beat_benchmark"] = float((bdf["strat"] > bdf["bench"]).mean())

    log.info(
        "Backtest : %d valeurs | CAGR %.1f%% vs bench %.1f%% | Sharpe %.2f | DD %.1f%%",
        prices.shape[1], metrics.get("cagr", float("nan")) * 100,
        bench_metrics.get("cagr", float("nan")) * 100,
        metrics.get("sharpe", float("nan")), metrics.get("max_drawdown", float("nan")) * 100,
    )
    return BacktestResult(equity, bench_equity, bdf["strat"], metrics, bench_metrics, prices.shape[1])
