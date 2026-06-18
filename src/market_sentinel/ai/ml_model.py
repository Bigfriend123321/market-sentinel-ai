"""IA locale (scikit-learn) — DÉTECTEUR DE « MINES D'OR ».

Idée : apprendre, à partir du passé, ce qui caractérise — AU MOMENT T — les
valeurs qui sont *ensuite* devenues les plus performantes, puis appliquer ce
motif aux valeurs d'aujourd'hui pour estimer leur « potentiel mine d'or ».

Cible (multi-horizons) : pour chaque date, on classe les valeurs selon leur
rendement futur à 3, 6 et 12 mois (percentile transversal), on combine les
trois, et l'étiquette = 1 si la valeur finit dans le **top quantile** combiné
(une « gagnante » durable, pas un simple pic). L'IA apprend les configurations
prix/volume qui précèdent ces gagnantes.

Modèle : Gradient Boosting (HistGradientBoostingClassifier), local, gère les
valeurs manquantes nativement, classes rééquilibrées (les gagnantes sont rares).

⚠️ HONNÊTETÉ :
 - Biais du survivant — on n'entraîne que sur des sociétés ENCORE cotées
   (les faillites/délistings sont absents), ce qui flatte les résultats.
 - Une étiquette à 12 mois exige 1 an de futur : seules les données d'il y a
   plus d'un an sont étiquetables.
 - La sortie est une PROBABILITÉ, jamais une promesse. On mesure le modèle par
   son « lift » (sa capacité à concentrer les gagnantes en haut du classement).

Utilisation :
    from market_sentinel.ai.ml_model import train_model, get_active_model
    train_model(tickers)                                  # entraîne + sauvegarde
    proba = get_active_model().predict_up_proba(df)       # 0..1 (potentiel) ou None
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..data.market_data import get_history
from ..logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_PATH = "data/predictor.joblib"
DEFAULT_HORIZONS = [63, 126, 252]   # ~3, 6, 12 mois de bourse

# Cache du modèle actif (chargé une seule fois par processus).
_active_model: Optional["MLPredictor"] = None
_active_path: Optional[str] = None


def _features_from_history(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Indicateurs de configuration prix/volume (précurseurs de gros mouvements).

    Les valeurs manquantes (historique court) sont conservées : le modèle
    Gradient Boosting les gère nativement.
    """
    if df is None or df.empty or len(df) < 60:
        return None
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df else pd.Series(index=close.index, dtype=float)

    feat = pd.DataFrame(index=close.index)

    # Momentum sur plusieurs horizons (le momentum précède souvent les gros runs)
    for n in (5, 10, 20, 60, 120):
        feat[f"ret_{n}"] = close.pct_change(n)

    # Distance aux moyennes mobiles (structure de tendance)
    feat["dist_sma20"] = close / close.rolling(20).mean() - 1.0
    feat["dist_sma50"] = close / close.rolling(50).mean() - 1.0
    feat["dist_sma200"] = close / close.rolling(200).mean() - 1.0

    # Position dans le canal 52 semaines (les gagnantes cassent souvent leurs plus hauts)
    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    feat["dist_52w_high"] = close / roll_max - 1.0
    feat["dist_52w_low"] = close / roll_min - 1.0

    # Volatilité réalisée + expansion de volatilité
    daily = close.pct_change()
    feat["vol_20"] = daily.rolling(20).std()
    feat["vol_60"] = daily.rolling(60).std()
    feat["vol_ratio"] = feat["vol_20"] / feat["vol_60"].replace(0, np.nan)

    # RSI (normalisé 0..1)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi"] = (100 - 100 / (1 + rs)) / 100.0

    # MACD histogramme normalisé par le prix
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat["macd_hist"] = (macd - signal) / close

    # %b des bandes de Bollinger
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    feat["bb_pct"] = (close - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)

    # Surge de volume (les gros mouvements s'accompagnent souvent d'un afflux de volume)
    feat["vol_trend"] = volume / volume.rolling(20).mean() - 1.0

    return feat


FEATURE_COLUMNS = [
    "ret_5", "ret_10", "ret_20", "ret_60", "ret_120",
    "dist_sma20", "dist_sma50", "dist_sma200",
    "dist_52w_high", "dist_52w_low",
    "vol_20", "vol_60", "vol_ratio", "rsi", "macd_hist", "bb_pct", "vol_trend",
]


def _ticker_panel(ticker: str, horizons: List[int], period: str) -> Optional[pd.DataFrame]:
    """Frame d'une valeur : features + rendements futurs (multi-horizons), par date."""
    try:
        df = get_history(ticker, period=period, interval="1d")
    except Exception:  # noqa: BLE001
        return None
    feat = _features_from_history(df)
    if feat is None:
        return None
    close = df["Close"].astype(float)

    frame = feat[FEATURE_COLUMNS].copy()
    fwd_cols = []
    for h in horizons:
        col = f"fwd_{h}"
        frame[col] = close.shift(-h) / close - 1.0
        fwd_cols.append(col)

    # On exige les rendements futurs (étiquettes) ; on garde les features NaN
    # (le Gradient Boosting les gère). Un minimum de signal récent est requis.
    frame = frame.dropna(subset=["ret_20"] + fwd_cols)
    if len(frame) < 60:
        return None

    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame["ticker"] = ticker
    return frame


def _build_panel(
    tickers: List[str],
    horizons: List[int],
    period: str = "max",
    top_quantile: float = 0.20,
    min_cross_section: int = 5,
) -> pd.DataFrame:
    """Panel multi-valeurs + étiquette « gagnante durable » (multi-horizons).

    Pour chaque date et chaque horizon, on calcule le percentile transversal du
    rendement futur. On moyenne les percentiles des 3 horizons : une valeur est
    étiquetée « mine d'or » (1) si ce score combiné la place dans le top
    `top_quantile` du marché ce jour-là.
    """
    frames = [
        f for f in (_ticker_panel(t, horizons, period) for t in tickers)
        if f is not None
    ]
    if not frames:
        raise ValueError("Aucune donnée exploitable pour l'entraînement.")

    panel = pd.concat(frames).sort_index()

    # Percentile transversal (par date) de chaque rendement futur, puis moyenne.
    pct_cols = []
    for h in horizons:
        col = f"fwd_{h}"
        pcol = f"pct_{h}"
        panel[pcol] = panel.groupby(level=0)[col].rank(pct=True)
        pct_cols.append(pcol)
    panel["_composite"] = panel[pct_cols].mean(axis=1)
    panel["_cnt"] = panel.groupby(level=0)[f"fwd_{horizons[0]}"].transform("count")

    panel = panel[panel["_cnt"] >= min_cross_section].copy()
    panel["target"] = (panel["_composite"] >= (1.0 - top_quantile)).astype(int)
    return panel


def train_model(
    tickers: List[str],
    horizons: Optional[List[int]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    period: str = "max",
    max_iter: int = 600,
    test_fraction: float = 0.2,
    top_quantile: float = 0.20,
) -> "MLPredictor":
    """Entraîne le détecteur de mines d'or et sauvegarde le modèle.

    Métriques honnêtes sur une validation par **panel temporel** :
      - AUC et average precision (AP),
      - et surtout le **lift@10%** : à quel point les vraies gagnantes sont
        concentrées dans le top 10% des valeurs les mieux notées par l'IA
        (lift = 2 => deux fois plus de gagnantes que le hasard).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    horizons = horizons or DEFAULT_HORIZONS
    panel = _build_panel(tickers, horizons, period=period, top_quantile=top_quantile)

    # Découpage temporel : dates anciennes -> entraînement, récentes -> test.
    dates = np.array(sorted(panel.index.unique()))
    cut_date = dates[int(len(dates) * (1.0 - test_fraction))]
    train = panel[panel.index <= cut_date]
    test = panel[panel.index > cut_date]
    X_tr, y_tr = train[FEATURE_COLUMNS], train["target"]
    X_te, y_te = test[FEATURE_COLUMNS], test["target"]

    def _make():
        return HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.05,
            max_depth=4,
            l2_regularization=1.0,
            class_weight="balanced",      # les gagnantes sont rares
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
        )

    # 1) Évaluation honnête sur le futur jamais vu.
    eval_model = _make()
    eval_model.fit(X_tr.values, y_tr.values)
    try:
        proba_te = eval_model.predict_proba(X_te.values)[:, 1]
        y = y_te.values
        auc = roc_auc_score(y, proba_te)
        ap = average_precision_score(y, proba_te)
        base = float(y.mean())
        order = np.argsort(-proba_te)
        k = max(1, int(0.10 * len(order)))
        prec_top = float(y[order[:k]].mean())
        lift = prec_top / base if base > 0 else float("nan")
        log.info(
            "IA (mines d'or) — validation : AUC %.3f | AP %.3f | base %.1f%% | "
            "précision top10%% %.1f%% | LIFT %.2fx (%d entraîn. / %d valid., %d valeurs)",
            auc, ap, base * 100, prec_top * 100, lift, len(X_tr), len(X_te), len(tickers),
        )
    except Exception as exc:  # noqa: BLE001
        log.info("IA entraînée (métriques indisponibles : %s)", exc)

    # 2) Modèle final : ré-entraîné sur l'intégralité des données.
    final_model = _make()
    final_model.fit(panel[FEATURE_COLUMNS].values, panel["target"].values)
    log.info("IA — modèle final entraîné sur %d échantillons.", len(panel))

    predictor = MLPredictor(final_model, FEATURE_COLUMNS, horizons)
    predictor.save(model_path)
    return predictor


class MLPredictor:
    """Enveloppe autour du modèle scikit-learn entraîné."""

    def __init__(self, model, columns: List[str], horizons: List[int]) -> None:
        self.model = model
        self.columns = columns
        self.horizons = horizons

    def predict_up_proba(self, df: pd.DataFrame) -> Optional[float]:
        """Probabilité (0..1) que la valeur soit une future « mine d'or ».

        Estimée depuis la configuration la plus récente. Plus c'est élevé, plus
        la valeur ressemble aux gagnantes durables du passé (jamais une garantie).
        """
        feat = _features_from_history(df)
        if feat is None:
            return None
        last = feat[self.columns].iloc[-1]
        if last.isna().all():            # aucune feature exploitable
            return None
        proba = self.model.predict_proba([last.values])[0]
        return float(proba[1])

    def save(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        import joblib

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": self.model, "columns": self.columns, "horizons": self.horizons},
            path,
        )

    @classmethod
    def load(cls, model_path: str = DEFAULT_MODEL_PATH) -> Optional["MLPredictor"]:
        import joblib

        path = Path(model_path)
        if not path.exists():
            return None
        blob = joblib.load(path)
        horizons = blob.get("horizons") or [blob.get("horizon_days", 126)]
        return cls(blob["model"], blob["columns"], horizons)


def get_active_model(model_path: str = DEFAULT_MODEL_PATH) -> Optional["MLPredictor"]:
    """Charge (une seule fois) et renvoie le modèle entraîné, ou None s'il n'existe pas."""
    global _active_model, _active_path
    if _active_model is not None and _active_path == model_path:
        return _active_model
    try:
        _active_model = MLPredictor.load(model_path)
        _active_path = model_path
        if _active_model is not None:
            log.info("IA locale chargée depuis %s", model_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Impossible de charger l'IA (%s) : %s", model_path, exc)
        _active_model = None
    return _active_model
