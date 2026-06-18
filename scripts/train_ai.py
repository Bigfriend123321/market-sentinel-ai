"""Entraîne l'IA locale — DÉTECTEUR DE « MINES D'OR » (multi-horizons).

    python scripts/train_ai.py

Télécharge l'historique des valeurs (univers scanné du marché par défaut),
construit les features de configuration prix/volume, apprend ce qui distingue
les futures gagnantes (top performeuses à 3/6/12 mois) et sauvegarde le
modèle dans data/predictor.joblib. À relancer de temps en temps pour
réentraîner sur des données fraîches.

⚠️ Les performances passées ne préjugent pas des performances futures.
Biais du survivant assumé (seules les sociétés encore cotées sont vues).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_sentinel.ai.ml_model import train_model  # noqa: E402
from market_sentinel.analysis.universe import get_universe  # noqa: E402
from market_sentinel.config import load_config  # noqa: E402
from market_sentinel.logging_setup import setup_logging  # noqa: E402


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.get("logging.level", "INFO"),
                  cfg.resolve_path(cfg.get("logging.file", "logs/market_sentinel.log")))

    model_path = str(cfg.resolve_path(cfg.get("ai.model_path", "data/predictor.joblib")))
    period = cfg.get("ai.history_period", "max")
    max_iter = int(cfg.get("ai.max_iter", 600))
    test_fraction = float(cfg.get("ai.test_fraction", 0.2))
    horizons = list(cfg.get("ai.explosive_horizons", [63, 126, 252]))
    top_quantile = float(cfg.get("ai.explosive_top_quantile", 0.20))

    # Univers d'entraînement : marché scanné (par défaut) ou watchlist.
    if cfg.get("ai.train_on_universe", True):
        tickers = get_universe(cfg)
        source = "univers scanné du marché"
    else:
        tickers = list(cfg.watchlist)
        source = "watchlist.yaml"

    print("\n=== Entraînement de l'IA — Détecteur de mines d'or ===")
    print(f"Source : {source} | {len(tickers)} valeurs")
    print(f"Horizons : {horizons} jours | Top quantile : {top_quantile:.0%} | Historique : {period}")
    print("Téléchargement de l'historique et entraînement… (2-8 min)\n")

    train_model(
        tickers,
        horizons=horizons,
        model_path=model_path,
        period=period,
        max_iter=max_iter,
        test_fraction=test_fraction,
        top_quantile=top_quantile,
    )

    print(f"\nModèle sauvegardé dans : {model_path}")
    print("Il sera utilisé automatiquement au prochain lancement du tableau de bord.\n")


if __name__ == "__main__":
    main()
