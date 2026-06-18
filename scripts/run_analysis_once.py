"""Exécute UN seul cycle d'analyse et affiche le TOP 10 dans la console.

Idéal pour tester rapidement l'installation sans lancer le service complet :

    python scripts/run_analysis_once.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_sentinel import DISCLAIMER  # noqa: E402
from market_sentinel.analysis.opportunities import build_top_opportunities  # noqa: E402
from market_sentinel.config import load_config  # noqa: E402
from market_sentinel.logging_setup import setup_logging  # noqa: E402
from market_sentinel.storage.database import Database  # noqa: E402


def main() -> None:
    cfg = load_config()
    setup_logging(
        cfg.get("logging.level", "INFO"),
        cfg.resolve_path(cfg.get("logging.file", "logs/market_sentinel.log")),
    )
    db = Database(
        cfg.resolve_path(cfg.get("storage.database_path", "data/market_sentinel.db"))
    )

    print("\n" + "=" * 70)
    print(" MARKET SENTINEL AI — TOP OPPORTUNITÉS DU JOUR")
    print("=" * 70)
    print(f" {DISCLAIMER}\n")

    top = build_top_opportunities(cfg, limit=10)
    for rank, a in enumerate(top, start=1):
        score = a.score
        scenario = a.scenario
        db.save_analysis(a.__dict__)
        print(
            f"{rank:>2}. {a.name:<28} ({a.ticker:<8}) "
            f"Score {score['global_score']:>5}/100 | "
            f"Potentiel {score['potential']:<10} | Risque {score['risk']:<7} | "
            f"Hausse {scenario['up']:>5}%"
        )
        print(f"     Horizon : {score['horizon']} — {a.justification}")

    print("\nAnalyses enregistrées dans la base de données.\n")


if __name__ == "__main__":
    main()
