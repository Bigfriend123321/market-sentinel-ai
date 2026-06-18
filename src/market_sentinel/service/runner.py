"""Service d'arrière-plan : exécute des cycles d'analyse en continu.

Conçu pour tourner silencieusement 24h/24 (faible empreinte CPU : il dort
entre deux cycles). Le sommeil est fractionné pour réagir rapidement à un
signal d'arrêt (Ctrl+C / SIGTERM) et s'arrêter proprement.

Lancement direct :  python -m market_sentinel.service.runner
Ou via le script  :  python scripts/run_service.py
"""

from __future__ import annotations

import signal
import time

from .. import DISCLAIMER
from ..alerts.notifier import evaluate_alert, send_notification
from ..analysis.opportunities import analyze_ticker
from ..analysis.universe import get_universe
from ..config import load_config
from ..logging_setup import get_logger, setup_logging
from ..storage.database import Database

_running = True


def _request_stop(*_args) -> None:
    """Gestionnaire de signal : demande l'arrêt en douceur."""
    global _running
    _running = False


def run_once(cfg, db: Database, log) -> None:
    """Exécute un cycle complet d'analyse sur tout l'univers (dynamique)."""
    log.info("=== Début du cycle d'analyse ===")
    enable_notif = cfg.get("alerts.enable_desktop_notifications", True)
    min_score = cfg.get("alerts.min_score", 75)
    min_confidence = cfg.get("alerts.min_confidence", 0.70)
    move_threshold = cfg.get("data.price_move_alert_pct", 5.0)

    universe = get_universe(cfg)
    log.info("Univers du cycle : %d valeurs", len(universe))
    for ticker in universe:
        try:
            analysis = analyze_ticker(ticker, cfg)
            db.save_analysis(analysis)

            # Alerte "opportunité" (score + confiance élevés).
            triggered = evaluate_alert(analysis, min_score, min_confidence)
            if triggered:
                title, message, confidence = triggered
                db.save_alert(ticker, title, message, confidence)
                if enable_notif:
                    send_notification(title, message)

            # Alerte "mouvement de prix important".
            change = analysis["quote"].get("change_pct")
            if change is not None and abs(change) >= move_threshold:
                title = f"Mouvement important : {ticker}"
                message = f"Variation du jour : {change:+.2f}%"
                db.save_alert(ticker, title, message, 0.5)
                if enable_notif:
                    send_notification(title, message)

        except Exception as exc:  # noqa: BLE001 - un ticker ne doit pas tout arrêter
            log.exception("Erreur lors de l'analyse de %s : %s", ticker, exc)

    log.info("=== Fin du cycle d'analyse ===")


def main() -> None:
    """Point d'entrée du service d'arrière-plan."""
    cfg = load_config()
    log = setup_logging(
        cfg.get("logging.level", "INFO"),
        cfg.resolve_path(cfg.get("logging.file", "logs/market_sentinel.log")),
        cfg.get("logging.max_bytes", 5_242_880),
        cfg.get("logging.backup_count", 5),
    )
    log.info("Démarrage du service Market Sentinel AI")
    log.info(DISCLAIMER)

    db = Database(
        cfg.resolve_path(cfg.get("storage.database_path", "data/market_sentinel.db"))
    )

    signal.signal(signal.SIGINT, _request_stop)
    try:
        signal.signal(signal.SIGTERM, _request_stop)  # absent sur certaines plateformes
    except (AttributeError, ValueError):
        pass

    interval = int(cfg.get("app.refresh_interval_seconds", 900))

    while _running:
        try:
            run_once(cfg, db, log)
        except Exception as exc:  # noqa: BLE001
            log.exception("Erreur durant le cycle : %s", exc)

        # Sommeil fractionné (réactif à l'arrêt).
        slept = 0
        while _running and slept < interval:
            step = min(5, interval - slept)
            time.sleep(step)
            slept += step

    log.info("Service arrêté proprement.")


if __name__ == "__main__":
    main()
