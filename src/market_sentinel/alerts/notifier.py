"""Notifications bureau et règles de déclenchement des alertes."""

from __future__ import annotations

from typing import Optional, Tuple

from ..logging_setup import get_logger

log = get_logger(__name__)


def send_notification(title: str, message: str) -> bool:
    """Affiche une notification bureau (Windows/macOS/Linux) via plyer.

    En cas d'indisponibilité (plyer absent, environnement sans GUI...), on
    se replie sur une trace de log : aucune alerte n'est jamais perdue.
    """
    try:
        from plyer import notification  # import local : dépendance optionnelle

        notification.notify(
            title=title,
            message=message,
            app_name="Market Sentinel AI",
            timeout=10,
        )
        log.info("[ALERTE] %s — %s", title, message.replace("\n", " "))
        return True
    except Exception as exc:  # noqa: BLE001
        log.info("[ALERTE] %s — %s", title, message.replace("\n", " "))
        log.debug("Notification bureau indisponible : %s", exc)
        return False


def evaluate_alert(
    analysis: dict, min_score: float, min_confidence: float
) -> Optional[Tuple[str, str, float]]:
    """Renvoie (titre, message, confiance) si l'analyse déclenche une alerte.

    Critère : score global >= min_score ET confiance du scénario >= min_confidence.
    """
    score = analysis["score"]["global_score"]
    confidence = analysis["scenario"]["confidence"]
    if score >= min_score and confidence >= min_confidence:
        title = f"Opportunité détectée : {analysis['name']}"
        message = (
            f"Score {score:.0f}/100 | Confiance {confidence:.0%}\n"
            f"{analysis['justification']}"
        )
        return title, message, confidence
    return None
