"""Screener "Pépites" : sociétés connues, prix d'action faible, fort potentiel.

À partir d'une liste d'analyses déjà calculées (sous forme de dictionnaires,
comme dans le tableau de bord et la base de données), on filtre les valeurs
dont :
    - le prix d'une action est <= max_share_price (action "peu chère"),
    - le score global est >= min_score,
puis on classe par un indice de potentiel combinant le score, le potentiel
haussier du scénario, l'objectif des analystes et l'IA locale.
"""

from __future__ import annotations

from typing import List


def opportunity_index(a: dict) -> float:
    """Indice de potentiel « mine d'or » (plus c'est haut, mieux c'est).

    Le potentiel appris par l'IA (probabilité d'être une future gagnante) est
    désormais un moteur principal du classement, aux côtés du score fondamental,
    du scénario haussier et de l'objectif des analystes.
    """
    score = a["score"]["global_score"]                 # 0..100
    up = a["scenario"]["up"]                            # 0..100
    upside = max(0.0, a.get("upside_pct") or 0.0)       # %
    ai = a.get("ai_up_proba")
    ai_potential = (ai * 100.0) if ai is not None else 50.0  # 0..100
    return 0.35 * score + 0.20 * up + 0.15 * upside + 0.30 * ai_potential


def find_gems(
    analyses: List[dict],
    max_share_price: float = 60.0,
    min_score: float = 60.0,
) -> List[dict]:
    """Renvoie les "pépites" triées par potentiel décroissant."""
    gems = [
        a
        for a in analyses
        if a.get("price") is not None
        and a["price"] <= max_share_price
        and a["score"]["global_score"] >= min_score
    ]
    gems.sort(key=opportunity_index, reverse=True)
    return gems
