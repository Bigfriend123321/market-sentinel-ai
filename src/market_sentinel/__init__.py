"""Market Sentinel AI — moteur modulaire d'analyse des marchés financiers.

Sous-paquets :
    data      : récupération des cours et des fondamentaux (yfinance)
    analysis  : indicateurs techniques, moteur de score, opportunités
    news      : récupération d'actualités et analyse de sentiment
    ai        : moteur prédictif probabiliste (+ modèle ML optionnel)
    alerts    : notifications et règles d'alerte
    storage   : persistance SQLite (analyses, alertes, portefeuille)
    service   : boucle d'exécution en arrière-plan 24h/24

AVERTISSEMENT IMPORTANT
-----------------------
Les analyses fournies sont des aides à la décision et ne constituent pas
des conseils financiers. Toutes les estimations sont probabilistes et ne
représentent jamais des certitudes.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "Les analyses fournies sont des aides à la décision et ne constituent "
    "pas des conseils financiers."
)
