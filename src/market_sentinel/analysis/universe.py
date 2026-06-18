"""Découverte automatique de l'univers d'analyse — l'app scanne le marché.

Plutôt qu'une liste figée d'entreprises, ce module interroge en direct les
"screeners" publics de Yahoo Finance (les plus actives, plus fortes hausses,
valeurs de croissance sous-évaluées, petites capis dynamiques…) pour
constituer **dynamiquement** la liste des valeurs à surveiller.

Mode :
  - "auto"      : l'app choisit elle-même les valeurs (scan du marché)
  - "watchlist" : on garde la liste fixe de config/watchlist.yaml

Le résultat est mis en cache (par défaut 1 h) pour ne pas re-scanner à
chaque rafraîchissement. En cas d'échec réseau, on retombe proprement sur
la watchlist.
"""

from __future__ import annotations

import time
from typing import List

from ..logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_SCREENS = [
    "most_actives",
    "day_gainers",
    "undervalued_growth_stocks",
    "growth_technology_stocks",
    "aggressive_small_caps",
]

_CACHE: dict[str, tuple[float, List[str]]] = {}
_TTL_SECONDS = 3600


def _scan_market(cfg) -> List[str]:
    """Interroge les screeners Yahoo et renvoie une liste de symboles (actions)."""
    import yfinance as yf

    screens = cfg.get("universe.screens", DEFAULT_SCREENS) or DEFAULT_SCREENS
    count = int(cfg.get("universe.count_per_screen", 25))

    symbols: List[str] = []
    for screen in screens:
        try:
            res = yf.screen(screen, count=count)
            quotes = res.get("quotes", []) if isinstance(res, dict) else []
            for q in quotes:
                sym = q.get("symbol")
                qtype = q.get("quoteType")
                if sym and qtype in (None, "EQUITY"):  # actions uniquement
                    symbols.append(sym)
            log.info("Scan '%s' : %d valeurs", screen, len(quotes))
        except Exception as exc:  # noqa: BLE001
            log.warning("Scan '%s' indisponible : %s", screen, exc)
    return symbols


def get_universe(cfg) -> List[str]:
    """Renvoie la liste des valeurs à analyser (dynamique en mode 'auto')."""
    mode = cfg.get("universe.mode", "auto")
    include_watchlist = cfg.get("universe.include_watchlist", True)
    max_symbols = int(cfg.get("universe.max_symbols", 60))

    base = list(cfg.watchlist) if include_watchlist else []

    if mode != "auto":
        return list(cfg.watchlist)

    now = time.time()
    cached = _CACHE.get("auto")
    if cached and (now - cached[0]) < _TTL_SECONDS:
        scanned = cached[1]
    else:
        scanned = _scan_market(cfg)
        if scanned:
            _CACHE["auto"] = (now, scanned)

    # Le MARCHÉ d'abord (priorité aux découvertes), puis les favoris éventuels.
    seen: set[str] = set()
    out: List[str] = []
    for sym in scanned + base:
        s = str(sym).upper().strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    if not out:  # repli si le scan a totalement échoué
        out = list(cfg.watchlist)
    return out[:max_symbols]


def clear_cache() -> None:
    """Vide le cache pour forcer un nouveau scan du marché."""
    _CACHE.clear()
