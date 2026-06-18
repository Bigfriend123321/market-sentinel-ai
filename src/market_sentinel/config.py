"""Chargement et accès à la configuration (config/config.yaml + watchlist.yaml).

L'objet `Config` expose :
    - .get("a.b.c", default)   accès par chemin pointé
    - .resolve_path("data/x")  conversion en chemin absolu depuis la racine projet
    - .watchlist               liste des tickers surveillés
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

# Racine du projet : .../market_sentinel_ai
# Ce fichier se trouve dans src/market_sentinel/config.py -> remonter de 2 niveaux.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Config:
    """Conteneur immuable de configuration avec accès par chemin pointé."""

    def __init__(self, data: dict[str, Any], watchlist: list[str]) -> None:
        self._data = data or {}
        self.watchlist = watchlist or []

    def get(self, path: str, default: Any = None) -> Any:
        """Récupère une valeur via un chemin pointé, ex: get('scoring.weights')."""
        node: Any = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def resolve_path(self, relative: str) -> Path:
        """Transforme un chemin relatif de la config en chemin absolu."""
        p = Path(relative)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


@lru_cache(maxsize=1)
def load_config(
    config_file: Optional[str] = None,
    watchlist_file: Optional[str] = None,
) -> Config:
    """Charge la configuration (mise en cache après le premier appel)."""
    config_path = Path(config_file) if config_file else CONFIG_DIR / "config.yaml"
    watchlist_path = (
        Path(watchlist_file) if watchlist_file else CONFIG_DIR / "watchlist.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    tickers: list[str] = []
    if watchlist_path.exists():
        with open(watchlist_path, "r", encoding="utf-8") as fh:
            wl = yaml.safe_load(fh) or {}
        tickers = list(wl.get("tickers", []) or [])

    return Config(data, tickers)
