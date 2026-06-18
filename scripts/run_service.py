"""Lance le service d'arrière-plan Market Sentinel AI.

    python scripts/run_service.py

Ajoute automatiquement `src/` au chemin d'import pour fonctionner sans
installation préalable du paquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_sentinel.service.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
