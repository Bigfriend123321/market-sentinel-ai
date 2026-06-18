"""Persistance locale via SQLite (bibliothèque standard, aucune dépendance).

Tables :
    analyses   : historique des analyses (avec payload JSON complet)
    alerts     : alertes déclenchées
    portfolio  : portefeuille virtuel de l'utilisateur
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, name TEXT, sector TEXT,
    global_score REAL, risk TEXT, potential TEXT, horizon TEXT,
    news_sentiment REAL,
    scenario_up REAL, scenario_stable REAL, scenario_down REAL,
    payload TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT, title TEXT, message TEXT, confidence REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS portfolio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT UNIQUE, shares REAL, avg_price REAL, added_at TEXT
);
"""


class Database:
    """Couche d'accès aux données. Chaque opération ouvre/ferme sa connexion."""

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # --- Analyses --------------------------------------------------------
    def save_analysis(self, analysis: dict) -> None:
        score = analysis["score"]
        scenario = analysis["scenario"]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO analyses
                   (ticker, name, sector, global_score, risk, potential, horizon,
                    news_sentiment, scenario_up, scenario_stable, scenario_down,
                    payload, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    analysis["ticker"], analysis["name"], analysis["sector"],
                    score["global_score"], score["risk"], score["potential"],
                    score["horizon"], analysis["news_sentiment"], scenario["up"],
                    scenario["stable"], scenario["down"],
                    json.dumps(analysis, ensure_ascii=False), analysis["timestamp"],
                ),
            )

    def latest_analyses(self, limit: int = 50) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Alertes ---------------------------------------------------------
    def save_alert(self, ticker: str, title: str, message: str, confidence: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO alerts (ticker, title, message, confidence, created_at) "
                "VALUES (?,?,?,?,?)",
                (ticker, title, message, confidence,
                 datetime.now(timezone.utc).isoformat()),
            )

    def recent_alerts(self, limit: int = 50) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- Portefeuille virtuel -------------------------------------------
    def upsert_position(self, ticker: str, shares: float, avg_price: float) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO portfolio (ticker, shares, avg_price, added_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(ticker) DO UPDATE SET
                       shares = excluded.shares,
                       avg_price = excluded.avg_price""",
                (ticker, shares, avg_price, datetime.now(timezone.utc).isoformat()),
            )

    def get_portfolio(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM portfolio ORDER BY ticker").fetchall()
        return [dict(r) for r in rows]

    def remove_position(self, ticker: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM portfolio WHERE ticker = ?", (ticker,))
