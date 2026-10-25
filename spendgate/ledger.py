"""Append-only SQLite spend ledger.

One row per completed upstream call. Writes are serialized with a lock so the
threaded HTTP server never corrupts the DB. The ledger is a plain file: it can
be scp'd off the box and queried with the stdlib ``sqlite3`` module or any
tool that speaks SQLite.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# The report CLI groups by these; keeping them in one allowlist means the
# column names that reach the SQL string are never user supplied.
GROUP_COLUMNS = {
    "key": "key_label",
    "model": "model",
    "day": "substr(timestamp, 1, 10)",
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,            -- UTC ISO-8601
    key_label       TEXT    NOT NULL,
    model           TEXT    NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cost            REAL
);

CREATE INDEX IF NOT EXISTS idx_spend_timestamp ON spend (timestamp);
CREATE INDEX IF NOT EXISTS idx_spend_key_label ON spend (key_label);
CREATE INDEX IF NOT EXISTS idx_spend_model ON spend (model);
"""


@dataclass
class SpendRow:
    id: int
    timestamp: str
    key_label: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None


@dataclass
class GroupTotal:
    """Aggregated spend for one group: a key label, a model, or a UTC day."""

    group: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    uncosted_calls: int


@dataclass
class KeyModelTotal:
    """One metrics series' worth of rows: a key label against a model name.

    The token sums stay ``None`` when no row in the group reported usage, which
    is a different fact from "zero tokens" and is what the metrics endpoint
    exposes.
    """

    key_label: str
    model: str
    calls: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None


class Ledger:
    """Thread-safe append-only ledger backed by SQLite."""

    def __init__(self, path: str, create: bool = True):
        self._path = path
        self._readonly = not create
        self._lock = threading.Lock()
        if self._readonly and not Path(path).exists():
            raise FileNotFoundError(f"Ledger file not found: {path}")
        if not self._readonly:
            self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        if self._readonly:
            # A ledger copied off the box is read as-is; never migrate it here.
            return sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=30.0)
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        return conn

    def _ensure_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def record(
        self,
        *,
        key_label: str,
        model: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cost: float | None,
        timestamp: str | None = None,
    ) -> int:
        """Append one spend row. Returns the new row id."""
        if self._readonly:
            raise RuntimeError("ledger opened read-only")
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "INSERT INTO spend "
                    "(timestamp, key_label, model, prompt_tokens, completion_tokens, cost) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (ts, key_label, model, prompt_tokens, completion_tokens, cost),
                )
                conn.commit()
                return int(cur.lastrowid)
            finally:
                conn.close()

    def sum_month(self, key_label: str, year: int, month: int) -> float:
        """Total spend for a key in a calendar month. 0.0 if unknown."""
        prefix = f"{year:04d}-{month:02d}"
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COALESCE(SUM(cost), 0) FROM spend "
                    "WHERE key_label = ? AND timestamp LIKE ?",
                    (key_label, prefix + "%"),
                ).fetchone()
                return float(row[0] or 0.0)
            finally:
                conn.close()

    def rows(self, month: str | None = None) -> list[SpendRow]:
        """Every stored call, oldest first, optionally limited to one month."""
        sql = "SELECT id, timestamp, key_label, model, prompt_tokens, completion_tokens, cost FROM spend"
        params: tuple[Any, ...] = ()
        if month is not None:
            sql += " WHERE timestamp LIKE ?"
            params = (month + "%",)
        sql += " ORDER BY id"
        with self._lock:
            conn = self._connect()
            try:
                fetched = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [SpendRow(*row) for row in fetched]

    def summarize(self, group_by: str, month: str | None = None) -> list[GroupTotal]:
        """Totals per group for the ledger, optionally limited to one month.

        ``group_by`` is one of the keys of ``GROUP_COLUMNS``: a key label, a
        model name, or the UTC calendar day. ``month`` is a ``YYYY-MM`` string.
        """
        try:
            column = GROUP_COLUMNS[group_by]
        except KeyError:
            raise ValueError(f"unknown group: {group_by!r}") from None
        sql = (
            "SELECT {col} AS grp, COUNT(*), "
            "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), "
            "SUM(cost), SUM(CASE WHEN cost IS NULL THEN 1 ELSE 0 END) FROM spend"
        ).format(col=column)
        params = ()
        if month is not None:
            sql += " WHERE timestamp LIKE ?"
            params = (month + "%",)
        sql += " GROUP BY grp ORDER BY " + (
            "grp" if group_by == "day" else "SUM(cost) IS NULL, SUM(cost) DESC, grp"
        )
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        return [
            GroupTotal(
                group=grp if grp is not None else "(unknown)",
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                uncosted_calls=uncosted,
            )
            for grp, calls, prompt_tokens, completion_tokens, cost, uncosted in rows
        ]

    def key_model_totals(self) -> list[KeyModelTotal]:
        """Totals per key label and model, for the metrics endpoint.

        Every row counts, costed or not: a scrape has to agree with the ledger
        even for models that are absent from the price table.
        """
        sql = (
            "SELECT key_label, model, COUNT(*), SUM(prompt_tokens), "
            "SUM(completion_tokens), SUM(cost) FROM spend "
            "GROUP BY key_label, model ORDER BY key_label, model"
        )
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql).fetchall()
            finally:
                conn.close()
        return [
            KeyModelTotal(
                key_label=key_label,
                model=model,
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
            )
            for key_label, model, calls, prompt_tokens, completion_tokens, cost in rows
        ]

    def row_count(self, month: str | None = None) -> int:
        sql = "SELECT COUNT(*) FROM spend"
        params: tuple[Any, ...] = ()
        if month is not None:
            sql += " WHERE timestamp LIKE ?"
            params = (month + "%",)
        with self._lock:
            conn = self._connect()
            try:
                return int(conn.execute(sql, params).fetchone()[0])
            finally:
                conn.close()