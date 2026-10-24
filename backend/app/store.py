"""SQLite persistence and audit store for DocuPilot AI.

Records every generation run, AST analysis, human review, rejection,
approval, and verification evidence with auditable timestamps.
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Optional


import logging
import shutil
import tempfile

logger = logging.getLogger("docupilot.store")


def is_path_writable(path: str) -> bool:
    """Check if a file or its target directory is writable."""
    try:
        if os.path.exists(path):
            return os.access(path, os.W_OK)
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(parent):
            return False
        return os.access(parent, os.W_OK)
    except Exception:
        return False


def db_path() -> str:
    """Determine the database path, ensuring write access across serverless and container runtimes."""
    # 1. If explicit DATABASE_PATH is provided, always honor it
    env_path = os.getenv("DATABASE_PATH")
    if env_path:
        return env_path

    # 2. Check for Serverless environments where root filesystem is read-only (e.g. Vercel, AWS Lambda)
    is_serverless = bool(
        os.getenv("VERCEL")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("FUNCTION_TARGET")
        or os.getenv("K_SERVICE")
    )

    default_path = "./docupilot.db"
    if not is_serverless and is_path_writable(default_path):
        return default_path

    # 3. For Serverless / read-only filesystems, use the system temporary directory (/tmp)
    tmp_path = os.path.join(tempfile.gettempdir(), "docupilot.db")
    if not os.path.exists(tmp_path):
        for candidate in ["./docupilot.db", "../docupilot.db", "backend/docupilot.db"]:
            if os.path.exists(candidate):
                try:
                    shutil.copyfile(candidate, tmp_path)
                    break
                except Exception as copy_err:
                    logger.warning("Could not copy seed database from %s to %s: %s", candidate, tmp_path, copy_err)

    if is_path_writable(tmp_path):
        return tmp_path

    # 4. Ultimate fallback: in-memory database
    return ":memory:"


@contextmanager
def connect():
    path = db_path()
    try:
        con = sqlite3.connect(path, timeout=10)
    except sqlite3.OperationalError:
        # Fallback to temp directory or in-memory
        fallback = os.path.join(tempfile.gettempdir(), "docupilot.db")
        try:
            con = sqlite3.connect(fallback, timeout=10)
        except Exception:
            con = sqlite3.connect(":memory:", timeout=10)

    con.row_factory = sqlite3.Row
    try:
        with con:
            yield con
    finally:
        con.close()


def init_db():
    try:
        with connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    reviewed_at TEXT,
                    language TEXT NOT NULL,
                    status TEXT NOT NULL,
                    original_code TEXT NOT NULL,
                    proposed_code TEXT NOT NULL,
                    final_code TEXT,
                    analysis_json TEXT NOT NULL,
                    verification_json TEXT,
                    mode TEXT NOT NULL,
                    model TEXT,
                    rejection_reason TEXT
                )
            """)

            # Backward compatibility migration for existing SQLite files
            existing_cols = {row["name"] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
            migrations = [
                ("updated_at", "TEXT"),
                ("reviewed_at", "TEXT"),
                ("model", "TEXT"),
                ("rejection_reason", "TEXT"),
                ("workflow_state", "TEXT"),
                ("agents_log_json", "TEXT"),
                ("revision_count", "INTEGER"),
            ]
            for col_name, col_type in migrations:
                if col_name not in existing_cols:
                    try:
                        con.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass
    except Exception as exc:
        logger.warning("Database init_db warning (read-only or in-memory fallback): %s", exc)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_run(
    language: str,
    original: str,
    proposed: str,
    analysis: dict,
    mode: str,
    model: Optional[str] = None,
    workflow_state: str = "AWAITING_APPROVAL",
    agents_log: Optional[Any] = None,
    revision_count: int = 0,
    run_id: Optional[str] = None,
) -> str:
    if not run_id:
        run_id = str(uuid.uuid4())
    created = now_utc()
    agents_json = json.dumps(agents_log, ensure_ascii=False) if agents_log is not None else None

    query = """
        INSERT INTO runs (
            id, created_at, updated_at, language, status,
            original_code, proposed_code, analysis_json, mode, model,
            workflow_state, agents_log_json, revision_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        run_id,
        created,
        created,
        language,
        "pending_review",
        original,
        proposed,
        json.dumps(analysis, ensure_ascii=False),
        mode,
        model,
        workflow_state,
        agents_json,
        revision_count,
    )

    try:
        with connect() as con:
            con.execute(query, params)
    except sqlite3.OperationalError as op_err:
        logger.warning("SQLite save_run OperationalError: %s. Re-routing to writable temp directory...", op_err)
        os.environ["DATABASE_PATH"] = os.path.join(tempfile.gettempdir(), f"docupilot_{uuid.uuid4().hex[:8]}.db")
        try:
            init_db()
            with connect() as con:
                con.execute(query, params)
        except Exception as retry_err:
            logger.error("Failed to persist run to SQLite fallback (%s). Generation completed successfully without persistent record.", retry_err)
    except Exception as exc:
        logger.error("Unexpected error persisting run (%s). Generation completed successfully.", exc)

    return run_id


def approve_run(run_id: str, final_code: str, verification: dict) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        if row["status"] != "pending_review":
            return {"conflict": True, "current_status": row["status"]}

        passed = verification.get("passed", False)
        global_status = verification.get("global_status", "verified" if passed else "failed")

        if not passed:
            new_status = "approved_checks_failed"
        elif global_status == "partially_verified":
            new_status = "approved_partially_verified"
        else:
            new_status = "approved_verified"

        reviewed = now_utc()
        con.execute(
            """
            UPDATE runs
            SET status = ?, workflow_state = 'COMPLETED', final_code = ?, verification_json = ?, reviewed_at = ?, updated_at = ?
            WHERE id = ? AND status = 'pending_review'
            """,
            (new_status, final_code, json.dumps(verification, ensure_ascii=False), reviewed, reviewed, run_id),
        )
    return get_run(run_id)


def reject_run(run_id: str, reason: Optional[str] = None) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        if row["status"] != "pending_review":
            return {"conflict": True, "current_status": row["status"]}

        reviewed = now_utc()
        con.execute(
            """
            UPDATE runs
            SET status = 'rejected', workflow_state = 'REJECTED', rejection_reason = ?, reviewed_at = ?, updated_at = ?
            WHERE id = ? AND status = 'pending_review'
            """,
            (reason or "Rejected by reviewer", reviewed, reviewed, run_id),
        )
    return get_run(run_id)


def get_run(run_id: str) -> Optional[dict]:
    with connect() as con:
        row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    for key in ("analysis_json", "verification_json", "agents_log_json"):
        if key in result:
            raw = result.pop(key)
            target_key = "agents_log" if key == "agents_log_json" else key.replace("_json", "")
            result[target_key] = json.loads(raw) if raw else None
    return result


def list_runs(limit: int = 25, offset: int = 0) -> list[dict]:
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    with connect() as con:
        rows = con.execute(
            """
            SELECT id, created_at, reviewed_at, language, status, mode, model, rejection_reason
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict[str, int]:
    """Calculate real metrics computed directly from persisted runs."""
    with connect() as con:
        total = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        approved = con.execute("SELECT COUNT(*) FROM runs WHERE status LIKE 'approved%'").fetchone()[0]
        rejected = con.execute("SELECT COUNT(*) FROM runs WHERE status = 'rejected'").fetchone()[0]
        pending = con.execute("SELECT COUNT(*) FROM runs WHERE status = 'pending_review'").fetchone()[0]
    return {
        "total_runs": total,
        "approved": approved,
        "rejected": rejected,
        "pending": pending,
    }
