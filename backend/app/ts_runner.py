"""TypeScript syntax validator and symbol extractor runner.

Executes isolated TypeScript AST parsing via Node.js and TypeScript compiler API.
Safely runs with timeouts and stdin piping without shell interpolation.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("docupilot.ts_runner")

SCRIPT_PATH = Path(__file__).resolve().parent / "ts_validator.cjs"


def validate_typescript(source: str, timeout_sec: float = 5.0) -> dict[str, Any]:
    """Parse TypeScript source code and return validation diagnostics and symbols.

    Returns:
        dict with keys:
          - available (bool): whether node and typescript compiler are usable
          - valid (bool | None): True if syntax is valid, False if invalid, None if not verifiable
          - status (str): 'verified', 'invalid', 'parser_unavailable', or 'error'
          - errors (list[dict]): syntax error diagnostics with line, col, message
          - symbols (list[dict]): extracted AST symbols
          - message (str): diagnostic summary
    """
    node_bin = shutil.which("node")
    if not node_bin or not SCRIPT_PATH.is_file():
        return {
            "available": False,
            "valid": None,
            "status": "parser_unavailable",
            "errors": [],
            "symbols": [],
            "message": "Node.js or TypeScript validator script not found.",
        }

    try:
        proc = subprocess.run(
            [node_bin, str(SCRIPT_PATH)],
            input=source,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            return {
                "available": False,
                "valid": None,
                "status": "error",
                "errors": [{"message": proc.stderr.strip() or "Node process failed"}],
                "symbols": [],
                "message": f"TypeScript validation process failed: {proc.stderr[:100]}",
            }
        data = json.loads(proc.stdout.strip())
        if not data.get("available", True):
            return {
                "available": False,
                "valid": None,
                "status": "parser_unavailable",
                "errors": [],
                "symbols": [],
                "message": data.get("error", "TypeScript module unavailable"),
            }
        is_valid = data.get("valid")
        errors = data.get("errors", [])
        status = "verified" if is_valid else "invalid"
        summary = (
            "TypeScript compiler AST parse passed."
            if is_valid
            else f"TypeScript syntax error(s): {', '.join(e.get('message', '') for e in errors[:2])}"
        )
        return {
            "available": True,
            "valid": is_valid,
            "status": status,
            "errors": errors,
            "symbols": data.get("symbols", []),
            "message": summary,
        }
    except subprocess.TimeoutExpired:
        return {
            "available": True,
            "valid": None,
            "status": "parser_unavailable",
            "errors": [{"message": "Parser execution timed out"}],
            "symbols": [],
            "message": "TypeScript parsing timed out.",
        }
    except Exception as exc:
        logger.warning(f"Failed to execute TypeScript validator: {exc}")
        return {
            "available": False,
            "valid": None,
            "status": "error",
            "errors": [{"message": str(exc)}],
            "symbols": [],
            "message": f"TypeScript validation exception: {exc}",
        }
