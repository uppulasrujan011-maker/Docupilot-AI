"""Test suite for DocuPilot AI.

Verifies:
- Genuine AST parsing for Python, JavaScript, and TypeScript
- Honest handling of unsupported/non-verified languages (valid_syntax=None)
- Rejection of malformed code (including balanced brackets with invalid syntax)
- Human-in-the-loop approval and rejection workflows
- Prevention of duplicate or invalid state transitions (409 Conflict)
- Honest error reporting for OpenAI API failures without silent mock fallback
- Explicit identification of demo/mock mode
- SQLite audit persistence and history tracking
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
try:
    from openai import AuthenticationError, RateLimitError
except ImportError:
    from app.agents.documentation import AuthenticationError, RateLimitError

from app.analyzer import SUPPORTED_LANGUAGES, analyze
from app.languages import REGISTRY, get_language, list_capabilities
from app.main import app, mock_documentation
from app.verifier import verify

client = TestClient(app)

LANG_SAMPLES = {
    "python": "def calculate_discount(price, discount):\n    if price <= 0:\n        return 0\n    return price - (price * discount / 100)\n",
    "javascript": "function calculateShipping(weight, distance) {\n    return weight * 2.5 + distance * 0.4;\n}",
    "typescript": "export function computeTotal(items: number[], rate: number): number {\n    return items.reduce((a, b) => a + b, 0) * rate;\n}",
    "java": "public class OrderProcessor {\n    public double calculateDiscount(double price, double discount) {\n        return price * discount;\n    }\n}",
    "c": "float calculate_discount(float price, float discount) {\n    return price - discount;\n}",
    "cpp": "class DiscountCalculator {\npublic:\n    double calculateDiscount(double price, double discount) {\n        return price * discount;\n    }\n};",
    "csharp": "public class ShoppingCart {\n    public decimal CalculateShipping(decimal weight, decimal distance) {\n        return weight * 2.5m;\n    }\n}",
    "sql": "CREATE FUNCTION calculate_discount(price DECIMAL, discount DECIMAL) RETURNS DECIMAL BEGIN RETURN price; END;",
}


# ==============================================================================
# Phase 3: Parser & Language Registry Tests
# ==============================================================================

def test_language_registry_completeness():
    caps = list_capabilities()
    assert len(caps) == len(SUPPORTED_LANGUAGES)
    py = get_language("python")
    assert py.syntax_validation_capable is True
    assert py.has_real_parser is True

    js = get_language("javascript")
    assert js.syntax_validation_capable is True
    assert js.parser_type == "esprima"

    ts = get_language("typescript")
    assert ts.syntax_validation_capable is True
    assert ts.parser_type == "typescript_compiler"

    java = get_language("java")
    assert java.syntax_validation_capable is False
    assert java.has_real_parser is False


def test_python_valid_syntax_passes():
    res = analyze("def add(a, b):\n    return a + b\n", "python")
    assert res["valid_syntax"] is True
    assert res["syntax_status"] == "verified"
    assert res["parser_used"] == "python_ast"
    assert any(s["name"] == "add" for s in res["symbols"])


def test_python_malformed_syntax_fails():
    res = analyze("def add(a, b\n    return a + b", "python")
    assert res["valid_syntax"] is False
    assert res["syntax_status"] == "invalid"
    assert "SyntaxError" in res["syntax_message"]

    v = verify("def add(a, b\n    return a + b", "python")
    assert v["passed"] is False
    assert v["global_status"] == "failed"


def test_javascript_valid_syntax_passes():
    res = analyze("function add(a, b) { return a + b; }", "javascript")
    assert res["valid_syntax"] is True
    assert res["syntax_status"] == "verified"
    assert res["parser_used"] == "esprima"
    assert any(s["name"] == "add" for s in res["symbols"])


def test_javascript_es6_module_analysis():
    code = "export function multiply(x, y) { return x * y; }"
    res = analyze(code, "javascript")
    assert res["valid_syntax"] is True
    assert res["syntax_status"] == "verified"
    assert any(s["name"] == "multiply" for s in res["symbols"])


def test_malformed_javascript_with_balanced_brackets_fails_verification():
    # Balanced brackets () and {}, but invalid JavaScript grammar:
    bad_js = "function add() { const x = ; return x; }"
    res = analyze(bad_js, "javascript")
    assert res["valid_syntax"] is False
    assert res["syntax_status"] == "invalid"

    v = verify(bad_js, "javascript")
    assert v["passed"] is False
    assert v["global_status"] == "failed"


def test_typescript_valid_syntax_passes():
    res = analyze(LANG_SAMPLES["typescript"], "typescript")
    assert res["valid_syntax"] is True
    assert res["syntax_status"] == "verified"
    assert res["parser_used"] == "typescript_compiler"
    assert len(res["symbols"]) > 0


def test_typescript_malformed_fails_verification():
    bad_ts = "export function computeTotal(items: number[ rate: number): number { return 0; }"
    res = analyze(bad_ts, "typescript")
    assert res["valid_syntax"] is False
    assert res["syntax_status"] == "invalid"

    v = verify(bad_ts, "typescript")
    assert v["passed"] is False
    assert v["global_status"] == "failed"


@pytest.mark.parametrize("lang", ["java", "c", "cpp", "csharp", "sql"])
def test_unsupported_languages_not_verified(lang):
    code = LANG_SAMPLES[lang]
    res = analyze(code, lang)
    # Must NOT claim syntax verification
    assert res["valid_syntax"] is None
    assert res["syntax_status"] == "not_verified"
    assert res["parser_used"] == "regex_heuristic"
    assert len(res["symbols"]) > 0


# ==============================================================================
# Phase 4: Verification Pipeline Tests
# ==============================================================================

def test_python_procedural_script_documentation():
    code = "a = int(input())\nb = int(input())\nprint(a + b)\n"
    analysis = analyze(code, "python")
    assert analysis["valid_syntax"] is True

    doc_code = mock_documentation(code, "python", analysis)
    assert '"""' in doc_code
    assert "a = int(input())" in doc_code

    v = verify(doc_code, "python", original_code=code)
    assert v["passed"] is True
    assert v["global_status"] == "verified"


def test_structural_consistency_failure():
    orig = "def keep_this(): pass\ndef also_keep(): pass"
    # Documented code accidentally removed also_keep()
    mutated = 'def keep_this():\n    """Docstring."""\n    pass'
    v = verify(mutated, "python", original_code=orig)
    assert v["passed"] is False
    assert v["global_status"] == "failed"
    assert any("also_keep" in c["details"] for c in v["checks"] if c["name"] == "Structural Consistency")


def test_bracket_mismatch_verification_fails():
    result = verify("public class Broken { void test() { return; }", "java")
    assert not result["passed"]
    assert result["global_status"] == "failed"


# ==============================================================================
# Phase 2: OpenAI Configuration & Safe Error Handling Tests
# ==============================================================================

def test_missing_openai_key_visibly_labels_mock_mode(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    resp = client.post("/api/generate", json={
        "language": "python",
        "code": "def add(a, b): return a + b"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "demo_mock"
    assert data["model"] == "mock-engine"
    assert "DEMO MOCK MODE" in data["warning"]


def test_gemini_generation_success(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.text = '"""Module doc."""\ndef add(a, b):\n    """Adds a and b."""\n    return a + b\n'
        mock_instance.models.generate_content.return_value = mock_response

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "gemini"
        assert data["model"] == "gemini-3.8-flash"
        assert data["warning"] is None


def test_gemini_auth_error_does_not_silently_switch_to_mock(monkeypatch):
    from google.genai.errors import APIError as GenAIAPIError
    monkeypatch.setenv("GEMINI_API_KEY", "invalid-gemini-key")

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_instance.models.generate_content.side_effect = GenAIAPIError(
            401, "API_KEY_INVALID: Invalid API key"
        )

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 401
        assert "gemini authentication failed" in resp.json()["detail"].lower()


def test_gemini_rate_limit_error_does_not_silently_switch_to_mock(monkeypatch):
    from google.genai.errors import APIError as GenAIAPIError
    monkeypatch.setenv("GEMINI_API_KEY", "valid-gemini-key")

    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_instance.models.generate_content.side_effect = GenAIAPIError(
            429, "RESOURCE_EXHAUSTED: Rate limit reached"
        )

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 429
        assert "gemini rate limit" in resp.json()["detail"].lower()


def test_openai_auth_error_does_not_silently_switch_to_mock(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-invalid-test-key")

    with patch("app.main.OpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_instance.chat.completions.create.side_effect = AuthenticationError(
            message="Incorrect API key provided",
            response=MagicMock(status_code=401),
            body={"message": "Incorrect API key provided"}
        )

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 401
        assert "authentication failed" in resp.json()["detail"].lower()


def test_openai_rate_limit_error_does_not_silently_switch_to_mock(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-key")

    with patch("app.main.OpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_instance.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit reached or quota exceeded",
            response=MagicMock(status_code=429),
            body={"message": "Rate limit reached"}
        )

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 429
        assert "rate limit" in resp.json()["detail"].lower()


# ==============================================================================
# Phase 5: Approval and Rejection Flow Tests
# ==============================================================================

def test_rejection_workflow(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # 1. Generate run
    gen = client.post("/api/generate", json={
        "language": "python",
        "code": LANG_SAMPLES["python"]
    })
    assert gen.status_code == 200
    run_id = gen.json()["run_id"]

    # 2. Reject run
    rej = client.post("/api/reject", json={
        "run_id": run_id,
        "reason": "Docstrings are not detailed enough for our standards."
    })
    assert rej.status_code == 200
    rej_data = rej.json()
    assert rej_data["status"] == "rejected"
    assert "detailed enough" in rej_data["rejection_reason"]
    assert rej_data["reviewed_at"] is not None

    # 3. Repeated rejection or approval should return 409 Conflict
    rep_rej = client.post("/api/reject", json={"run_id": run_id})
    assert rep_rej.status_code == 409

    rep_appr = client.post("/api/approve", json={"run_id": run_id, "code": "pass"})
    assert rep_appr.status_code == 409


def test_approval_workflow_persists_proposal(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    gen = client.post("/api/generate", json={
        "language": "python",
        "code": LANG_SAMPLES["python"]
    })
    run_id = gen.json()["run_id"]
    proposed = gen.json()["proposed_code"]

    # User modifies proposal
    user_edited = proposed + "\n# Extra custom reviewer note"

    appr = client.post("/api/approve", json={
        "run_id": run_id,
        "code": user_edited
    })
    assert appr.status_code == 200
    data = appr.json()
    assert data["status"] == "approved_verified"
    assert data["final_code"] == user_edited
    assert data["verification"]["passed"] is True

    # Check persistence in history item
    item_resp = client.get(f"/api/history/{run_id}")
    assert item_resp.status_code == 200
    item = item_resp.json()
    assert item["final_code"] == user_edited
    assert item["status"] == "approved_verified"


def test_invalid_run_id_returns_404():
    appr = client.post("/api/approve", json={"run_id": "non-existent-id", "code": "test"})
    assert appr.status_code == 404

    rej = client.post("/api/reject", json={"run_id": "non-existent-id"})
    assert rej.status_code == 404


def test_failed_verification_does_not_claim_verified(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    gen = client.post("/api/generate", json={
        "language": "python",
        "code": LANG_SAMPLES["python"]
    })
    run_id = gen.json()["run_id"]

    # Send syntax-broken code for approval
    broken_code = "def broken(:\n    pass"
    appr = client.post("/api/approve", json={
        "run_id": run_id,
        "code": broken_code
    })
    assert appr.status_code == 200
    data = appr.json()
    assert data["status"] == "approved_checks_failed"
    assert data["verification"]["passed"] is False


# ==============================================================================
# Phase 6 & 8: Endpoints, Health, Languages, History
# ==============================================================================

def test_api_health_and_config():
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert health.json()["database"] == "connected"

    cfg = client.get("/api/config")
    assert cfg.status_code == 200
    assert "model" in cfg.json()
    assert "languages" in cfg.json()


def test_history_pagination(monkeypatch):
    hist = client.get("/api/history?limit=5&offset=0")
    assert hist.status_code == 200
    items = hist.json().get("items", [])
    assert isinstance(items, list)
    assert len(items) <= 5


def test_live_openai_generation_respects_configured_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-valid-live-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")

    with patch("app.main.OpenAI") as mock_openai_cls:
        mock_instance = MagicMock()
        mock_openai_cls.return_value = mock_instance
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = '"""Module doc."""\ndef add(a, b):\n    """Adds a and b."""\n    return a + b\n'
        mock_completion.choices = [mock_choice]
        mock_instance.chat.completions.create.return_value = mock_completion

        resp = client.post("/api/generate", json={
            "language": "python",
            "code": "def add(a, b): return a + b"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["mode"] == "openai"
        assert data["model"] == "gpt-4.1-mini"
        assert data["warning"] is None

        # Confirm the configured model name was passed directly to OpenAI SDK
        _, kwargs = mock_instance.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4.1-mini"


def test_structural_consistency_catches_executable_statement_changes():
    # 1. Python executable alteration (+ changed to -)
    py_orig = "def calculate(a, b):\n    return a + b\n"
    py_mutated = 'def calculate(a, b):\n    """Docstring added."""\n    return a - b\n'
    v_py_mutated = verify(py_mutated, "python", original_code=py_orig)
    assert v_py_mutated["passed"] is False
    assert v_py_mutated["global_status"] == "failed"
    chk_py = next(c for c in v_py_mutated["checks"] if c["name"] == "Structural Consistency")
    assert chk_py["status"] == "failed"
    assert "Executable statements or logic were modified" in chk_py["details"]

    # 2. Python documentation-only addition (passes)
    py_valid_doc = 'def calculate(a, b):\n    """Docstring added."""\n    return a + b\n'
    v_py_doc = verify(py_valid_doc, "python", original_code=py_orig)
    assert v_py_doc["passed"] is True
    assert v_py_doc["global_status"] == "verified"
    chk_py_doc = next(c for c in v_py_doc["checks"] if c["name"] == "Structural Consistency")
    assert chk_py_doc["status"] == "passed"

    # 3. JavaScript executable alteration (* changed to /)
    js_orig = "function calc(a, b) { return a * b; }"
    js_mutated = "/** Calculates. */\nfunction calc(a, b) {\n  return a / b;\n}"
    v_js_mutated = verify(js_mutated, "javascript", original_code=js_orig)
    assert v_js_mutated["passed"] is False
    chk_js = next(c for c in v_js_mutated["checks"] if c["name"] == "Structural Consistency")
    assert chk_js["status"] == "failed"
    assert "Executable code modified" in chk_js["details"]


def test_isolated_temporary_database_workflow(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_docupilot.db")
    monkeypatch.setenv("DATABASE_PATH", test_db)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    from app import store
    store.init_db()

    # Initial stats should be zero
    stats_init = store.get_stats()
    assert stats_init["total_runs"] == 0
    assert stats_init["approved"] == 0
    assert stats_init["rejected"] == 0

    # 1. Generate run
    gen = client.post("/api/generate", json={
        "language": "python",
        "code": "def sample(x): return x * 2"
    })
    assert gen.status_code == 200
    run_id = gen.json()["run_id"]

    # Stats should show 1 total, 1 pending
    stats_after_gen = store.get_stats()
    assert stats_after_gen["total_runs"] == 1
    assert stats_after_gen["pending"] == 1

    # 2. Approve run with reviewed proposal
    reviewed_code = gen.json()["proposed_code"] + "\n# Approved by QA"
    appr = client.post("/api/approve", json={
        "run_id": run_id,
        "code": reviewed_code
    })
    assert appr.status_code == 200
    assert appr.json()["final_code"] == reviewed_code

    stats_after_appr = store.get_stats()
    assert stats_after_appr["approved"] == 1
    assert stats_after_appr["pending"] == 0

    # 3. Verify duplicate transition rejected with 409
    dup = client.post("/api/approve", json={"run_id": run_id, "code": reviewed_code})
    assert dup.status_code == 409

    dup_rej = client.post("/api/reject", json={"run_id": run_id})
    assert dup_rej.status_code == 409

    # 4. Generate second run to test rejection
    gen2 = client.post("/api/generate", json={
        "language": "python",
        "code": "def sample2(x): return x * 3"
    })
    run_id2 = gen2.json()["run_id"]
    rej = client.post("/api/reject", json={
        "run_id": run_id2,
        "reason": "Security review failed: sensitive parameter not sanitized."
    })
    assert rej.status_code == 200
    assert rej.json()["status"] == "rejected"

    stats_final = store.get_stats()
    assert stats_final["total_runs"] == 2
    assert stats_final["approved"] == 1
    assert stats_final["rejected"] == 1
    assert stats_final["pending"] == 0

