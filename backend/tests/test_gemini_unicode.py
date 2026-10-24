"""Tests for Gemini Unicode handling, error classification, and secret sanitization."""

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app, sanitize_error
from app.agents.documentation import DocumentationGenerationAgent, build_generation_prompt


client = TestClient(app)


def test_sanitize_error_redacts_keys():
    """Verify API keys are never exposed in error messages."""
    raw_error_1 = "Google error: key=AQ.test_mock_key_123456789012345678901234567890 invalid"
    raw_error_2 = "Failed with AIzaSyD9u1234567890abcdef1234567890abcde in request"
    raw_error_3 = "OpenAI key sk-proj1234567890abcdef1234567890 rejected"

    clean_1 = sanitize_error(raw_error_1)
    clean_2 = sanitize_error(raw_error_2)
    clean_3 = sanitize_error(raw_error_3)

    assert "AQ.test_mock_key_123456789012345678901234567890" not in clean_1
    assert "AIzaSyD9u1234567890abcdef1234567890abcde" not in clean_2
    assert "sk-proj1234567890abcdef1234567890" not in clean_3
    assert "[REDACTED_API_KEY]" in clean_1
    assert "[REDACTED_API_KEY]" in clean_2
    assert "[REDACTED_API_KEY]" in clean_3


def test_build_generation_prompt_preserves_unicode():
    """Verify build_generation_prompt preserves em dashes, quotes, and non-English scripts."""
    code_with_unicode = (
        "# Calculation — computes discounted price\n"
        "# Rules: “Special discount applies”\n"
        "def calcular_descuento(precio: float, tasa: float) -> float:\n"
        "    return precio * (1 - tasa)\n"
    )
    prompt = build_generation_prompt(
        code=code_with_unicode,
        language="python",
        analysis={"functions": [{"name": "calcular_descuento", "params": ["precio", "tasa"]}]},
        doc_style="google",
    )

    assert "—" in prompt
    assert "“" in prompt
    assert "”" in prompt
    assert "calcular_descuento" in prompt


def test_gemini_generation_with_em_dash_success():
    """Verify successful Gemini generation with em dash (mocked API client)."""
    mock_response = MagicMock()
    mock_response.text = (
        'def process_data(items):\n'
        '    """Process item pipeline — filters and returns active items."""\n'
        '    return [x for x in items if x > 0]\n'
    )

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client_instance):

        agent = DocumentationGenerationAgent()
        code_input = (
            "# Main processor — pipeline entry point\n"
            "def process_data(items):\n"
            "    return [x for x in items if x > 0]\n"
        )
        res = agent.run(
            code=code_input,
            language="python",
            analysis={"functions": [{"name": "process_data", "params": ["items"]}]},
            doc_style="google",
        )

        assert res.status == "completed"
        assert res.mode == "gemini"
        assert "—" in res.proposed_code


def test_gemini_unicode_smart_quotes_and_multiline():
    """Verify handling of smart quotes, accented characters, and multiline docstrings."""
    mock_response = MagicMock()
    mock_response.text = (
        'def compute_metric(x: float, y: float) -> float:\n'
        '    """Calculates weighted metric “alpha” — based on standard deviations.\n'
        '\n'
        '    Args:\n'
        '        x (float): Première coordonnée.\n'
        '        y (float): Seconde coordonnée.\n'
        '\n'
        '    Returns:\n'
        '        float: Résultat normalisé.\n'
        '    """\n'
        '    return (x * 0.4) + (y * 0.6)\n'
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        input_code = (
            "# Calcul de métriques “alpha” — analyse de données\n"
            "def compute_metric(x: float, y: float) -> float:\n"
            "    return (x * 0.4) + (y * 0.6)\n"
        )
        res = agent.run(
            code=input_code,
            language="python",
            analysis={"functions": [{"name": "compute_metric", "params": ["x", "y"]}]},
            doc_style="google",
        )

        assert res.status == "completed"
        assert "“alpha”" in res.proposed_code
        assert "—" in res.proposed_code


def test_gemini_auth_error_handling():
    """Verify Gemini 401 AuthenticationError is properly categorized and sanitized."""
    from google.genai import errors

    mock_err = errors.APIError(401, "API_KEY_INVALID: key AQ.test_mock_key_123456789012345678901234567890 is not valid")
    mock_err.code = 401

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_err

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def test_fn(): pass",
                language="python",
                analysis={},
            )

        assert exc_info.value.status_code == 401
        assert "AQ.test_mock_key" not in str(exc_info.value.detail)
        assert "[REDACTED_API_KEY]" in str(exc_info.value.detail)
        assert "authentication failed" in str(exc_info.value.detail)


def test_gemini_rate_limit_error_handling():
    """Verify Gemini 429 RateLimit/Quota error is cleanly returned as HTTP 429."""
    from google.genai import errors

    mock_err = errors.APIError(429, "RESOURCE_EXHAUSTED: Rate limit reached for model gemini-3.5-flash")
    mock_err.code = 429

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_err

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def test_fn(): pass",
                language="python",
                analysis={},
            )

        assert exc_info.value.status_code == 429
        assert "rate limit or quota exceeded" in str(exc_info.value.detail)


def test_gemini_model_not_found_handling():
    """Verify Gemini 404 Model Not Found error returns HTTP 404 with clear message."""
    from google.genai import errors

    mock_err = errors.APIError(404, "NOT_FOUND: models/nonexistent-model is not found")
    mock_err.code = 404

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_err

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def test_fn(): pass",
                language="python",
                analysis={},
            )

        assert exc_info.value.status_code == 404
        assert "not supported or not available" in str(exc_info.value.detail)


def test_gemini_unicode_error_handling():
    """Verify UnicodeEncodeError is cleanly categorized as HTTP 422 with explanation."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = UnicodeEncodeError("ascii", "test — string", 5, 6, "ordinal not in range(128)")

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def test_fn(): pass",
                language="python",
                analysis={},
            )

        assert exc_info.value.status_code == 422
        assert "Unicode encoding error" in str(exc_info.value.detail)


def test_api_generate_endpoint_with_unicode_contract():
    """Verify the /api/generate endpoint contract preserves Unicode and returns structured response."""
    code_with_em_dash = (
        "# High-performance sorter — uses TimSort\n"
        "def sort_items(items: list) -> list:\n"
        "    return sorted(items)\n"
    )

    mock_resp = MagicMock()
    mock_resp.text = (
        'def sort_items(items: list) -> list:\n'
        '    """Sorts items in ascending order — using TimSort."""\n'
        '    return sorted(items)\n'
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        resp = client.post(
            "/api/generate",
            json={"code": code_with_em_dash, "language": "python", "doc_style": "google"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "pending_review"
    assert "proposed_code" in data
    assert "analysis" in data
    assert "verification" in data
    assert "stages" in data
    assert "agent_executions" in data
    # Confirm original code with em dash was preserved in response
    assert "—" in data["original_code"]


def test_gemini_default_model_is_gemini_38_flash():
    """Verify gemini-3.8-flash is the configured default when GEMINI_MODEL is unset."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "AQ.dummy_key_123456789012345678901234567890", "GEMINI_MODEL": "", "LLM_PROVIDER": "gemini"}):
        from app.main import get_configured_model
        assert get_configured_model() == "gemini-3.8-flash"


def test_no_silent_mock_fallback_on_gemini_failure():
    """Verify that when Gemini is configured, API failures raise an error and do NOT return mock output."""
    from google.genai import errors

    mock_err = errors.APIError(500, "Internal Server Error in Gemini API")
    mock_err.code = 500

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_err

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def do_work(): return 42",
                language="python",
                analysis={},
            )

        # Must raise HTTPException preserving upstream status code (500) and not return mock code
        assert exc_info.value.status_code == 500
        assert "Google Gemini API error" in str(exc_info.value.detail)


def test_gemini_503_high_demand_handling():
    """Verify 503 UNAVAILABLE is categorized as 503 with user-friendly retry message."""
    from google.genai import errors

    mock_err = errors.APIError(503, "UNAVAILABLE: This model is currently experiencing high demand. Please try again later.")
    mock_err.code = 503

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_err

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationGenerationAgent()
        with pytest.raises(HTTPException) as exc_info:
            agent.generate_with_gemini(
                code="def compute(): return 1",
                language="python",
                analysis={},
            )

        assert exc_info.value.status_code == 503
        assert "temporarily experiencing high demand" in str(exc_info.value.detail)


def test_review_agent_with_unicode_and_gemini_audit():
    """Verify DocumentationReviewAgent audits code with Unicode punctuation without error."""
    from app.agents.review import DocumentationReviewAgent

    code_with_unicode = (
        '# Sorter — sorts elements using “custom” comparator\n'
        'def custom_sort(items: list) -> list:\n'
        '    """Custom sorter — arranges items based on priority."""\n'
        '    return sorted(items)\n'
    )

    mock_audit_resp = MagicMock()
    mock_audit_resp.text = json.dumps({
        "grounded": True,
        "findings": ["Well documented with custom — description."],
        "hallucinations": [],
        "actionable_feedback": None
    })
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_audit_resp

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        agent = DocumentationReviewAgent()
        res = agent.run(
            original_code=code_with_unicode,
            proposed_code=code_with_unicode,
            language="python",
            analysis={"symbols": [{"name": "custom_sort", "type": "FunctionDef", "parameters": ["items"]}]},
            doc_style="google",
        )

        assert res.status == "passed"
        assert res.review_type == "independent_llm"
        assert any("—" in f for f in res.findings)


def test_orchestrator_multi_agent_workflow_with_unicode_and_revision():
    """Verify Orchestrator workflow preserves Unicode across generation, review, and revision."""
    from app.orchestrator import Orchestrator

    code_input = (
        '# Financial engine — tax & interest calculation\n'
        'def compute_tax(income: float, rate: float) -> float:\n'
        '    """Compute tax — based on “standard” deduction."""\n'
        '    return income * rate\n'
    )

    mock_gen_resp = MagicMock()
    mock_gen_resp.text = (
        'def compute_tax(income: float, rate: float) -> float:\n'
        '    """Calculates tax — based on bracket and “standard” deduction.\n'
        '\n'
        '    Args:\n'
        '        income (float): Gross taxable income.\n'
        '        rate (float): Applicable tax bracket.\n'
        '\n'
        '    Returns:\n'
        '        float: Net computed tax obligation.\n'
        '    """\n'
        '    return income * rate\n'
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_gen_resp

    with patch("app.agents.documentation.get_gemini_key", return_value="AQ.test_mock_key_123456789012345678901234567890"), \
         patch("app.agents.documentation.get_active_provider", return_value="gemini"), \
         patch("google.genai.Client", return_value=mock_client):

        orch = Orchestrator()
        result = orch.process_code_generation(
            code=code_input,
            language="python",
            doc_style="google",
        )

        assert result["status"] == "pending_review"
        assert "—" in result["original_code"]
        assert "—" in result["proposed_code"]
        assert "“standard”" in result["proposed_code"]
        assert result["workflow_state"] == "AWAITING_APPROVAL"


def test_http_exception_not_rewrapped_into_502():
    """Verify HTTP exceptions (401, 404, 422, 429, 503) from Gemini are not converted into generic 502."""
    from google.genai import errors

    test_cases = [
        (401, "API_KEY_INVALID: bad key", 401),
        (404, "NOT_FOUND: model not found", 404),
        (429, "RESOURCE_EXHAUSTED: quota reached", 429),
        (503, "UNAVAILABLE: high demand", 503),
    ]

    for err_code, err_msg, expected_status in test_cases:
        mock_err = errors.APIError(err_code, err_msg)
        mock_err.code = err_code

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = mock_err

        with patch("google.genai.Client", return_value=mock_client):
            resp = client.post(
                "/api/generate",
                json={"code": "def example(): pass", "language": "python", "doc_style": "google"},
            )
            assert resp.status_code == expected_status, f"Expected status {expected_status} but got {resp.status_code} for {err_msg}"
            assert resp.status_code != 502, f"Status code was incorrectly wrapped into 502 for {err_msg}"


def test_gemini_model_override_via_env():
    """Verify GEMINI_MODEL env override is respected and can be configured dynamically."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "AQ.dummy_key_123456789012345678901234567890", "GEMINI_MODEL": "gemini-custom-model-v2", "LLM_PROVIDER": "gemini"}):
        from app.main import get_configured_model
        assert get_configured_model() == "gemini-custom-model-v2"


def test_ts_runner_unicode_subprocess_handling():
    """Verify validate_typescript handles em dashes and curly quotes without UnicodeEncodeError in subprocess."""
    from app.ts_runner import validate_typescript

    ts_code = (
        '// Helper — computes “tax” rates\n'
        'export function computeTax(income: number, rate: number): number {\n'
        '    return income * rate;\n'
        '}\n'
    )
    res = validate_typescript(ts_code)
    # Even if Node/TS is not installed, it must NOT raise UnicodeEncodeError
    assert isinstance(res, dict)
    assert "status" in res


def test_sqlite_unicode_roundtrip_preservation(tmp_path, monkeypatch):
    """Verify SQLite persistence preserves raw Unicode characters (em dashes, quotes) without ASCII escaping."""
    from app import store
    test_db = str(tmp_path / "unicode_test.db")
    monkeypatch.setenv("DATABASE_PATH", test_db)
    store.init_db()

    unicode_source = '# Pipeline — initial “stage”\ndef process(): pass'
    unicode_proposed = '# Pipeline — documented “stage”\ndef process(): pass'
    unicode_analysis = {"symbols": [{"name": "process", "description": "Action — execute “workflow”"}]}

    run_id = store.save_run(
        language="python",
        original=unicode_source,
        proposed=unicode_proposed,
        analysis=unicode_analysis,
        mode="gemini",
        model="gemini-3.8-flash",
    )

    saved = store.get_run(run_id)
    assert saved is not None
    assert "—" in saved["original_code"]
    assert "“stage”" in saved["original_code"]
    assert "—" in saved["proposed_code"]
    assert "“workflow”" in str(saved["analysis"])


def test_store_resilient_to_readonly_database(tmp_path, monkeypatch):
    """Verify store recovers safely when running in read-only filesystems (e.g. Vercel Serverless)."""
    import os
    import sqlite3
    from app import store

    ro_db = str(tmp_path / "ro_test.db")
    con = sqlite3.connect(ro_db)
    con.execute("CREATE TABLE runs (id TEXT PRIMARY KEY)")
    con.commit()
    con.close()
    os.chmod(ro_db, 0o444)

    monkeypatch.setenv("DATABASE_PATH", ro_db)
    run_id = store.save_run(
        language="python",
        original="def example(): pass",
        proposed="def example(): pass",
        analysis={},
        mode="demo_mock",
    )
    assert run_id is not None


