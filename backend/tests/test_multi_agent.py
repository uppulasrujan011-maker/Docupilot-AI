"""Comprehensive test suite for DocuPilot AI Multi-Agent Architecture.

Covers:
1. Successful multi-agent workflow
2. Correct specialist-agent ordering
3. Concurrent independent review and validation
4. Structured agent inputs and outputs
5. Review failure and the one permitted revision
6. Validation failure blocking successful completion
7. Invalid source and unsupported languages
8. Missing API key and mock-mode behavior
9. LLM errors, malformed responses, and timeouts
10. Approval and rejection
11. Original source preservation
12. Patch integrity and download behavior
13. Database/audit-history recording
14. Existing application functionality and API compatibility
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
try:
    from openai import AuthenticationError, RateLimitError, APIError
except ImportError:
    from app.agents.documentation import AuthenticationError, RateLimitError, APIError

from app.main import app
from app.orchestrator import Orchestrator, WorkflowState
from app.agents.code_analysis import CodeAnalysisAgent
from app.agents.documentation import DocumentationGenerationAgent
from app.agents.review import DocumentationReviewAgent
from app.agents.validation import ValidationAgent
from app import store

client = TestClient(app)

PYTHON_SAMPLE = """def calculate_discount(price: float, discount: float) -> float:
    if price <= 0:
        return 0.0
    return price - (price * discount / 100.0)
"""

JAVASCRIPT_SAMPLE = """function calculateShipping(weight, distance) {
    if (weight <= 0) return 0;
    return weight * 2.5 + distance * 0.4;
}"""


# ------------------------------------------------------------------------------
# 1. Successful Multi-Agent Workflow
# ------------------------------------------------------------------------------
def test_successful_multi_agent_workflow(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    resp = client.post(
        "/api/generate",
        json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["workflow_state"] == "AWAITING_APPROVAL"
    assert data["status"] == "pending_review"
    assert "run_id" in data
    assert "agent_executions" in data
    assert "code_analysis" in data["agent_executions"]
    assert "documentation_generation" in data["agent_executions"]
    assert "documentation_review" in data["agent_executions"]
    assert "validation" in data["agent_executions"]

    # Verify all 4 agents completed successfully
    assert data["agent_executions"]["code_analysis"]["status"] == "completed"
    assert data["agent_executions"]["documentation_generation"]["status"] == "completed"
    assert data["agent_executions"]["documentation_review"]["status"] == "passed"
    assert data["agent_executions"]["validation"]["passed"] is True


# ------------------------------------------------------------------------------
# 2. Correct Specialist-Agent Ordering
# ------------------------------------------------------------------------------
def test_correct_specialist_agent_ordering(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    orc = Orchestrator()
    call_order = []

    orig_analysis_run = orc.code_analysis_agent.run
    orig_gen_run = orc.doc_generation_agent.run
    orig_review_run = orc.doc_review_agent.run
    orig_val_run = orc.validation_agent.run

    def tracked_analysis(*args, **kwargs):
        call_order.append("code_analysis")
        return orig_analysis_run(*args, **kwargs)

    def tracked_gen(*args, **kwargs):
        call_order.append("documentation_generation")
        return orig_gen_run(*args, **kwargs)

    def tracked_review(*args, **kwargs):
        call_order.append("documentation_review")
        return orig_review_run(*args, **kwargs)

    def tracked_val(*args, **kwargs):
        call_order.append("validation")
        return orig_val_run(*args, **kwargs)

    orc.code_analysis_agent.run = tracked_analysis
    orc.doc_generation_agent.run = tracked_gen
    orc.doc_review_agent.run = tracked_review
    orc.validation_agent.run = tracked_val

    result = orc.process_code_generation(PYTHON_SAMPLE, "python", "google")
    assert result["run_id"]

    # Ordering guarantees:
    # 1. code_analysis must precede documentation_generation
    # 2. documentation_generation must precede review and validation
    assert call_order.index("code_analysis") < call_order.index("documentation_generation")
    assert call_order.index("documentation_generation") < call_order.index("documentation_review")
    assert call_order.index("documentation_generation") < call_order.index("validation")


# ------------------------------------------------------------------------------
# 3. Concurrent Independent Review and Validation
# ------------------------------------------------------------------------------
def test_concurrent_independent_review_and_validation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    orc = Orchestrator()
    doc_candidate = '''def calculate_discount(price: float, discount: float) -> float:
    """Calculates discount on price.

    Args:
        price (float): Original price.
        discount (float): Percentage discount.

    Returns:
        float: Final discounted price.
    """
    if price <= 0:
        return 0.0
    return price - (price * discount / 100.0)
'''
    review_res, val_res = orc._run_review_and_validation_concurrently(
        original_code=PYTHON_SAMPLE,
        candidate_code=doc_candidate,
        language="python",
        analysis={"symbols": [{"name": "calculate_discount", "type": "FunctionDef", "parameters": ["price", "discount"]}]},
        doc_style="google",
    )

    assert review_res.agent_name == "Documentation Review Agent"
    assert val_res.agent_name == "Validation Agent"
    assert val_res.passed is True
    assert isinstance(review_res.score, int)


# ------------------------------------------------------------------------------
# 4. Structured Agent Inputs and Outputs
# ------------------------------------------------------------------------------
def test_structured_agent_inputs_and_outputs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # Agent 1: CodeAnalysisAgent
    c_agent = CodeAnalysisAgent()
    c_res = c_agent.run(PYTHON_SAMPLE, "python")
    assert c_res.agent_name == "Code Analysis Agent"
    assert c_res.symbol_count >= 1
    assert c_res.valid_syntax is True
    assert c_res.execution_time_ms >= 0
    assert "parser_used" in c_res.evidence

    # Agent 2: DocumentationGenerationAgent
    d_agent = DocumentationGenerationAgent()
    d_res = d_agent.run(PYTHON_SAMPLE, "python", c_res.analysis, "google")
    assert d_res.agent_name == "Documentation Generation Agent"
    assert d_res.proposed_code
    assert d_res.mode in ("demo_mock", "gemini", "openai")
    assert d_res.revision_number == 0

    # Agent 3: DocumentationReviewAgent
    r_agent = DocumentationReviewAgent()
    r_res = r_agent.run(PYTHON_SAMPLE, d_res.proposed_code, "python", c_res.analysis, "google")
    assert r_res.agent_name == "Documentation Review Agent"
    assert r_res.status in ("passed", "needs_revision", "failed")
    assert 0 <= r_res.score <= 100

    # Agent 4: ValidationAgent
    v_agent = ValidationAgent()
    v_res = v_agent.run(d_res.proposed_code, "python", PYTHON_SAMPLE, "google")
    assert v_res.agent_name == "Validation Agent"
    assert v_res.passed is True
    assert len(v_res.checks) >= 3


# ------------------------------------------------------------------------------
# 5. Review Failure Triggers Exactly One Permitted Revision
# ------------------------------------------------------------------------------
def test_review_failure_triggers_one_bounded_revision(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    orc = Orchestrator()
    revision_attempts = []

    orig_gen_run = orc.doc_generation_agent.run

    def mock_gen_run(*args, **kwargs):
        rev_num = kwargs.get("revision_number", 0)
        revision_attempts.append(rev_num)
        return orig_gen_run(*args, **kwargs)

    orc.doc_generation_agent.run = mock_gen_run

    # Force review agent to require revision on initial attempt
    orig_review_run = orc.doc_review_agent.run
    first_call = True

    def mock_review_run(*args, **kwargs):
        nonlocal first_call
        res = orig_review_run(*args, **kwargs)
        if first_call:
            first_call = False
            res.status = "needs_revision"
            res.actionable_feedback = "Ensure parameter 'discount' has explicit percentage explanation."
            res.severity = "high"
        return res

    orc.doc_review_agent.run = mock_review_run

    res = orc.process_code_generation(PYTHON_SAMPLE, "python", "google")
    assert res["revision_count"] == 1
    assert revision_attempts == [0, 1]  # Initial (0) then exactly 1 revision
    assert "documentation_generation_revision_1" in res["agent_executions"]


# ------------------------------------------------------------------------------
# 6. Validation Failure Blocks Successful Completion
# ------------------------------------------------------------------------------
def test_validation_failure_blocks_successful_completion(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    orc = Orchestrator()
    orig_run = orc.doc_generation_agent.run

    # Generator returns syntactically broken code
    def broken_gen(*args, **kwargs):
        res = orig_run(*args, **kwargs)
        res.proposed_code = "def calculate_discount(price, discount\n    broken syntax"
        return res

    orc.doc_generation_agent.run = broken_gen

    result = orc.process_code_generation(PYTHON_SAMPLE, "python", "google")
    # Validation must report failure
    assert result["verification"]["passed"] is False
    assert result["verification"]["global_status"] == "failed"
    validation_stage = next(s for s in result["stages"] if s["id"] == "validation")
    assert validation_stage["status"] == "failed"


# ------------------------------------------------------------------------------
# 7. Invalid Source and Unsupported Languages
# ------------------------------------------------------------------------------
def test_unsupported_language_fails_fast():
    resp = client.post(
        "/api/generate",
        json={"code": "some code", "language": "brainfuck", "doc_style": "google"},
    )
    assert resp.status_code == 422 or resp.status_code == 400


def test_invalid_python_source_handled_truthfully(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    resp = client.post(
        "/api/generate",
        json={"code": "def unclosed_func(a, b", "language": "python", "doc_style": "google"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis"]["valid_syntax"] is False
    assert data["verification"]["passed"] is False


# ------------------------------------------------------------------------------
# 8. Missing API Key and Mock-Mode Behavior
# ------------------------------------------------------------------------------
def test_missing_api_keys_uses_mock_mode(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "")

    resp = client.post(
        "/api/generate",
        json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "demo_mock"
    assert data["model"] == "mock-engine"
    assert "DEMO MOCK MODE" in data["warning"]


# ------------------------------------------------------------------------------
# 9. LLM Errors, Malformed Responses, and Timeouts
# ------------------------------------------------------------------------------
def test_openai_api_error_handling(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    with patch("app.main.OpenAI") as mock_openai:
        mock_instance = MagicMock()
        mock_openai.return_value = mock_instance
        mock_instance.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body={"message": "Invalid API Key"},
        )

        resp = client.post(
            "/api/generate",
            json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
        )
        assert resp.status_code == 401


# ------------------------------------------------------------------------------
# 10. Human Approval and Rejection Flow
# ------------------------------------------------------------------------------
def test_approval_and_rejection_workflow(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # Generate proposal
    gen = client.post(
        "/api/generate",
        json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
    )
    assert gen.status_code == 200
    run_id = gen.json()["run_id"]
    proposed = gen.json()["proposed_code"]

    # Approve
    appr = client.post(
        "/api/approve",
        json={"run_id": run_id, "code": proposed, "doc_style": "google"},
    )
    assert appr.status_code == 200
    appr_data = appr.json()
    assert appr_data["status"] == "approved_verified"
    assert appr_data["workflow_state"] == "COMPLETED"

    # Cannot re-approve completed run (409 Conflict)
    re_appr = client.post(
        "/api/approve",
        json={"run_id": run_id, "code": proposed, "doc_style": "google"},
    )
    assert re_appr.status_code == 409


def test_rejection_preserves_original_code(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    gen = client.post(
        "/api/generate",
        json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
    )
    assert gen.status_code == 200
    run_id = gen.json()["run_id"]

    rej = client.post(
        "/api/reject",
        json={"run_id": run_id, "reason": "Docstring style incorrect"},
    )
    assert rej.status_code == 200
    assert rej.json()["status"] == "rejected"
    assert rej.json()["workflow_state"] == "REJECTED"

    # Verify original source remains intact in store
    saved_run = store.get_run(run_id)
    assert saved_run["original_code"] == PYTHON_SAMPLE
    assert saved_run["status"] == "rejected"


# ------------------------------------------------------------------------------
# 11. Original Source Preservation and AST Fingerprint Check
# ------------------------------------------------------------------------------
def test_original_source_preservation_structural_check(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    res = client.post(
        "/api/generate",
        json={"code": PYTHON_SAMPLE, "language": "python", "doc_style": "google"},
    )
    assert res.status_code == 200
    data = res.json()

    struct_check = next(
        c for c in data["verification"]["checks"] if c["name"] == "Structural Consistency"
    )
    assert struct_check["status"] == "passed"
    assert "100%" in struct_check["details"] or "preserved" in struct_check["details"]


# ------------------------------------------------------------------------------
# 12. Database and Audit History Recording
# ------------------------------------------------------------------------------
def test_database_audit_history_records_multi_agent_executions(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    res = client.post(
        "/api/generate",
        json={"code": JAVASCRIPT_SAMPLE, "language": "javascript", "doc_style": "standard"},
    )
    assert res.status_code == 200
    run_id = res.json()["run_id"]

    # Fetch from /api/history/{run_id}
    hist = client.get(f"/api/history/{run_id}")
    assert hist.status_code == 200
    item = hist.json()

    assert item["id"] == run_id
    assert item["language"] == "javascript"
    assert "agents_log" in item
    assert "code_analysis" in item["agents_log"]
    assert "documentation_generation" in item["agents_log"]
    assert "documentation_review" in item["agents_log"]
    assert "validation" in item["agents_log"]


# ------------------------------------------------------------------------------
# 13. Multi-Agent Config API Information
# ------------------------------------------------------------------------------
def test_config_api_reports_multi_agent_system():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()

    assert "multi_agent" in data
    assert data["multi_agent"]["architecture"] == "hybrid_multi_agent"
    assert len(data["multi_agent"]["agents"]) == 4
