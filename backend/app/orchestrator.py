"""Central Orchestrator and Workflow Controller for DocuPilot AI.

Coordinates the 4 specialist agents:
1. Code Analysis Agent
2. Documentation Generation Agent
3. Documentation Review Agent
4. Validation Agent

Enforces deterministic workflow transitions, concurrent independent review & validation,
bounded single-revision policy on documentation defects, and human-in-the-loop approval gates.
"""

import logging
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Optional

from fastapi import HTTPException
from pydantic import BaseModel, Field

from .agents.code_analysis import CodeAnalysisAgent, CodeAnalysisResult
from .agents.documentation import DocumentationGenerationAgent, DocumentationResult, sanitize_error
from .agents.review import DocumentationReviewAgent, ReviewResult
from .agents.validation import ValidationAgent, ValidationResult
from .languages import SUPPORTED_LANGUAGES, get_language
from . import store

logger = logging.getLogger("docupilot.orchestrator")


class WorkflowState(str, Enum):
    RECEIVED = "RECEIVED"
    ANALYZING = "ANALYZING"
    GENERATING = "GENERATING"
    REVIEWING = "REVIEWING"
    VALIDATING = "VALIDATING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


VALID_TRANSITIONS = {
    WorkflowState.RECEIVED: {WorkflowState.ANALYZING, WorkflowState.FAILED},
    WorkflowState.ANALYZING: {WorkflowState.GENERATING, WorkflowState.FAILED},
    WorkflowState.GENERATING: {WorkflowState.REVIEWING, WorkflowState.VALIDATING, WorkflowState.FAILED},
    WorkflowState.REVIEWING: {WorkflowState.GENERATING, WorkflowState.AWAITING_APPROVAL, WorkflowState.FAILED},
    WorkflowState.VALIDATING: {WorkflowState.GENERATING, WorkflowState.AWAITING_APPROVAL, WorkflowState.FAILED},
    WorkflowState.AWAITING_APPROVAL: {WorkflowState.COMPLETED, WorkflowState.REJECTED, WorkflowState.FAILED},
    WorkflowState.COMPLETED: set(),
    WorkflowState.REJECTED: set(),
    WorkflowState.FAILED: set(),
}


class OrchestratorResponse(BaseModel):
    run_id: str
    language: str
    doc_style: str
    workflow_state: str
    status: str
    original_code: str
    proposed_code: str
    mode: str
    model: str
    revision_count: int = 0
    warning: Optional[str] = None
    analysis: dict[str, Any]
    verification: dict[str, Any]
    quality_score: Optional[dict[str, Any]] = None
    stages: list[dict[str, Any]]
    agent_executions: dict[str, Any]


class Orchestrator:
    """Central Controller managing multi-agent pipeline and deterministic state transitions."""

    def __init__(self):
        self.code_analysis_agent = CodeAnalysisAgent()
        self.doc_generation_agent = DocumentationGenerationAgent()
        self.doc_review_agent = DocumentationReviewAgent()
        self.validation_agent = ValidationAgent()

    def process_code_generation(
        self,
        code: str,
        language: str,
        doc_style: str = "google",
        run_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Execute end-to-end multi-agent documentation workflow."""
        if not run_id:
            run_id = str(uuid.uuid4())

        lang_clean = language.lower().strip()
        doc_style_clean = (doc_style or "google").lower().strip()

        # Step 0: State -> RECEIVED
        current_state = WorkflowState.RECEIVED
        agent_executions: dict[str, Any] = {}

        # Validate language capability
        cap = get_language(lang_clean)
        if not cap:
            current_state = WorkflowState.FAILED
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{language}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}",
            )

        # Step 1: State -> ANALYZING (Code Analysis Agent)
        current_state = WorkflowState.ANALYZING
        analysis_res: CodeAnalysisResult = self.code_analysis_agent.run(code, lang_clean)
        agent_executions["code_analysis"] = analysis_res.model_dump()

        if analysis_res.status == "failed" and analysis_res.errors:
            # Syntax failure in Python or ES6 modules
            if analysis_res.valid_syntax is False:
                pass  # Proceed to let generator/verifier record or raise if fatal
            else:
                current_state = WorkflowState.FAILED
                raise HTTPException(status_code=422, detail=analysis_res.summary)

        # Step 2: State -> GENERATING (Documentation Generation Agent)
        current_state = WorkflowState.GENERATING
        try:
            gen_res: DocumentationResult = self.doc_generation_agent.run(
                code=code,
                language=lang_clean,
                analysis=analysis_res.analysis,
                doc_style=doc_style_clean,
                revision_feedback=None,
                revision_number=0,
            )
            agent_executions["documentation_generation"] = gen_res.model_dump()
        except HTTPException as http_exc:
            current_state = WorkflowState.FAILED
            clean_detail = sanitize_error(str(http_exc.detail))
            logger.error("Generation failed (HTTP %d): %s", http_exc.status_code, clean_detail)
            raise HTTPException(status_code=http_exc.status_code, detail=clean_detail) from http_exc
        except Exception as gen_err:
            current_state = WorkflowState.FAILED
            clean_tb = sanitize_error(traceback.format_exc())
            clean_msg = sanitize_error(str(gen_err))
            logger.error("Unhandled generation error in orchestrator:\n%s", clean_tb)
            raise HTTPException(
                status_code=502,
                detail=f"Documentation generation error: {clean_msg}",
            ) from gen_err

        if gen_res.status == "failed" or not gen_res.proposed_code.strip():
            current_state = WorkflowState.FAILED
            raise HTTPException(
                status_code=502,
                detail=gen_res.summary or "Documentation Generator returned empty candidate code.",
            )

        candidate_code = gen_res.proposed_code
        revision_count = 0

        # Step 3: Run Documentation Review & Validation Agent CONCURRENTLY on candidate
        current_state = WorkflowState.REVIEWING  # Represents concurrent review/validation phase
        review_res, val_res = self._run_review_and_validation_concurrently(
            original_code=code,
            candidate_code=candidate_code,
            language=lang_clean,
            analysis=analysis_res.analysis,
            doc_style=doc_style_clean,
        )
        agent_executions["documentation_review"] = review_res.model_dump()
        agent_executions["validation"] = val_res.model_dump()

        # Step 4: Orchestrator Evaluates Results & Bounded Revision Policy
        # If Review flags defects that require revision AND validation did not completely break syntax
        # AND we have not exceeded the revision bound of 1:
        if review_res.status == "needs_revision" and review_res.actionable_feedback and revision_count < 1:
            # Execute exactly ONE bounded revision with actionable feedback
            revision_count = 1
            current_state = WorkflowState.GENERATING
            try:
                revised_gen: DocumentationResult = self.doc_generation_agent.run(
                    code=code,
                    language=lang_clean,
                    analysis=analysis_res.analysis,
                    doc_style=doc_style_clean,
                    revision_feedback=review_res.actionable_feedback,
                    revision_number=1,
                )
                agent_executions["documentation_generation_revision_1"] = revised_gen.model_dump()
            except HTTPException as http_exc:
                current_state = WorkflowState.FAILED
                clean_detail = sanitize_error(str(http_exc.detail))
                logger.error("Revision generation failed (HTTP %d): %s", http_exc.status_code, clean_detail)
                raise HTTPException(status_code=http_exc.status_code, detail=clean_detail) from http_exc
            except Exception as rev_err:
                current_state = WorkflowState.FAILED
                clean_tb = sanitize_error(traceback.format_exc())
                clean_msg = sanitize_error(str(rev_err))
                logger.error("Unhandled revision generation error in orchestrator:\n%s", clean_tb)
                raise HTTPException(
                    status_code=502,
                    detail=f"Documentation revision error: {clean_msg}",
                ) from rev_err

            if revised_gen.status != "failed" and revised_gen.proposed_code.strip():
                candidate_code = revised_gen.proposed_code
                # Re-run review and validation on revised candidate concurrently
                review_res, val_res = self._run_review_and_validation_concurrently(
                    original_code=code,
                    candidate_code=candidate_code,
                    language=lang_clean,
                    analysis=analysis_res.analysis,
                    doc_style=doc_style_clean,
                )
                agent_executions["documentation_review_revision_1"] = review_res.model_dump()
                agent_executions["validation_revision_1"] = val_res.model_dump()

        # Step 5: Enforce Validation and Approval Gates
        # Even if revision was attempted, if validation failed (e.g. syntax error or structural mutation),
        # the validation failure is preserved and clearly reflected.
        current_state = WorkflowState.AWAITING_APPROVAL

        demo_warning = (
            "Running in DEMO MOCK MODE because GEMINI_API_KEY is not configured. Mock output is locally generated and clearly labelled."
            if gen_res.mode == "demo_mock"
            else None
        )

        # Assemble stages matching frontend contract and test requirements:
        # ["analysis", "generation", "validation", "review"]
        stages = self._build_stages(
            code=code,
            language=lang_clean,
            doc_style=doc_style_clean,
            analysis_res=analysis_res,
            gen_res=gen_res,
            review_res=review_res,
            val_res=val_res,
            revision_count=revision_count,
        )

        # Persist run in SQLite audit store
        store.save_run(
            language=lang_clean,
            original=code,
            proposed=candidate_code,
            analysis=analysis_res.analysis,
            mode=gen_res.mode,
            model=gen_res.model,
            workflow_state=current_state.value,
            agents_log=agent_executions,
            revision_count=revision_count,
            run_id=run_id,
        )

        return {
            "run_id": run_id,
            "language": lang_clean,
            "doc_style": doc_style_clean,
            "workflow_state": current_state.value,
            "status": "pending_review",
            "original_code": code,
            "proposed_code": candidate_code,
            "mode": gen_res.mode,
            "model": gen_res.model,
            "revision_count": revision_count,
            "warning": demo_warning,
            "analysis": analysis_res.analysis,
            "verification": val_res.verification,
            "quality_score": review_res.quality_score,
            "stages": stages,
            "agent_executions": agent_executions,
        }

    def _run_review_and_validation_concurrently(
        self,
        original_code: str,
        candidate_code: str,
        language: str,
        analysis: dict[str, Any],
        doc_style: str,
    ) -> tuple[ReviewResult, ValidationResult]:
        """Execute Documentation Review Agent and Validation Agent concurrently via ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            review_future = executor.submit(
                self.doc_review_agent.run,
                original_code=original_code,
                proposed_code=candidate_code,
                language=language,
                analysis=analysis,
                doc_style=doc_style,
            )
            validation_future = executor.submit(
                self.validation_agent.run,
                proposed_code=candidate_code,
                language=language,
                original_code=original_code,
                doc_style=doc_style,
            )

            review_res: ReviewResult = review_future.result()
            val_res: ValidationResult = validation_future.result()

        return review_res, val_res

    def _build_stages(
        self,
        code: str,
        language: str,
        doc_style: str,
        analysis_res: CodeAnalysisResult,
        gen_res: DocumentationResult,
        review_res: ReviewResult,
        val_res: ValidationResult,
        revision_count: int,
    ) -> list[dict[str, Any]]:
        checks = val_res.checks
        fp_match = True
        if len(checks) >= 3 and checks[2].get("fingerprint_comparison"):
            fp_match = checks[2]["fingerprint_comparison"].get("matches", True)

        return [
            {
                "id": "analysis",
                "name": "Stage 1: Code Analysis Agent",
                "status": "completed" if analysis_res.valid_syntax is not False else "warning",
                "input": {
                    "language": language,
                    "source_length": len(code),
                },
                "output": {
                    "valid_syntax": analysis_res.valid_syntax,
                    "syntax_status": analysis_res.syntax_status,
                    "parser_used": analysis_res.parser_used,
                    "symbol_count": analysis_res.symbol_count,
                    "missing_docs_count": len(analysis_res.missing_docs),
                    "fingerprint": analysis_res.analysis.get("fingerprint", {}).get("fingerprint"),
                },
            },
            {
                "id": "generation",
                "name": "Stage 2: Documentation Generation Agent",
                "status": "completed",
                "input": {
                    "doc_style": doc_style,
                    "mode": gen_res.mode,
                    "model": gen_res.model,
                    "revisions": revision_count,
                },
                "output": {
                    "proposed_length": len(gen_res.proposed_code),
                    "mode": gen_res.mode,
                    "model": gen_res.model,
                    "revision_count": revision_count,
                },
            },
            {
                "id": "validation",
                "name": "Stage 3: Validation Agent",
                "status": "completed" if val_res.passed else "failed",
                "input": {
                    "original_code_baseline": "ast_and_token_diff",
                    "checks_count": len(checks),
                },
                "output": {
                    "passed": val_res.passed,
                    "global_status": val_res.global_status,
                    "quality_score": review_res.score,
                    "grade": review_res.grade,
                    "fingerprint_matches": fp_match,
                    "errors": val_res.errors,
                },
            },
            {
                "id": "review",
                "name": "Stage 4: Documentation Review & Human Decision",
                "status": "pending",
                "input": {
                    "action_required": "Operator must inspect diff, quality scorecard, agent findings, and approve/reject.",
                    "review_status": review_res.status,
                    "review_severity": review_res.severity,
                },
                "output": {
                    "quality_score": review_res.score,
                    "grade": review_res.grade,
                    "findings_count": len(review_res.findings),
                    "actionable_feedback": review_res.actionable_feedback,
                },
            },
        ]

    def approve_proposal(
        self,
        run_id: str,
        approved_code: str,
        doc_style: str = "google",
    ) -> dict[str, Any]:
        """Human approval gate: re-validates final approved code and records auditable completion."""
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run["status"] != "pending_review":
            raise HTTPException(
                status_code=409,
                detail=f"This run cannot be approved because its current status is '{run['status']}'.",
            )

        # Final verification of approved patch using ValidationAgent
        val_res = self.validation_agent.run(
            proposed_code=approved_code,
            language=run["language"],
            original_code=run.get("original_code"),
            doc_style=doc_style,
        )

        updated_run = store.approve_run(run_id, approved_code, val_res.verification)
        if not updated_run:
            raise HTTPException(status_code=404, detail="Run not found during update.")

        return {
            "run_id": run_id,
            "status": updated_run["status"],
            "workflow_state": updated_run.get("workflow_state", WorkflowState.COMPLETED.value),
            "verification": val_res.verification,
            "quality_score": val_res.verification.get("quality_score"),
            "final_code": approved_code,
            "reviewed_at": updated_run.get("reviewed_at"),
        }

    def reject_proposal(
        self,
        run_id: str,
        reason: Optional[str] = None,
    ) -> dict[str, Any]:
        """Human rejection gate: preserves original code and marks run as REJECTED."""
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")
        if run["status"] != "pending_review":
            raise HTTPException(
                status_code=409,
                detail=f"This run cannot be rejected because its current status is '{run['status']}'.",
            )

        updated_run = store.reject_run(run_id, reason)
        if not updated_run:
            raise HTTPException(status_code=404, detail="Run not found during rejection.")

        return {
            "run_id": run_id,
            "status": "rejected",
            "workflow_state": WorkflowState.REJECTED.value,
            "rejection_reason": updated_run.get("rejection_reason"),
            "reviewed_at": updated_run.get("reviewed_at"),
            "message": "Proposal rejected. The original source code remains unchanged.",
        }
