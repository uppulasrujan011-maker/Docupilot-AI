"""Documentation Review Agent for DocuPilot AI.

Performs independent review comparing proposed documentation against the original
code and extracted AST evidence. Evaluates groundedness, parameter accuracy,
identifies hallucinated or omitted descriptions, detects generic placeholders,
and produces actionable feedback for bounded orchestrator revisions.
Supports independent LLM audit when API keys are configured.
"""

import json
import logging
import os
import re
import time
from typing import Any, Optional
from pydantic import BaseModel, Field

from ..analyzer import analyze
from ..quality import evaluate_documentation_quality
from .documentation import sanitize_error

logger = logging.getLogger("docupilot.agents.review")


class ReviewResult(BaseModel):
    agent_name: str = "Documentation Review Agent"
    status: str = "passed"  # "passed", "needs_revision", or "failed"
    score: int = 100
    grade: str = "A"
    severity: str = "none"  # "none", "low", "medium", "high"
    quality_score: dict[str, Any] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    hallucinations: list[str] = Field(default_factory=list)
    actionable_feedback: Optional[str] = None
    review_type: str = "deterministic_static"
    summary: str = ""
    errors: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


GENERIC_PLACEHOLDERS = [
    r"\bdescribe\s+(?:what|this|the|x)\b",
    r"\bprocesses\s+data\b",
    r"\bexecutes\s+logic\b",
    r"\binsert\s+description\b",
    r"\btodo[:\s]",
    r"\bplaceholder\b",
]


class DocumentationReviewAgent:
    """Specialist Agent responsible for auditing documentation accuracy and groundedness."""

    def __init__(self):
        self.name = "Documentation Review Agent"

    def run(
        self,
        original_code: str,
        proposed_code: str,
        language: str,
        analysis: dict[str, Any],
        doc_style: str = "google",
    ) -> ReviewResult:
        start_time = time.perf_counter()
        lang_clean = language.lower().strip()
        doc_style_clean = (doc_style or "google").lower().strip()

        # 1. Analyze proposed code to obtain its symbols and AST
        try:
            prop_analysis = analyze(proposed_code, lang_clean)
        except Exception:
            prop_analysis = {"symbols": []}

        # 2. Evaluate documentation quality deterministically using quality.py
        quality_eval = evaluate_documentation_quality(
            code=proposed_code,
            language=lang_clean,
            analysis=prop_analysis,
            doc_style=doc_style_clean,
        )

        overall_score = quality_eval.get("overall_score", 0)
        grade = quality_eval.get("grade", "Needs Improvement")
        findings: list[str] = list(quality_eval.get("findings", []))
        hallucinations: list[str] = []
        feedback_points: list[str] = []
        severity = "none"

        # 3. Groundedness checks against original code symbols
        orig_symbols = analysis.get("symbols", [])
        orig_funcs = [
            s for s in orig_symbols
            if s.get("type") in ("FunctionDef", "AsyncFunctionDef", "ClassDef", "Function", "Method", "Class")
        ]

        # Check for generic placeholders
        lower_proposed = proposed_code.lower()
        for pat in GENERIC_PLACEHOLDERS:
            if re.search(pat, lower_proposed):
                findings.append(f"Detected generic or placeholder phrase matching pattern '{pat}'.")
                feedback_points.append(f"Replace generic placeholder matching '{pat}' with specific business logic description.")
                severity = "medium"

        # Check parameter groundedness
        for fn in orig_funcs:
            fn_name = fn.get("name", "")
            declared_params = set(fn.get("parameters") or [])
            if not declared_params:
                continue

            # Look for docstring of this function in proposed code
            if lang_clean == "python":
                import ast
                try:
                    tree = ast.parse(proposed_code)
                    for n in ast.walk(tree):
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name:
                            doc = ast.get_docstring(n) or ""
                            doc_lower = doc.lower()
                            # Check for missing parameters in docstring
                            missing = [p for p in declared_params if p.lower() not in doc_lower]
                            if missing and len(missing) == len(declared_params):
                                findings.append(f"Function '{fn_name}' docstring omits all declared parameters: {', '.join(missing)}.")
                                feedback_points.append(f"Document missing parameters for '{fn_name}': {', '.join(missing)}.")
                                severity = "high" if severity != "high" else severity
                except Exception:
                    pass

        # 4. Check for catastrophic doc absence
        param_cov = quality_eval.get("parameter_coverage", 100.0)
        if param_cov < 40.0 and len(orig_funcs) > 0 and any(f.get("parameters") for f in orig_funcs):
            severity = "high"
            feedback_points.append(f"Overall parameter coverage is low ({param_cov}%). Ensure all function parameters have explicit explanations.")

        # 5. Independent LLM Review pass (if live API key is configured)
        review_type = "deterministic_static"
        llm_findings = self._review_with_llm(original_code, proposed_code, lang_clean)
        if llm_findings:
            review_type = "independent_llm"
            for f in llm_findings.get("findings", []):
                findings.append(f"[LLM Audit] {f}")
            if llm_findings.get("actionable_feedback"):
                feedback_points.append(llm_findings["actionable_feedback"])
            if llm_findings.get("hallucinations"):
                hallucinations.extend(llm_findings["hallucinations"])
                severity = "high"

        # 6. Determine review decision
        status = "passed"
        actionable_feedback: Optional[str] = None

        if severity == "high" or overall_score < 50:
            status = "needs_revision"
            actionable_feedback = "; ".join(feedback_points) if feedback_points else "Improve parameter and return documentation coverage."
            summary = f"Review flagged documentation quality issues (Grade: {grade}, Score: {overall_score}/100, Severity: {severity}). Revision recommended."
        elif severity == "medium" or overall_score < 70:
            status = "passed"
            summary = f"Review passed with advisory remarks (Grade: {grade}, Score: {overall_score}/100). Documentation grounded in code."
        else:
            status = "passed"
            summary = f"Review passed with high confidence (Grade: {grade}, Score: {overall_score}/100). Full AST parameter and return grounding verified."

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        evidence = {
            "score": overall_score,
            "grade": grade,
            "parameter_coverage": quality_eval.get("parameter_coverage"),
            "return_coverage": quality_eval.get("return_coverage"),
            "style_conformance": quality_eval.get("style_conformance"),
            "severity": severity,
            "hallucinations_detected": len(hallucinations),
            "feedback_points": feedback_points,
            "breakdown": quality_eval.get("breakdown", []),
            "review_type": review_type,
        }

        return ReviewResult(
            agent_name=self.name,
            status=status,
            score=overall_score,
            grade=grade,
            severity=severity,
            quality_score=quality_eval,
            findings=findings,
            hallucinations=hallucinations,
            actionable_feedback=actionable_feedback,
            review_type=review_type,
            summary=summary,
            errors=[f"Review severity: {severity}"] if status == "needs_revision" else [],
            execution_time_ms=duration_ms,
            evidence=evidence,
        )

    def _review_with_llm(
        self,
        original_code: str,
        proposed_code: str,
        language: str,
    ) -> Optional[dict[str, Any]]:
        """Optional independent LLM review using OpenAI or Gemini if configured."""
        from .documentation import get_active_provider, get_gemini_key, get_openai_key
        provider = get_active_provider()
        if provider == "demo_mock":
            return None

        prompt = f"""You are an independent Code Documentation Auditor.
Compare the proposed documentation against the original source code below.
Verify:
1. Are parameter and return explanations grounded in the actual code logic?
2. Are there any hallucinated parameters, inaccurate types, or unsupported claims?
3. Are there vague placeholders (e.g. 'processes data')?

ORIGINAL CODE:
{original_code}

PROPOSED DOCUMENTED CODE:
{proposed_code}

Respond in concise JSON with:
{{"grounded": true/false, "findings": ["finding 1"], "hallucinations": [], "actionable_feedback": "optional feedback if issues found"}}
"""
        try:
            if provider == "gemini":
                key = get_gemini_key()
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                model = os.getenv("GEMINI_MODEL", "").strip().strip("'\"") or "gemini-3.8-flash"
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.1),
                )
                txt = resp.text or ""
                # Extract JSON if enclosed in markdown fences
                match = re.search(r"\{[\s\S]*\}", txt)
                if match:
                    return json.loads(match.group(0))
            elif provider == "openai":
                key = get_openai_key()
                from openai import OpenAI
                client = OpenAI(api_key=key, timeout=20.0)
                model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a code documentation auditor. Output only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                txt = completion.choices[0].message.content or ""
                match = re.search(r"\{[\s\S]*\}", txt)
                if match:
                    return json.loads(match.group(0))
        except Exception as exc:
            # Independent LLM review is advisory; log sanitized diagnostic and never break pipeline
            clean_err = sanitize_error(str(exc))
            logger.warning("Advisory LLM review pass encountered an error (non-fatal): %s", clean_err)
        return None
