"""Documentation Generation Agent for DocuPilot AI.

Specialist agent that invokes LLMs (Google Gemini, OpenAI) or the deterministic
mock engine to synthesize deep, explanatory docstrings and inline comments.
Respects structural AST findings from the Code Analysis Agent, supports target
documentation conventions, and incorporates actionable revision feedback.
"""

import os
import re
import time
from typing import Any, Optional
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Support both direct OpenAI and mocked imports from main
try:
    from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
except ImportError:
    OpenAI = None


class DocumentationResult(BaseModel):
    agent_name: str = "Documentation Generation Agent"
    status: str = "completed"
    proposed_code: str
    mode: str
    model: str
    doc_style: str
    revision_number: int = 0
    summary: str = ""
    errors: list[str] = Field(default_factory=list)
    execution_time_ms: float = 0.0
    evidence: dict[str, Any] = Field(default_factory=dict)


def get_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()


def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_active_provider() -> str:
    explicit_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit_provider == "gemini" and get_gemini_key():
        return "gemini"
    if explicit_provider == "openai" and get_openai_key():
        return "openai"

    if get_gemini_key():
        return "gemini"
    if get_openai_key():
        return "openai"
    return "demo_mock"


def get_configured_model() -> str:
    provider = get_active_provider()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    return "mock-engine"


def strip_code_fences(output: str) -> str:
    """Safely strip markdown code blocks from LLM output."""
    output = output.strip()
    if output.startswith("```"):
        lines = output.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        output = "\n".join(lines).strip()
    return output


def build_generation_prompt(
    code: str,
    language: str,
    analysis: dict,
    doc_style: str = "google",
    revision_feedback: Optional[str] = None,
) -> str:
    """Build standardized, high-quality prompt for code documentation generation."""
    style_clean = (doc_style or "google").lower().strip()
    if language.lower() == "python":
        if style_clean == "numpy":
            style_instruction = "- Python Style: NUMPY DOCSTRING STANDARD. Use 'Parameters\\n----------', 'Returns\\n-------', 'Raises\\n------' headers with underlines."
        else:
            style_instruction = "- Python Style: GOOGLE DOCSTRING STANDARD. Use PEP 257 triple quotes with 'Args:', 'Returns:', 'Raises:' sections."
    else:
        style_instruction = f"- Target Style: {style_clean.upper()} documentation standard for {language}."

    revision_clause = ""
    if revision_feedback:
        revision_clause = f"""
CRITICAL REVISION DIRECTIVE (REVIEW AGENT FEEDBACK):
The previous candidate received the following feedback from the Documentation Review Agent:
"{revision_feedback}"
You MUST directly address and rectify every issue listed above in your updated documentation.
"""

    return f"""You are an expert senior software engineer and technical documentation specialist.
Your task: Return ONLY the complete updated source code with deep, crystal-clear, explanatory docstrings and insightful inline comments.

STRICT INSTRUCTIONS FOR CLARITY AND QUALITY (NO VAGUE ANSWERS):
1. ZERO VAGUE PLACEHOLDERS:
   - Absolutely NEVER use generic phrases such as "Describe what this function does", "Processes data", "Executes logic", or "Describe x".
   - Every single docstring and comment must specifically describe the ACTUAL business logic, mathematical formulas, constraints, and operations present in the code.

2. LANGUAGE-SPECIFIC DOCUMENTATION CONVENTIONS:
   {style_instruction}
   - Java: Standard Javadoc /** ... */ with descriptive summary, @param, @return, and @throws tags.
   - C#: XML documentation comments using /// <summary> ... </summary>, /// <param name="..."> ... </param>, and /// <returns> ... </returns>.
   - JavaScript & TypeScript: Comprehensive JSDoc/TSDoc /** ... */ with @param {{type}}, @returns {{type}}, and @throws {{type}}.
   - C & C++: Doxygen style /** ... @brief ... @param ... @return ... */ or clear /* ... */ block comments.
   - Go: Idiomatic Go doc format (single-line comments `// FunctionName ...` immediately preceding declarations).
   - PHP: PHPDoc /** ... */ with @param type $name, @return type, and @throws type.
   - Ruby: YARD / RDoc conventions using # comments with @param and @return tags.
   - Swift: Swift documentation markup using /// with "- Parameters:" and "- Returns:" sections.
   - Kotlin: KDoc /** ... */ with @param and @return tags.
   - Rust: Rustdoc /// with markdown headings "# Arguments", "# Returns", and "# Errors".
   - SQL: Professional SQL comments using -- or /* ... */ documenting query/procedure purpose, tables, CTEs, joins, and parameters.

3. INSIGHTFUL INLINE COMMENTS:
   - Add concise inline comments above critical non-obvious statements (e.g. edge-case guard checks, sorting criteria, mathematical calculations, filtering loops, user inputs, outputs).
   - Explain the "WHY" behind the implementation, not merely restating the syntax.

4. PRESERVE ORIGINAL CODE:
   - Preserve 100% of executable statements, variable names, functions, and semantics. Do not refactor or delete any existing logic.
   - Return ONLY the updated code without markdown fences (no ```python or ```javascript) and without any conversational introduction or conclusion.
{revision_clause}
Language: {language}
Target Style: {style_clean}
Parsed structural hints: {analysis}

SOURCE CODE:
{code}"""


class DocumentationGenerationAgent:
    """Specialist Agent responsible for generating high-quality documentation using LLMs."""

    def __init__(self):
        self.name = "Documentation Generation Agent"

    def run(
        self,
        code: str,
        language: str,
        analysis: dict[str, Any],
        doc_style: str = "google",
        revision_feedback: Optional[str] = None,
        revision_number: int = 0,
    ) -> DocumentationResult:
        start_time = time.perf_counter()
        doc_style_clean = (doc_style or "google").lower().strip()
        provider = get_active_provider()

        proposed = ""
        mode = provider
        model_name = get_configured_model()

        if provider == "gemini":
            proposed, mode, model_name = self.generate_with_gemini(
                code=code,
                language=language,
                analysis=analysis,
                doc_style=doc_style_clean,
                revision_feedback=revision_feedback,
            )
        elif provider == "openai":
            proposed, mode, model_name = self.generate_with_openai(
                code=code,
                language=language,
                analysis=analysis,
                doc_style=doc_style_clean,
                revision_feedback=revision_feedback,
            )
        else:
            from ..main import mock_documentation
            proposed = mock_documentation(code, language, analysis, doc_style=doc_style_clean)
            mode = "demo_mock"
            model_name = "mock-engine"

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        if not proposed.strip():
            return DocumentationResult(
                agent_name=self.name,
                status="failed",
                proposed_code="",
                mode=mode,
                model=model_name,
                doc_style=doc_style_clean,
                revision_number=revision_number,
                summary="Generator returned empty code.",
                errors=["The generator returned empty code."],
                execution_time_ms=duration_ms,
            )

        summary = f"Generated {doc_style_clean} documentation via {mode} ({model_name})."
        if revision_number > 0:
            summary += f" (Revision #{revision_number} incorporating Review Agent feedback)."

        evidence = {
            "mode": mode,
            "model": model_name,
            "doc_style": doc_style_clean,
            "revision_number": revision_number,
            "had_feedback": bool(revision_feedback),
            "original_length": len(code),
            "proposed_length": len(proposed),
        }

        return DocumentationResult(
            agent_name=self.name,
            status="completed",
            proposed_code=proposed,
            mode=mode,
            model=model_name,
            doc_style=doc_style_clean,
            revision_number=revision_number,
            summary=summary,
            execution_time_ms=duration_ms,
            evidence=evidence,
        )

    def generate_with_gemini(
        self,
        code: str,
        language: str,
        analysis: dict,
        doc_style: str = "google",
        revision_feedback: Optional[str] = None,
    ) -> tuple[str, str, str]:
        key = get_gemini_key()
        if not key:
            from ..main import mock_documentation
            return (
                mock_documentation(code, language, analysis, doc_style=doc_style),
                "demo_mock",
                "mock-engine",
            )

        model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.5-flash"

        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as imp_err:
            raise HTTPException(
                status_code=500,
                detail=f"Google GenAI SDK is not installed: {imp_err}",
            ) from imp_err

        prompt = build_generation_prompt(code, language, analysis, doc_style, revision_feedback)

        output = ""
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert code documentation assistant. Return only updated code with docstrings and comments without markdown backticks.",
                    temperature=0.2,
                ),
            )
            output = response.text or ""
        except errors.APIError as api_err:
            err_msg = str(api_err)
            status_code = getattr(api_err, "code", None)
            if status_code in (400, 401, 403) or "API_KEY_INVALID" in err_msg or "PERMISSION_DENIED" in err_msg:
                raise HTTPException(
                    status_code=401,
                    detail=f"Google Gemini authentication failed. Invalid API key provided in configuration ({err_msg}).",
                ) from api_err
            elif status_code == 429 or "RESOURCE_EXHAUSTED" in err_msg:
                raise HTTPException(
                    status_code=429,
                    detail=f"Google Gemini rate limit or quota exceeded ({err_msg}). Please try again later.",
                ) from api_err
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Google Gemini API error: {err_msg}",
                ) from api_err
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Google Gemini generation error: {exc}",
            ) from exc

        return strip_code_fences(output), "gemini", model

    def generate_with_openai(
        self,
        code: str,
        language: str,
        analysis: dict,
        doc_style: str = "google",
        revision_feedback: Optional[str] = None,
    ) -> tuple[str, str, str]:
        key = get_openai_key()
        if not key:
            from ..main import mock_documentation
            return (
                mock_documentation(code, language, analysis, doc_style=doc_style),
                "demo_mock",
                "mock-engine",
            )

        # Check if app.main has a mocked OpenAI class for testing
        client_cls = OpenAI
        try:
            from .. import main as main_module
            if hasattr(main_module, "OpenAI"):
                client_cls = main_module.OpenAI
        except Exception:
            pass

        model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
        client = client_cls(api_key=key, timeout=30.0)
        prompt = build_generation_prompt(code, language, analysis, doc_style, revision_feedback)

        output = ""
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert code documentation assistant. Return only updated code with docstrings and comments without markdown backticks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            if completion.choices and completion.choices[0].message:
                output = completion.choices[0].message.content or ""
        except AuthenticationError as auth_err:
            raise HTTPException(
                status_code=401,
                detail="OpenAI authentication failed. Invalid API key provided in configuration.",
            ) from auth_err
        except RateLimitError as rate_err:
            raise HTTPException(
                status_code=429,
                detail="OpenAI rate limit or credit quota exceeded. Please check your OpenAI billing plan.",
            ) from rate_err
        except APIConnectionError as conn_err:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to OpenAI service: {conn_err}",
            ) from conn_err
        except APIError as api_err:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI API error: {api_err.message}",
            ) from api_err
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"OpenAI generation error: {exc}",
            ) from exc

        return strip_code_fences(output), "openai", model
