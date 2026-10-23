"""DocuPilot AI Backend API.

FastAPI application providing multi-language AST/heuristic analysis,
OpenAI code documentation generation, human review workflow (approve/reject),
modular verification pipeline, and auditable SQLite persistence.
"""

import ast
import io
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from . import store
from .analyzer import analyze
from .languages import REGISTRY, SUPPORTED_LANGUAGES, list_capabilities
from .verifier import verify
from .orchestrator import Orchestrator

logger = logging.getLogger("docupilot.api")

load_dotenv()
app = FastAPI(
    title="DocuPilot AI API",
    version="1.1.0",
    description="Agentic code documentation engine with human-in-the-loop review and multi-stage verification.",
)

# Configure CORS policy for production Vercel frontend, preview domains, and local dev
ALLOWED_ORIGINS = [
    "https://docupilot-ai-three.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
]

# Read optional environment overrides (comma-separated, clean whitespace & trailing slashes)
env_origins = [
    o.strip().rstrip("/")
    for o in os.getenv("FRONTEND_ORIGIN", "").split(",")
    if o.strip()
]
for origin in env_origins:
    if origin not in ALLOWED_ORIGINS:
        ALLOWED_ORIGINS.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https:\/\/([a-zA-Z0-9_-]+\.)?vercel\.app$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store.init_db()
orchestrator = Orchestrator()

SUPPORTED_LANGS_REGEX = "^(" + "|".join(SUPPORTED_LANGUAGES) + ")$"


class GenerateRequest(BaseModel):
    code: str = Field(min_length=3, max_length=50000)
    language: str = Field(pattern=SUPPORTED_LANGS_REGEX)
    doc_style: Optional[str] = Field(default="google")


class ApproveRequest(BaseModel):
    run_id: str
    code: str = Field(min_length=1, max_length=50000)
    doc_style: Optional[str] = Field(default="google")


class RejectRequest(BaseModel):
    run_id: str
    reason: Optional[str] = Field(default=None, max_length=1000)


def get_gemini_key() -> str:
    """Retrieve configured Gemini API key (supports GEMINI_API_KEY or GOOGLE_API_KEY)."""
    return os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()


def get_openai_key() -> str:
    """Retrieve configured OpenAI API key."""
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_active_provider() -> str:
    """Determine the active AI provider: 'gemini', 'openai', or 'demo_mock'."""
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
    """Retrieve the model for the active provider as single source of truth."""
    provider = get_active_provider()
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    return "mock-engine"



COMMON_PARAM_DESCS = {
    "price": "Base unit price or monetary amount before adjustments.",
    "discount": "Percentage discount rate applied to the total price (0-100).",
    "orders": "Collection of order records containing item, status, and total details.",
    "minimum_total": "Minimum value threshold required for an order to qualify.",
    "weight": "Shipment weight used to calculate base freight rates.",
    "distance": "Transit distance used to calculate distance-based surcharges.",
    "express": "Boolean flag indicating whether expedited transit priority applies.",
    "items": "List of items or elements to process.",
    "threshold": "Numerical cut-off boundary for qualifying elements.",
    "name": "Identifier name or string label.",
    "query": "Search query expression or filter criteria.",
    "data": "Input payload or dataset to evaluate.",
    "a": "First operand value.",
    "b": "Second operand value.",
    "x": "Primary input coordinate or numerical argument.",
    "y": "Secondary input coordinate or numerical argument.",
    "n": "Integer quantity or numeric parameter.",
}


def humanize_name(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1 \2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s1)
    words = s2.replace("_", " ").strip().split()
    if not words:
        return "Executes the operation based on provided inputs."
    verb = words[0].lower()
    if verb.endswith("e"):
        verb = verb.capitalize() + "s"
    elif verb.endswith(("ch", "sh", "ss", "x")):
        verb = verb.capitalize() + "es"
    elif verb.endswith("y") and verb[-2:] not in ("ay", "ey", "oy", "uy"):
        verb = verb[:-1].capitalize() + "ies"
    else:
        verb = verb.capitalize() + "s"
    rest = " ".join(words[1:]) if len(words) > 1 else "operation"
    return f"{verb} {rest} according to business rules and input parameters."


def describe_param(param: str) -> str:
    lower = param.lower().strip().lstrip("$")
    if lower in COMMON_PARAM_DESCS:
        return COMMON_PARAM_DESCS[lower]
    clean = param.replace("_", " ").strip().lstrip("$")
    return f"Parameter representing the {clean} value."


def generate_python_documentation(
    code: str, analysis: dict, doc_style: str = "google"
) -> str:
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    style = doc_style.lower().strip() if doc_style else "google"
    lines = code.splitlines()
    inserts: list[tuple[int, str]] = []

    has_module_doc = ast.get_docstring(tree) is not None
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    if not has_module_doc:
        if funcs or classes:
            names = [c.name for c in classes] + [f.name for f in funcs]
            summary = f"Provides core implementations for {', '.join(names[:4])}."
        else:
            has_input = any(
                isinstance(n, ast.Call) and getattr(n.func, "id", "") == "input"
                for n in ast.walk(tree)
            )
            has_print = any(
                isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"
                for n in ast.walk(tree)
            )
            if has_input and has_print:
                summary = "Interactive console script.\nPrompts for user input, computes operations, and outputs results."
            elif has_input:
                summary = "Interactive user input processing script."
            elif has_print:
                summary = "Execution script with formatted console output."
            else:
                summary = "Operational workflow script."
        mod_doc = f'"""\n{summary}\n"""\n'
        inserts.append((0, mod_doc))

    symbol_map: dict[Any, Any] = {}
    for s in analysis.get("symbols", []):
        if s.get("name"):
            symbol_map[(s.get("name"), s.get("line"))] = s
            if s.get("name") not in symbol_map:
                symbol_map[s.get("name")] = s

    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and ast.get_docstring(node) is None
        ):
            if not node.body:
                continue
            first_body = node.body[0]
            target_idx = first_body.lineno - 1
            body_line = lines[target_idx]
            leading = len(body_line) - len(body_line.lstrip(" "))
            indent = " " * (leading if leading > node.col_offset else (node.col_offset + 4))
            inner = indent + "    "
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                summary = humanize_name(node.name)
                sym_info = symbol_map.get((node.name, node.lineno)) or symbol_map.get(node.name) or {}
                param_details = sym_info.get("param_details") or []
                if not param_details:
                    raw_args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
                    param_details = [{"name": a, "type": None} for a in raw_args]

                has_return = sym_info.get("has_return")
                if has_return is None:
                    has_return = any(
                        isinstance(n, ast.Return) and n.value is not None for n in ast.walk(node)
                    )

                return_type = sym_info.get("return_type")
                if not return_type and node.returns:
                    try:
                        return_type = ast.unparse(node.returns)
                    except Exception:
                        return_type = None

                detected_raises = sym_info.get("raises") or []
                if not detected_raises:
                    for n in ast.walk(node):
                        if isinstance(n, ast.Raise) and n.exc:
                            if isinstance(n.exc, ast.Name):
                                detected_raises.append(n.exc.id)
                            elif isinstance(n.exc, ast.Call) and isinstance(n.exc.func, ast.Name):
                                detected_raises.append(n.exc.func.id)

                doc_lines = [
                    f'{indent}"""{summary}',
                    "",
                    f"{indent}Evaluates input arguments, performs calculation/processing, and returns results.",
                ]

                if style == "numpy":
                    if param_details:
                        doc_lines.extend(["", f"{indent}Parameters", f"{indent}----------"])
                        for p in param_details:
                            p_name = p["name"]
                            p_type = p.get("type")
                            if p_type:
                                doc_lines.append(f"{indent}{p_name} : {p_type}")
                            else:
                                doc_lines.append(f"{indent}{p_name}")
                            doc_lines.append(f"{inner}{describe_param(p_name)}")
                    if has_return:
                        doc_lines.extend(["", f"{indent}Returns", f"{indent}-------"])
                        if return_type:
                            doc_lines.append(f"{indent}{return_type}")
                        else:
                            doc_lines.append(f"{indent}object")
                        doc_lines.append(f"{inner}Computed result derived from the processed inputs.")
                    if detected_raises:
                        doc_lines.extend(["", f"{indent}Raises", f"{indent}------"])
                        for r in sorted(set(detected_raises)):
                            doc_lines.append(f"{indent}{r}")
                            doc_lines.append(f"{inner}If input values fail validation checks.")
                else:
                    # Google style
                    if param_details:
                        doc_lines.extend(["", f"{indent}Args:"])
                        for p in param_details:
                            p_name = p["name"]
                            p_type = p.get("type")
                            if p_type:
                                doc_lines.append(f"{inner}{p_name} ({p_type}): {describe_param(p_name)}")
                            else:
                                doc_lines.append(f"{inner}{p_name}: {describe_param(p_name)}")
                    if has_return:
                        doc_lines.extend(["", f"{indent}Returns:"])
                        if return_type:
                            doc_lines.append(f"{inner}{return_type}: Computed result derived from the processed inputs.")
                        else:
                            doc_lines.append(f"{inner}Computed result derived from the processed inputs.")
                    if detected_raises:
                        doc_lines.extend(["", f"{indent}Raises:"])
                        for r in sorted(set(detected_raises)):
                            doc_lines.append(f"{inner}{r}: If input values fail validation checks.")

                doc_lines.append(f'{indent}"""')
                inserts.append((target_idx, "\n".join(doc_lines)))
            else:
                summary = humanize_name(node.name)
                doc = f'{indent}"""{summary}\n\n{indent}Encapsulates state and logic for this component.\n{indent}"""'
                inserts.append((target_idx, doc))

    def get_comment_for_stmt(node: ast.AST) -> str | None:
        if isinstance(node, ast.Assign):
            var_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            target_str = ", ".join(var_names) if var_names else "value"
            val = node.value
            if isinstance(val, ast.Call):
                f_name = getattr(val.func, "id", "")
                if f_name in ("int", "float", "str"):
                    if (
                        val.args
                        and isinstance(val.args[0], ast.Call)
                        and getattr(val.args[0].func, "id", "") == "input"
                    ):
                        return f"# Prompt user and parse {f_name} input for '{target_str}'"
                    return f"# Cast expression to {f_name} and store in '{target_str}'"
                elif f_name == "input":
                    return f"# Prompt user for '{target_str}' string input"
                elif f_name:
                    return f"# Initialize '{target_str}' from {f_name}() result"
            elif isinstance(val, ast.BinOp):
                op_map = {
                    ast.Add: "sum",
                    ast.Sub: "difference",
                    ast.Mult: "product",
                    ast.Div: "quotient",
                    ast.FloorDiv: "floor quotient",
                    ast.Mod: "modulo remainder",
                }
                op_name = op_map.get(type(val.op), "arithmetic result")
                return f"# Calculate {op_name} and assign to '{target_str}'"
            elif isinstance(val, (ast.List, ast.Dict, ast.Set)):
                return f"# Initialize '{target_str}' collection"
            return f"# Initialize '{target_str}'"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            return f"# Declare and initialize typed variable '{node.target.id}'"
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f_name = getattr(node.value.func, "id", "")
            if f_name == "print":
                args = node.value.args
                if (
                    args
                    and isinstance(args[0], ast.BinOp)
                    and isinstance(args[0].op, ast.Add)
                ):
                    return "# Calculate sum and display result to console"
                elif args and isinstance(args[0], ast.BinOp):
                    return "# Evaluate expression and print result to console"
                return "# Output results to console"
            elif f_name:
                return f"# Invoke {f_name}()"
        elif isinstance(node, ast.If):
            return "# Check conditional criteria and route control flow"
        elif isinstance(node, (ast.For, ast.While)):
            return "# Iterate through sequence elements"
        elif isinstance(node, ast.Return):
            return "# Return evaluated result to caller"
        return None

    commented_line_indices = set()
    for idx, l in enumerate(lines):
        if l.strip().startswith("#"):
            commented_line_indices.add(idx)
            commented_line_indices.add(idx + 1)

    candidate_stmts = list(tree.body)
    for f in funcs:
        candidate_stmts.extend(f.body)

    for stmt in candidate_stmts:
        target_idx = stmt.lineno - 1
        if (
            target_idx in commented_line_indices
            or target_idx < 0
            or target_idx >= len(lines)
        ):
            continue
        comment = get_comment_for_stmt(stmt)
        if comment:
            stmt_line = lines[target_idx]
            indent = " " * (len(stmt_line) - len(stmt_line.lstrip(" ")))
            inserts.append((target_idx, indent + comment))
            commented_line_indices.add(target_idx)

    for idx, doc in sorted(
        inserts,
        key=lambda x: (x[0], 0 if x[1].startswith('"""') else 1),
        reverse=True,
    ):
        lines.insert(idx, doc)

    return "\n".join(lines)


def mock_documentation(
    code: str, language: str, analysis: dict, doc_style: str = "google"
) -> str:
    lang = language.lower().strip()
    if lang == "python":
        return generate_python_documentation(code, analysis, doc_style=doc_style)

    symbols = analysis.get("symbols") or []
    lines = code.splitlines()
    inserts = []

    func_symbols = [s for s in symbols if s.get("type") not in ("Variable", "Output", "Query")]

    for sym in func_symbols:
        line_idx = (sym.get("line") or 1) - 1
        if line_idx < 0 or line_idx >= len(lines):
            continue
        line_text = lines[line_idx]
        indent = " " * (len(line_text) - len(line_text.lstrip(" ")))
        name = sym.get("name", "function")
        params = sym.get("parameters", [])
        summary = humanize_name(name)

        doc_lines = []
        if lang in ("javascript", "typescript"):
            doc_lines.append("/**")
            doc_lines.append(f" * {summary}")
            doc_lines.append(" *")
            for p in params:
                doc_lines.append(f" * @param {{*}} {p} - {describe_param(p)}")
            doc_lines.append(" * @returns {*} Computed result based on the processed inputs.")
            doc_lines.append(" */")
        elif lang == "java":
            doc_lines.append("/**")
            doc_lines.append(f" * {summary}")
            doc_lines.append(" *")
            for p in params:
                doc_lines.append(f" * @param {p} {describe_param(p)}")
            doc_lines.append(" * @return Computed result derived from the processed inputs.")
            doc_lines.append(" */")
        elif lang in ("c", "cpp"):
            doc_lines.append("/**")
            doc_lines.append(f" * @brief {summary}")
            doc_lines.append(" *")
            for p in params:
                doc_lines.append(f" * @param {p} {describe_param(p)}")
            doc_lines.append(" * @return Computed result derived from the processed inputs.")
            doc_lines.append(" */")
        elif lang == "csharp":
            doc_lines.append("/// <summary>")
            doc_lines.append(f"/// {summary}")
            doc_lines.append("/// </summary>")
            for p in params:
                doc_lines.append(f'/// <param name="{p}">{describe_param(p)}</param>')
            doc_lines.append("/// <returns>Computed result derived from the processed inputs.</returns>")
        elif lang == "go":
            doc_lines.append(f"// {name} {summary.lower()}")
            for p in params:
                doc_lines.append(f"// {p}: {describe_param(p)}")
            doc_lines.append("// Returns the computed outcome based on input conditions.")
        elif lang == "php":
            doc_lines.append("/**")
            doc_lines.append(f" * {summary}")
            doc_lines.append(" *")
            for p in params:
                doc_lines.append(f" * @param mixed ${p} {describe_param(p)}")
            doc_lines.append(" * @return mixed Computed result derived from the processed inputs.")
            doc_lines.append(" */")
        elif lang == "ruby":
            doc_lines.append(f"# {summary}")
            doc_lines.append("#")
            for p in params:
                doc_lines.append(f"# @param {p} [Object] {describe_param(p)}")
            doc_lines.append("# @return [Object] Computed result derived from the processed inputs.")
        elif lang == "swift":
            doc_lines.append(f"/// {summary}")
            if params:
                doc_lines.append("///")
                doc_lines.append("/// - Parameters:")
                for p in params:
                    doc_lines.append(f"///   - {p}: {describe_param(p)}")
            doc_lines.append("/// - Returns: Computed result derived from the processed inputs.")
        elif lang == "kotlin":
            doc_lines.append("/**")
            doc_lines.append(f" * {summary}")
            doc_lines.append(" *")
            for p in params:
                doc_lines.append(f" * @param {p} {describe_param(p)}")
            doc_lines.append(" * @return Computed result derived from the processed inputs.")
            doc_lines.append(" */")
        elif lang == "rust":
            doc_lines.append(f"/// {summary}")
            if params:
                doc_lines.append("///")
                doc_lines.append("/// # Arguments")
                doc_lines.append("///")
                for p in params:
                    doc_lines.append(f"/// * `{p}` - {describe_param(p)}")
            doc_lines.append("///")
            doc_lines.append("/// # Returns")
            doc_lines.append("///")
            doc_lines.append("/// * Computed result derived from the processed inputs.")
        elif lang == "sql":
            doc_lines.append("-- ============================================================================")
            doc_lines.append(f"-- Definition: {name}")
            doc_lines.append(f"-- Purpose: {summary}")
            doc_lines.append("-- ============================================================================")

        if doc_lines:
            indented = "\n".join(indent + l for l in doc_lines)
            inserts.append((line_idx, indented))

    if not func_symbols:
        header = ""
        c_pfx = "// "
        if lang in ("javascript", "typescript"):
            header = "/**\n * Application script and workflow execution.\n * Handles user inputs, data operations, and output results.\n */\n"
        elif lang == "java":
            header = "/**\n * Application routine for data operations and computational workflow.\n */\n"
        elif lang in ("c", "cpp"):
            header = "/**\n * @file\n * @brief Application routine and operational logic execution.\n */\n"
        elif lang == "csharp":
            header = "/// <summary>\n/// Application script executing operational logic and workflow.\n/// </summary>\n"
        elif lang == "go":
            header = "// Package main provides operational script routines and data workflows.\n"
        elif lang == "php":
            header = "/**\n * Script workflow execution and operations.\n */\n"
        elif lang == "ruby":
            c_pfx = "# "
            header = "# Module providing execution logic and data operations.\n"
        elif lang == "swift":
            header = "/// Application routine executing data processing and workflows.\n"
        elif lang == "kotlin":
            header = "/**\n * Application script executing routine logic.\n */\n"
        elif lang == "rust":
            header = "//! Application module executing data operations and workflow.\n"
        elif lang == "sql":
            c_pfx = "-- "
            header = "-- ============================================================================\n-- Script: Database Query and Data Manipulation\n-- ============================================================================\n"

        if header:
            inserts.append((0, header))

        for idx, line in enumerate(lines):
            l_strip = line.strip()
            if not l_strip or l_strip.startswith(("//", "#", "/*", "*", "--")):
                continue
            indent = " " * (len(line) - len(line.lstrip(" ")))
            if re.search(r"(?:prompt|scanf|cin\s*>>|readline|Console\.ReadLine)\s*\(?", l_strip, re.I):
                inserts.append((idx, f"{indent}{c_pfx}Read user input from standard stream"))
            elif re.search(r"(?:console\.log|printf|cout\s*<<|println|print|echo|System\.out\.println)\s*\(?", l_strip, re.I):
                inserts.append((idx, f"{indent}{c_pfx}Output evaluated result to console"))
            elif re.search(r"^\s*(?:let|const|var|val|int|float|double|char|string|\$)\s+([A-Za-z0-9_]+)", l_strip):
                inserts.append((idx, f"{indent}{c_pfx}Declare and initialize variable"))
            elif re.search(r"^\s*if\s*\(", l_strip):
                inserts.append((idx, f"{indent}{c_pfx}Evaluate condition and route control flow"))

    if inserts:
        for idx, doc in sorted(
            inserts,
            key=lambda x: (0 if ("/**" in x[1] or "/*" in x[1] or "-- ===" in x[1]) else 1),
            reverse=True,
        ):
            lines.insert(idx, doc)
        return "\n".join(lines)

    comment_prefix = "-- " if lang == "sql" else ("# " if lang == "ruby" else "// ")
    return f"{comment_prefix}Module documentation for {lang}\n" + code


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


def generate_with_gemini(
    code: str,
    language: str,
    analysis: dict,
    doc_style: str = "google",
    revision_feedback: Optional[str] = None,
) -> tuple[str, str, str]:
    """Generate documentation using Google Gemini API.

    Returns:
        tuple of (proposed_code, mode, model_name)
    Raises:
        HTTPException if Gemini API call fails (never silently switches to mock).
    """
    key = get_gemini_key()
    if not key:
        return (
            mock_documentation(code, language, analysis, doc_style=doc_style),
            "demo_mock",
            "mock-engine",
        )

    model = os.getenv("GEMINI_MODEL", "").strip() or "gemini-3.8-flash"

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
    code: str,
    language: str,
    analysis: dict,
    doc_style: str = "google",
    revision_feedback: Optional[str] = None,
) -> tuple[str, str, str]:
    """Generate documentation using OpenAI API.

    Returns:
        tuple of (proposed_code, mode, model_name)
    Raises:
        HTTPException if OpenAI API call fails (never silently switches to mock).
    """
    key = get_openai_key()
    if not key:
        return (
            mock_documentation(code, language, analysis, doc_style=doc_style),
            "demo_mock",
            "mock-engine",
        )

    model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
    client = OpenAI(api_key=key, timeout=30.0)
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


def generate_documentation(
    code: str,
    language: str,
    analysis: dict,
    doc_style: str = "google",
    revision_feedback: Optional[str] = None,
) -> tuple[str, str, str]:
    """Route code generation to the active AI provider (Gemini, OpenAI, or Demo Mock)."""
    provider = get_active_provider()
    if provider == "gemini":
        return generate_with_gemini(code, language, analysis, doc_style=doc_style, revision_feedback=revision_feedback)
    elif provider == "openai":
        return generate_with_openai(code, language, analysis, doc_style=doc_style, revision_feedback=revision_feedback)
    return (
        mock_documentation(code, language, analysis, doc_style=doc_style),
        "demo_mock",
        "mock-engine",
    )


@app.get("/")
def root():
    provider = get_active_provider()
    return {
        "name": "DocuPilot AI API",
        "version": "1.1.0",
        "status": "ok",
        "docs": "/docs",
        "provider": provider,
        "model": get_configured_model(),
        "has_gemini_key": bool(get_gemini_key()),
        "has_openai_key": bool(get_openai_key()),
    }


@app.get("/api/health")
def health():
    try:
        with store.connect() as con:
            con.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "error",
    }


@app.get("/api/languages")
def languages():
    return {"languages": list_capabilities()}


@app.get("/api/config")
def config():
    provider = get_active_provider()
    has_key = provider != "demo_mock"
    return {
        "provider": provider,
        "model": get_configured_model(),
        "mode": "live" if has_key else "demo_mock",
        "demo_mode": not has_key,
        "has_gemini_key": bool(get_gemini_key()),
        "has_openai_key": bool(get_openai_key()),
        "languages": list_capabilities(),
        "multi_agent": {
            "architecture": "hybrid_multi_agent",
            "orchestrator": "Central Orchestrator (workflow & state machine)",
            "agents": [
                {"id": "code_analysis", "name": "Code Analysis Agent", "role": "AST & structural analysis"},
                {"id": "documentation_generation", "name": "Documentation Generation Agent", "role": "Context-aware generation with revision support"},
                {"id": "documentation_review", "name": "Documentation Review Agent", "role": "Independent groundedness & quality audit"},
                {"id": "validation", "name": "Validation Agent", "role": "Syntax, AST structural consistency & fingerprint checks"},
            ],
        },
    }


@app.get("/api/stats")
def stats():
    return store.get_stats()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    return orchestrator.process_code_generation(
        code=req.code,
        language=req.language,
        doc_style=req.doc_style,
    )


@app.post("/api/approve")
def approve(req: ApproveRequest):
    return orchestrator.approve_proposal(
        run_id=req.run_id,
        approved_code=req.code,
        doc_style=req.doc_style or "google",
    )


@app.post("/api/reject")
def reject(req: RejectRequest):
    return orchestrator.reject_proposal(
        run_id=req.run_id,
        reason=req.reason,
    )


@app.get("/api/history")
def history(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    return {"items": store.list_runs(limit=limit, offset=offset)}


@app.get("/api/history/{run_id}")
def history_item(run_id: str):
    item = store.get_run(run_id)
    if not item:
        raise HTTPException(status_code=404, detail="Run not found.")
    return item


ALLOWED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sql": "sql",
}

MAX_UPLOAD_SIZE = 2 * 1024 * 1024  # 2MB

FORBIDDEN_DIR_SEGMENTS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    ".aws", ".ssh", "dist", "build", ".next", ".cache"
}

FORBIDDEN_NAME_PATTERNS = [
    r"^\.env(\..+)?$",
    r".*secret.*",
    r".*token.*",
    r".*credential.*",
    r".*id_rsa.*",
]


@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Safely upload and inspect a single code file or ZIP repository archive.

    Security guarantees:
    - Maximum upload size strictly capped at 2MB.
    - Path traversal guard (prevents '..' and absolute paths).
    - Excludes secrets, credentials, and dependency trees (node_modules, .env, .git, venv).
    - Prohibits binary/executable extraction.
    - Uploaded source code is parsed statically and NEVER executed.
    """
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}")

    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size limit of 2MB (received {len(content)} bytes).",
        )

    fname = (file.filename or "uploaded_file").strip()
    ext = os.path.splitext(fname)[1].lower()

    if ext == ".zip":
        try:
            zip_buffer = io.BytesIO(content)
            with zipfile.ZipFile(zip_buffer, "r") as zf:
                valid_files = []
                for info in zf.infolist():
                    if info.is_dir():
                        continue

                    clean_path = info.filename.replace("\\", "/")
                    parts = [p for p in clean_path.split("/") if p]
                    if any(p == ".." for p in parts) or clean_path.startswith(("/", "\\")):
                        continue

                    if any(seg in FORBIDDEN_DIR_SEGMENTS for seg in parts):
                        continue

                    base_name = parts[-1].lower()
                    if any(re.match(pat, base_name, re.I) for pat in FORBIDDEN_NAME_PATTERNS):
                        continue

                    sub_ext = os.path.splitext(base_name)[1]
                    if sub_ext not in ALLOWED_EXTENSIONS:
                        continue

                    if info.file_size > 500 * 1024:
                        continue

                    try:
                        file_bytes = zf.read(info)
                        text = file_bytes.decode("utf-8", errors="replace")
                        lang = ALLOWED_EXTENSIONS[sub_ext]
                        valid_files.append({
                            "path": clean_path,
                            "filename": parts[-1],
                            "language": lang,
                            "code": text,
                            "size": len(file_bytes),
                        })
                    except Exception:
                        continue

                if not valid_files:
                    raise HTTPException(
                        status_code=400,
                        detail="No valid source code files found in archive. Executables, dependencies, and sensitive files were safely excluded.",
                    )

                entry_names = {"main.py", "app.py", "index.js", "index.ts", "app.ts"}
                primary = None
                for vf in valid_files:
                    if vf["filename"].lower() in entry_names:
                        primary = vf
                        break
                if not primary:
                    primary = max(valid_files, key=lambda f: f["size"])

                return {
                    "mode": "archive",
                    "filename": fname,
                    "language": primary["language"],
                    "code": primary["code"],
                    "primary_file": primary["path"],
                    "total_files": len(valid_files),
                    "files": [
                        {"path": f["path"], "language": f["language"], "size": f["size"]}
                        for f in valid_files[:20]
                    ],
                }
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP archive file.")

    elif ext in ALLOWED_EXTENSIONS:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"File encoding error: {e}")

        lang = ALLOWED_EXTENSIONS[ext]
        return {
            "mode": "single_file",
            "filename": fname,
            "language": lang,
            "code": text,
            "primary_file": fname,
            "total_files": 1,
            "files": [{"path": fname, "language": lang, "size": len(content)}],
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: .zip or source files ({', '.join(sorted(set(ALLOWED_EXTENSIONS.values())))}).",
        )
