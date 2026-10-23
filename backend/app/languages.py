"""Central language capability registry for DocuPilot AI.

Defines the exact capabilities available for each supported language,
differentiating generation support from syntax validation and linting.
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageCapability:
    id: str
    name: str
    extension: str
    parser_type: str  # 'ast', 'esprima', 'typescript_compiler', 'regex_heuristic'
    has_real_parser: bool
    syntax_validation_capable: bool
    doc_format: str
    linter_available: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


REGISTRY: dict[str, LanguageCapability] = {
    "python": LanguageCapability(
        id="python",
        name="Python",
        extension="py",
        parser_type="ast",
        has_real_parser=True,
        syntax_validation_capable=True,
        doc_format='PEP 257 / Google docstrings (""")',
        linter_available="ast",
        notes="Full AST parser and syntax verification via Python ast module.",
    ),
    "javascript": LanguageCapability(
        id="javascript",
        name="JavaScript",
        extension="js",
        parser_type="esprima",
        has_real_parser=True,
        syntax_validation_capable=True,
        doc_format="JSDoc (/** ... */)",
        linter_available="esprima",
        notes="Full ECMAScript parser and syntax verification via Esprima.",
    ),
    "typescript": LanguageCapability(
        id="typescript",
        name="TypeScript",
        extension="ts",
        parser_type="typescript_compiler",
        has_real_parser=True,
        syntax_validation_capable=True,
        doc_format="TSDoc / JSDoc (/** ... */)",
        linter_available="tsc",
        notes="Full TypeScript compiler AST parser and syntax diagnostics.",
    ),
    "java": LanguageCapability(
        id="java",
        name="Java",
        extension="java",
        parser_type="regex_heuristic",
        has_real_parser=False,
        syntax_validation_capable=False,
        doc_format="Javadoc (/** ... */)",
        linter_available=None,
        notes="AI generation supported. Syntax verification not available in local environment.",
    ),
    "c": LanguageCapability(
        id="c",
        name="C",
        extension="c",
        parser_type="regex_heuristic",
        has_real_parser=False,
        syntax_validation_capable=False,
        doc_format="Doxygen (/** ... */ / /* ... */)",
        linter_available=None,
        notes="AI generation supported. Syntax verification not available in local environment.",
    ),
    "cpp": LanguageCapability(
        id="cpp",
        name="C++",
        extension="cpp",
        parser_type="regex_heuristic",
        has_real_parser=False,
        syntax_validation_capable=False,
        doc_format="Doxygen (/** ... */ / /* ... */)",
        linter_available=None,
        notes="AI generation supported. Syntax verification not available in local environment.",
    ),
    "csharp": LanguageCapability(
        id="csharp",
        name="C#",
        extension="cs",
        parser_type="regex_heuristic",
        has_real_parser=False,
        syntax_validation_capable=False,
        doc_format="XML Documentation (/// <summary>)",
        linter_available=None,
        notes="AI generation supported. Syntax verification not available in local environment.",
    ),
    "sql": LanguageCapability(
        id="sql",
        name="SQL",
        extension="sql",
        parser_type="regex_heuristic",
        has_real_parser=False,
        syntax_validation_capable=False,
        doc_format="SQL comments (-- ... / /* ... */)",
        linter_available=None,
        notes="AI generation supported. Syntax verification not available in local environment.",
    ),
}

SUPPORTED_LANGUAGES = tuple(REGISTRY.keys())


def get_language(lang_id: str) -> Optional[LanguageCapability]:
    if not lang_id:
        return None
    return REGISTRY.get(lang_id.lower().strip())


def is_supported(lang_id: str) -> bool:
    return lang_id.lower().strip() in REGISTRY


def list_capabilities() -> list[dict]:
    return [cap.to_dict() for cap in REGISTRY.values()]
