"""DocuPilot AI Reproducible Benchmark Suite.

Executes representative test fixtures for:
1. Simple typed functions
2. Complex multi-argument business functions
3. Exception-raising functions
4. Procedural / interactive scripts
5. Malformed syntax detection
6. Executable code mutation (structural preservation failure)
7. JavaScript functions
8. TypeScript typed code

Measures and reports real latency, syntax detection, preservation failure detection,
quality scores, and estimated cost without fabricated numbers.
"""

import ast
import json
import os
import sys
import time
from typing import Any

# Ensure backend directory is in sys.path so 'app' imports resolve regardless of cwd
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.analyzer import analyze, compute_executable_fingerprint
from app.languages import list_capabilities
from app.main import generate_python_documentation, mock_documentation
from app.quality import evaluate_documentation_quality
from app.verifier import verify, verify_structural_consistency, verify_syntax


BENCHMARK_FIXTURES = [
    {
        "id": "simple_tax_calc",
        "name": "Simple Typed Function",
        "language": "python",
        "doc_style": "google",
        "source": """def calculate_tax(amount: float, rate: float = 0.05) -> float:
    if amount < 0:
        raise ValueError("Amount cannot be negative")
    return amount * rate
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
    {
        "id": "complex_order_pipeline",
        "name": "Complex Multi-Argument Function (NumPy Style)",
        "language": "python",
        "doc_style": "numpy",
        "source": """def process_order(price: float, discount: float, express: bool) -> float:
    if price <= 0:
        raise ValueError("Price must be positive")
    discounted = price * (1.0 - (discount / 100.0))
    shipping = 15.0 if express else 5.0
    return discounted + shipping
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
    {
        "id": "exception_handling",
        "name": "Multi-Exception Function",
        "language": "python",
        "doc_style": "google",
        "source": """def validate_user_payload(data: dict) -> bool:
    if not isinstance(data, dict):
        raise TypeError("Payload must be a dictionary")
    if "username" not in data:
        raise KeyError("Missing required field 'username'")
    if len(data["username"]) < 3:
        raise ValueError("Username too short")
    return True
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
    {
        "id": "procedural_script",
        "name": "Procedural Console Script",
        "language": "python",
        "doc_style": "google",
        "source": """price = float(input("Enter base price: "))
discount = float(input("Enter discount: "))
total = price * (1 - discount / 100)
print(f"Final total is {total}")
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
    {
        "id": "malformed_python",
        "name": "Malformed Syntax Detection",
        "language": "python",
        "doc_style": "google",
        "source": """def broken_syntax(x, y:
    return x + y
""",
        "expected_syntax": False,
        "expected_preservation": None,
    },
    {
        "id": "mutated_executable_ast",
        "name": "Executable Mutation (Preservation Failure Detection)",
        "language": "python",
        "doc_style": "google",
        "source": """def compute_discount(price: float, discount: float) -> float:
    return price * (1.0 - (discount / 100.0))
""",
        "mutated_proposal": """def compute_discount(price: float, discount: float) -> float:
    \"\"\"Calculates discount with mutated operator.\"\"\"
    return price * (1.0 + (discount / 100.0))
""",
        "expected_syntax": True,
        "expected_preservation": False,
    },
    {
        "id": "javascript_function",
        "name": "JavaScript ES6 Function",
        "language": "javascript",
        "doc_style": "jsdoc",
        "source": """function calculateShipping(weight, distance) {
    if (weight <= 0) {
        throw new Error("Invalid weight");
    }
    const rate = weight * 0.75 + distance * 0.1;
    return rate;
}
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
    {
        "id": "typescript_function",
        "name": "TypeScript Typed Function",
        "language": "typescript",
        "doc_style": "jsdoc",
        "source": """function formatSummary(name: string, total: number): string {
    return `${name}: $${total.toFixed(2)}`;
}
""",
        "expected_syntax": True,
        "expected_preservation": True,
    },
]


def run_benchmark():
    print("=" * 80)
    print("   DocuPilot AI — Benchmark & Verification Report (Hackathon 2026)")
    print("=" * 80)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"Python: {sys.version.split()[0]} on {sys.platform}")
    print(f"Environment Mode: {'LIVE OPENAI' if os.getenv('OPENAI_API_KEY') else 'DEMO MOCK ENGINE'}")
    print("-" * 80)

    results = []
    total_start = time.perf_counter()

    for idx, fixture in enumerate(BENCHMARK_FIXTURES, 1):
        f_id = fixture["id"]
        f_name = fixture["name"]
        lang = fixture["language"]
        source = fixture["source"]
        doc_style = fixture["doc_style"]

        start_time = time.perf_counter()

        # Step 1: Analysis
        analysis = analyze(source, lang)
        syntax_valid = analysis.get("valid_syntax")

        # Step 2: Documentation proposal
        if "mutated_proposal" in fixture:
            proposed = fixture["mutated_proposal"]
            mode = "synthetic_mutation"
        else:
            if lang == "python":
                proposed = generate_python_documentation(source, analysis, doc_style=doc_style)
            else:
                proposed = mock_documentation(source, lang, analysis, doc_style=doc_style)
            mode = "local_engine"

        # Step 3: Verification & Preservation check
        if syntax_valid:
            verification = verify(
                proposed_code=proposed,
                language=lang,
                original_code=source,
                doc_style=doc_style,
            )
            struct_check = [c for c in verification["checks"] if c["name"] == "Structural Consistency"]
            preservation_passed = (struct_check[0]["status"] == "passed") if struct_check else False
            quality_score = verification.get("quality_score", {}).get("overall_score", 0)
            grade = verification.get("quality_score", {}).get("grade", "N/A")
            fp_matches = verification.get("checks", [{}, {}, {}])[2].get("fingerprint_comparison", {}).get("matches", False)
        else:
            preservation_passed = None
            quality_score = 0
            grade = "N/A"
            fp_matches = None

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Expected assertion checks
        syntax_correct = (syntax_valid == fixture["expected_syntax"])
        preservation_correct = (preservation_passed == fixture["expected_preservation"])

        status = "PASSED" if (syntax_correct and preservation_correct) else "FAILED"

        results.append({
            "fixture": f_name,
            "language": lang,
            "style": doc_style,
            "syntax_valid": syntax_valid,
            "preservation": preservation_passed,
            "fingerprint_match": fp_matches,
            "quality_score": quality_score,
            "grade": grade,
            "latency_ms": latency_ms,
            "status": status,
        })

        print(
            f"[{idx}/8] {f_name:<36} | {lang:<10} | {latency_ms:>6.2f}ms | "
            f"Q: {quality_score:>3}/100 ({grade:<2}) | "
            f"Preserve: {str(preservation_passed):<5} | Status: {status}"
        )

    total_duration = round((time.perf_counter() - total_start) * 1000, 2)

    print("-" * 80)
    print("BENCHMARK SUMMARY RESULTS:")
    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    print(f"Total Fixtures Tested: {len(results)}")
    print(f"Passed All Criteria:   {passed_count} / {len(results)} ({round((passed_count / len(results)) * 100)}%)")
    print(f"Total Benchmark Time:  {total_duration} ms")
    avg_latency = round(sum(r["latency_ms"] for r in results) / len(results), 2)
    print(f"Average Engine Latency:{avg_latency} ms")
    valid_scores = [r["quality_score"] for r in results if r["quality_score"] > 0]
    avg_score = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0
    print(f"Average Quality Score: {avg_score} / 100")
    print("-" * 80)
    print("DISCLAIMER & SEMANTIC LIMITATIONS:")
    print("Static syntax parsing (Python AST / Esprima) and executable AST fingerprinting")
    print("guarantee that executable statements have not been altered or deleted.")
    print("Static checks do NOT claim or guarantee runtime behavioral semantic equivalence.")
    print("=" * 80)

    all_passed = (passed_count == len(results))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_benchmark())
