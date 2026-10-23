"""Automated tests for FastAPI CORS middleware.

Verifies:
- Production Vercel origin (https://docupilot-ai-three.vercel.app) is allowed.
- CORS preflight OPTIONS requests return Access-Control-Allow-Origin and methods.
- Dynamic Vercel preview subdomains (*.vercel.app) are allowed via regex.
- Credentials and standard headers are allowed.
- Disallowed/arbitrary origins are rejected (no Access-Control-Allow-Origin header).
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

PRODUCTION_ORIGIN = "https://docupilot-ai-three.vercel.app"
PREVIEW_ORIGIN = "https://docupilot-ai-git-feature.vercel.app"
DISALLOWED_ORIGIN = "https://malicious-site.example.com"


def test_cors_preflight_production_origin():
    """Verify CORS preflight OPTIONS request for production Vercel frontend."""
    headers = {
        "Origin": PRODUCTION_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }
    response = client.options("/api/generate", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == PRODUCTION_ORIGIN
    assert response.headers.get("access-control-allow-credentials") == "true"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_cors_simple_get_request():
    """Verify simple GET request from production frontend receives CORS headers."""
    response = client.get("/api/config", headers={"Origin": PRODUCTION_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == PRODUCTION_ORIGIN
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_health_endpoint():
    """Verify /api/health returns CORS header for production frontend."""
    response = client.get("/api/health", headers={"Origin": PRODUCTION_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == PRODUCTION_ORIGIN


def test_cors_stats_endpoint():
    """Verify /api/stats returns CORS header for production frontend."""
    response = client.get("/api/stats", headers={"Origin": PRODUCTION_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == PRODUCTION_ORIGIN


def test_cors_vercel_preview_subdomain_allowed():
    """Verify Vercel preview subdomains are allowed via regex."""
    headers = {
        "Origin": PREVIEW_ORIGIN,
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/config", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == PREVIEW_ORIGIN


def test_cors_disallowed_origin_rejected():
    """Verify arbitrary origins do NOT receive Access-Control-Allow-Origin header."""
    headers = {
        "Origin": DISALLOWED_ORIGIN,
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/generate", headers=headers)
    # When origin is not allowed, Starlette CORS middleware does not attach the header
    assert response.headers.get("access-control-allow-origin") != DISALLOWED_ORIGIN
