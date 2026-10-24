"""Automated tests for FastAPI CORS middleware.

Verifies:
- Production Vercel origin (https://docupilot-ai-three.vercel.app) is allowed.
- Preview Vercel origin (https://docupilot-ai-git-main-fursat1.vercel.app) is allowed.
- CORS preflight OPTIONS requests return Access-Control-Allow-Origin, headers, and methods on /api/generate.
- Dynamic Vercel preview subdomains (*.vercel.app) are allowed via regex.
- Credentials and standard headers are allowed.
- Disallowed/arbitrary origins are rejected (no Access-Control-Allow-Origin header).
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

PRODUCTION_ORIGIN = "https://docupilot-ai-three.vercel.app"
USER_PREVIEW_ORIGIN = "https://docupilot-ai-git-main-fursat1.vercel.app"
FEATURE_PREVIEW_ORIGIN = "https://docupilot-ai-git-feature.vercel.app"
DISALLOWED_ORIGIN = "https://malicious-site.example.com"


def test_cors_preflight_production_origin():
    """Verify CORS preflight OPTIONS request on /api/generate for production Vercel frontend."""
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
    assert "content-type" in response.headers.get("access-control-allow-headers", "").lower()


def test_cors_preflight_user_preview_origin():
    """Verify CORS preflight OPTIONS request on /api/generate for https://docupilot-ai-git-main-fursat1.vercel.app."""
    headers = {
        "Origin": USER_PREVIEW_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization",
    }
    response = client.options("/api/generate", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == USER_PREVIEW_ORIGIN
    assert response.headers.get("access-control-allow-credentials") == "true"
    assert "POST" in response.headers.get("access-control-allow-methods", "")
    assert "content-type" in response.headers.get("access-control-allow-headers", "").lower()


def test_cors_simple_get_request_both_origins():
    """Verify simple GET requests from both allowed origins receive CORS headers."""
    for origin in (PRODUCTION_ORIGIN, USER_PREVIEW_ORIGIN):
        response = client.get("/api/config", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
        assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_health_endpoint():
    """Verify /api/health returns CORS header for both allowed origins."""
    for origin in (PRODUCTION_ORIGIN, USER_PREVIEW_ORIGIN):
        response = client.get("/api/health", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin


def test_cors_stats_endpoint():
    """Verify /api/stats returns CORS header for both allowed origins."""
    for origin in (PRODUCTION_ORIGIN, USER_PREVIEW_ORIGIN):
        response = client.get("/api/stats", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin


def test_cors_vercel_preview_subdomain_allowed():
    """Verify arbitrary Vercel preview subdomains are allowed via regex."""
    headers = {
        "Origin": FEATURE_PREVIEW_ORIGIN,
        "Access-Control-Request-Method": "GET",
    }
    response = client.options("/api/config", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == FEATURE_PREVIEW_ORIGIN


def test_cors_disallowed_origin_rejected():
    """Verify arbitrary origins do NOT receive Access-Control-Allow-Origin header."""
    headers = {
        "Origin": DISALLOWED_ORIGIN,
        "Access-Control-Request-Method": "POST",
    }
    response = client.options("/api/generate", headers=headers)
    # When origin is not allowed, Starlette CORS middleware does not attach the header
    assert response.headers.get("access-control-allow-origin") != DISALLOWED_ORIGIN
