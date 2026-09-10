"""--api-key: both header forms accepted, wrong/missing keys rejected, unset is a no-op."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.api_server import install_api_key_auth


def _client(api_key):
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"data": []}

    install_api_key_auth(app, api_key)
    return TestClient(app)


def test_no_key_configured_serves_openly():
    assert _client(None).get("/v1/models").status_code == 200
    assert _client("").get("/v1/models").status_code == 200


def test_bearer_and_x_api_key_both_accepted():
    c = _client("sk-secret")
    assert c.get("/v1/models", headers={"Authorization": "Bearer sk-secret"}).status_code == 200
    assert c.get("/v1/models", headers={"x-api-key": "sk-secret"}).status_code == 200


def test_missing_or_wrong_key_rejected():
    c = _client("sk-secret")
    assert c.get("/v1/models").status_code == 401
    assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/v1/models", headers={"x-api-key": "sk-secre"}).status_code == 401
    body = c.get("/v1/models").json()
    assert "API key" in body["error"]
