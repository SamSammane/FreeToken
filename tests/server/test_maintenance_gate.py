"""One maintenance state->detail table for every wire surface.

The regression this pins: "failed" (and "stopping") used to be reported by the
Anthropic/Responses/generate gates as "cache rebuild in progress" -- and the Anthropic
adapter tagged it overloaded_error, which tells SDK clients to retry forever against
an engine that requires a restart.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.anthropic_api import register_anthropic_routes
from freetoken.server.generation import maintenance_status


def _state(mstate):
    return SimpleNamespace(maintenance_state=mstate)


def test_status_table_covers_every_state():
    assert maintenance_status(_state("serving")) == ("serving", None)
    assert maintenance_status(_state("loading")) == ("loading", "model is still loading")
    assert maintenance_status(_state("rebuilding")) == ("rebuilding", "cache rebuild in progress")
    assert maintenance_status(_state("failed")) == (
        "failed", "maintenance failed (restart required)")
    assert maintenance_status(_state("stopping")) == ("stopping", "server is stopping")
    # a state-less object (early startup) counts as serving
    assert maintenance_status(SimpleNamespace()) == ("serving", None)


def _anthropic_client(mstate):
    app = FastAPI()
    register_anthropic_routes(app, lambda: _state(mstate), lambda: {})
    return TestClient(app)


def test_anthropic_failed_is_a_terminal_api_error():
    resp = _anthropic_client("failed").post("/v1/messages", json={
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 503
    body = json.loads(resp.content)
    assert body["error"]["type"] == "api_error"  # NOT overloaded_error: retries can't help
    assert "restart required" in body["error"]["message"]


def test_anthropic_rebuilding_stays_retryable():
    resp = _anthropic_client("rebuilding").post("/v1/messages", json={
        "model": "m", "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.status_code == 503
    body = json.loads(resp.content)
    assert body["error"]["type"] == "overloaded_error"
    assert "rebuild" in body["error"]["message"]


def test_openai_gate_reports_failed_and_stopping():
    from freetoken.server.openai_api import _maintenance_gate

    for mstate, needle in (
        ("failed", "restart required"),
        ("stopping", "stopping"),
        ("rebuilding", "rebuild"),
        ("loading", "loading"),
    ):
        resp = _maintenance_gate(_state(mstate))
        assert resp is not None and resp.status_code == 503
        assert needle in json.loads(resp.body)["error"]
    assert _maintenance_gate(_state("serving")) is None
