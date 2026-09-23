import pytest
from fastapi.testclient import TestClient

from stellar_ai.api.app import create_app


@pytest.fixture
def client(system):
    return TestClient(create_app(system=system))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks"] >= 2


def test_ingest_and_query(client):
    r = client.post("/ingest", json={"text": "太阳是离地球最近的恒星，提供光和热。", "title": "太阳"})
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/query", json={"question": "恒星的主要成分是什么？", "top_n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["doc_id"] == "d1"


def test_agent_calculator(client):
    r = client.post("/agent", json={"task": "计算 7 * 6"})
    assert r.status_code == 200
    assert "42" in r.json()["result"]


def test_index_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
