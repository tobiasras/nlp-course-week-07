import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))

from main import app


def test_root():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "app is running"}


def test_query_is_blank():
    response = TestClient(app).post(
        "/v1/query",
        json={"text": "What is the width of a Yamaha P-150?"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "query": "What is the width of a Yamaha P-150?",
        "items_as_strings": [],
        "properties_as_strings": [],
        "sparql": "",
        "results": [],
    }
