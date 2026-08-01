from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """The root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    """The health endpoint should report the screener status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_score_resumes_rejects_short_job_description():
    """Job descriptions under 10 characters should be rejected."""
    response = client.post(
        "/score-resumes",
        data={"job_description": "hi"},
        files={"files": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    )
    assert response.status_code == 422


def test_score_resumes_rejects_non_pdf_file():
    """Non-PDF files should be rejected with a clear message, not crash."""
    response = client.post(
        "/score-resumes",
        data={"job_description": "Looking for a Python developer with FastAPI experience"},
        files={"files": ("test.txt", b"just some text", "text/plain")}
    )
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["score"] == 0
    assert "only PDF files" in result["reason"]


def test_history_endpoint_returns_list():
    """The history endpoint should return a results list."""
    response = client.get("/history")
    assert response.status_code == 200
    assert "results" in response.json()
    assert isinstance(response.json()["results"], list)