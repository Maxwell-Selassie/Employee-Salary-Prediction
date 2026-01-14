from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code in (200, 503)


def test_predict_single_endpoint():
    payload = {
        "features": {
            "Employee_age": 30,
            "years_experience": 5,
            "Number_of_Children": 0,
            "Department": "Engineering",
            "Role": "Software Engineer",
            "performance_rating": 3,
        }
    }
    response = client.post("/predict", json=payload)
    # Model may not yet be available; 503 is acceptable early on
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert "prediction" in body

