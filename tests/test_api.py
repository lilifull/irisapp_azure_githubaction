from fastapi.testclient import TestClient
from app import app
from Model import IrisSpecies

client = TestClient(app)

def test_predict_species():
    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post('/predict', json=input_data)

    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert "probability" in response_json

    assert isinstance(response_json["prediction"], str)
    assert isinstance(response_json["probability"], float)
    assert 0 <= response_json["probability"] <= 1
