import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

client = TestClient(app)

# Test 1 : Texte valide => tags attendus
@patch("app.use_model")
@patch("app.svc_model")
@patch("app.mlb")
def test_predict_tags(mock_mlb, mock_svc_model, mock_use_model):
    mock_embedding = [[0.1] * 512]
    mock_use_model.return_value = MagicMock()
    mock_use_model.return_value.numpy.return_value = mock_embedding

    mock_svc_model.predict.return_value = [[1, 0, 1]]
    mock_mlb.inverse_transform.return_value = [["python", "pandas"]]

    response = client.post("/predict", json={"text": "How to use pandas in python?"})

    assert response.status_code == 200
    assert "tags" in response.json()
    assert response.json()["tags"] == ["python", "pandas"]

# Test 2 : Texte vide => devrait retourner quand même une réponse (selon ton choix)
@patch("app.use_model")
@patch("app.svc_model")
@patch("app.mlb")
def test_empty_text(mock_mlb, mock_svc_model, mock_use_model):
    mock_embedding = [[0.0] * 512]
    mock_use_model.return_value = MagicMock()
    mock_use_model.return_value.numpy.return_value = mock_embedding

    mock_svc_model.predict.return_value = [[0, 0, 0]]
    mock_mlb.inverse_transform.return_value = [[]]

    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 200
    assert "tags" in response.json()
    assert response.json()["tags"] == []

# Test 3 : Pas de champ "text" dans le JSON => erreur 422 (Unprocessable Entity)
def test_missing_text_field():
    response = client.post("/predict", json={"wrong_field": "test"})
    assert response.status_code == 422

# Test 4 : Autre texte valide => autres tags
@patch("app.use_model")
@patch("app.svc_model")
@patch("app.mlb")
def test_another_text(mock_mlb, mock_svc_model, mock_use_model):
    mock_embedding = [[0.5] * 512]
    mock_use_model.return_value = MagicMock()
    mock_use_model.return_value.numpy.return_value = mock_embedding

    mock_svc_model.predict.return_value = [[0, 1, 1]]
    mock_mlb.inverse_transform.return_value = [["machine-learning", "tensorflow"]]

    response = client.post("/predict", json={"text": "How to train a model using TensorFlow?"})

    assert response.status_code == 200
    assert "tags" in response.json()
    assert response.json()["tags"] == ["machine-learning", "tensorflow"]

# Test 5 : Mauvais format JSON (ex. : string au lieu d’objet)
def test_invalid_json():
    response = client.post(
        "/predict",
        data="just a string",  # type: ignore[arg-type]
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


