import pytest
from fastapi.testclient import TestClient
from app import app, load_model, Item
from tests.features import DataFaker

TEST_SAMPLE_SIZE = 1

@pytest.fixture
def test_client():
    with TestClient(app) as test_client:
        yield test_client

def test_healh(test_client):
    result = test_client.get("/health")

    assert result.status_code == 200, (
        f"Loaded model expected with start service and status_code = 200 from /health rout, but return status_code {result.status_code}"
    )

def test_predict(test_client):
    uci_model = load_model('model.pkl')
    test_sample = DataFaker(uci_model.dataset)
    test_sample.ditribution_calc()
    test_data = test_sample.generate_samples(TEST_SAMPLE_SIZE)
    result = test_client.get("/predict", json=test_data.iloc[0].to_dict())

    assert result.status_code == 200, (
        f"Expected status_code = 200 from /predict rout, but return status_code {result.status_code}"
    )

    assert 'label' in result.json().keys(), (
        f"Expected 'label' key return in json from /predict rout, but in json keys only {result.json().keys()}"
    )

def test_validation(test_client):
    incorrect_data = {
        'age': 1,
        'sex': 1,
        'cp':  1,
        'chol': 15,
        'restecg': 1,
        'thalach': 1,
        'exang': 10,
        'slope': 10,
        'ca': 1,
        'thal': 1
    }

    result = test_client.get("/predict", json=incorrect_data)

    assert 400 == result.status_code, (
        f"Expected status_code = 400 from /predict rout, but return status_code {result.status_code}"
    )

    assert 'error of data validation in age' == result.json()['detail'], (
        f"Expected that data validation error in age collumn, but return detail: {result.json()['detail']}"
    )
