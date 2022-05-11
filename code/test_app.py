"""
Unit tests for the API app.py
"""
import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module", name="test_app")
def fixture_test_app():
    """
    Define the clinet
    """
    client = TestClient(app)
    yield client  # testing happens here

def test_main_page(test_app):
    """
    Check if the main page is running.
    """
    response = test_app.get("/")
    assert response.status_code == 200

def test_predict_route(test_app):
    """
    Unit test for the /predict route
    """
    x_input = [1.2, 0.31, -1.44, -1.11, 1.86]
    y_input = [-0.3, 1.6, -2.65, -0.93, -1.38]
    response_ = [1, 0, -1, 2, 1]
    dict_data = []
    for x_i, y_i in zip(x_input,y_input):
        dict_data.append({'x':x_i, 'y':y_i})
    for i, _ in enumerate(dict_data):
        data = dict_data[i]
        response = test_app.post("/predict", json=data )
        assert response.status_code == 200
        assert response.json() == { "response": response_[i]}
