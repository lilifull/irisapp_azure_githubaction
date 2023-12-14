import pytest
from Model import IrisModel

def model_init(model_name):
    return IrisModel(model_name)

@pytest.fixture

def test_valid_model_loading():
    instance = model_init('iris_model.pkl')
    assert instance.model is not None

def test_invalid_model_loading():
    with pytest.raises(FileNotFoundError):
        model_init('none.pkl')

def test_prediction():
    sepal_length = 5.1
    sepal_width = 3.5
    petal_length = 1.4
    petal_width = 0.2

    species, probability = model_init('iris_model.pkl').predict_species(
        sepal_length, sepal_width, petal_length, petal_width
    )

    assert isinstance(species, str)
    assert isinstance(probability, float)
    assert 0 <= probability <= 1

def test_invalid_prediction_input():
    with pytest.raises(ValueError):
        model_init('iris_model.pkl').predict_species('a', 3.5, 1.4, 0.2)
