import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = joblib.load(self.model_name)

    def predict_species(self, sepal_length, sepal_width, petal_length, petal_width):
        data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability
