import uvicorn
from fastapi import FastAPI
from model import IrisModel, IrisSpecies

app = FastAPI()
model = IrisModel('iris_model.pkl')

@app.post('/predict')
def predict_species(iris: IrisSpecies):
    data = iris.dict()
    prediction, probability = model.predict_species(
        data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']
    )
    return {
        'prediction': prediction,
        'probability': probability
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
