import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel, conlist
import pickle

def load_model(path):
    """
    Function for load model

    Keyword arguments:
        - path - string with path for load model
    """
    with open(path, "rb") as file:
        return pickle.load(file)

class Item(BaseModel):
    age: float
    sex: float
    cp:  float
    chol: float
    restecg: float
    thalach: float
    exang: float
    slope: float
    ca: float
    thal: float

    def convert_to_pandas(self) -> pd.DataFrame:
        data = pd.DataFrame.from_dict([self.dict()], orient='columns')
        return data

if __name__ == "__main__":
    data = {
        'age': 1,
        'sex': 1,
        'cp':  1,
        'chol': 1,
        'restecg': 1,
        'thalach': 1,
        'exang': 1,
        'slope': 1,
        'ca': 1,
        'thal': 1
    }


    #uci_model = load_model('model.pkl')

    #print(uci_model.auc()[0])

    #print(uci_model.model.predict(Item(**data).convert_to_pandas()))

    response = requests.get(
        "http://127.0.0.1:8000/predict",
        json=data,
    )

    print(response.status_code)
    print(response.json())
