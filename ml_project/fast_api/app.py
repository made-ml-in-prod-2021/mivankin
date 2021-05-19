import os
import pickle
import pandas as pd
import json
import requests
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, conlist
from typing import List, Union, Optional
from sklearn.pipeline import Pipeline


def load_model(path):
    """
    Function for load model

    Keyword arguments:
        - path - string with path for load model
    """
    with open(path, "rb") as file:
        return pickle.load(file)


class OutputDataModel(BaseModel):
    label: int

class PriceResponse(BaseModel):
    id: str
    price: float


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

app = FastAPI()

uci_model: None

@app.get("/")
def main():
    uci_model = load_model('model.pkl')
    uci_model.auc()
    return {uci_model.auc()[0]}

@app.get("/auc")
def get_auc():
    #uci_model = load_model('model.pkl').model
    return {uci_model.auc()[0]}

@app.get("/predict", response_model = OutputDataModel)
def predict(request: Item):
    uci_model = load_model('model.pkl')
    ans = uci_model.model.predict(request.convert_to_pandas())
    return OutputDataModel(label=uci_model.model.predict(request.convert_to_pandas()))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 50557))