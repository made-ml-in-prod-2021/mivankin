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

from validation import validate

def load_model(path):
    """
    Function for load model

    Keyword arguments:
        - path - string with path for load model
    """
    with open(path, "rb") as file:
        return pickle.load(file)


class OutputItem(BaseModel):
    label: int

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

app = FastAPI()

uci_model: Optional[Pipeline] = None

@app.get("/")
def main():
    return "UCI service"

@app.on_event("startup")
def load():
    global uci_model
    uci_model = load_model('model.pkl')

@app.get("/auc")
def get_auc():
    #uci_model = load_model('model.pkl').model
    return {uci_model.auc()[0]}
    pass

@app.get("/health")
def health() -> bool:
    return not (uci_model is None)

@app.get("/predict", response_model = OutputItem)
def predict(request: Item):
    assert uci_model is not None
    check_data, fake_col = validate(uci_model, pd.DataFrame.from_dict([request.dict()], orient='columns'))
    if check_data:
        return OutputItem(label=uci_model.model.predict(pd.DataFrame.from_dict([request.dict()], orient='columns')))
    else:
        raise HTTPException(status_code=400, detail=f"error of data validation in {fake_col}")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))