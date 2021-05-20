import pandas as pd
import numpy as np
import requests
from pydantic import BaseModel, conlist
import pickle

if __name__ == "__main__":
    data = {
        'age': 58,
        'sex': 0,
        'cp':  2,
        'chol': 258,
        'restecg': 0,
        'thalach': 170,
        'exang': 0,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }


    response = requests.get(
        "http://127.0.0.1:8000/predict",
        json=data,
    )

    print(response.status_code)
    print(response.json())
