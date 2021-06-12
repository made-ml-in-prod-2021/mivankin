import json
from pathlib import Path
import pickle

import click
import pandas as pd
from sklearn.metrics import f1_score


@click.command("train")
@click.option("--input_dir")
@click.option("--model_dir")
def validate(input_dir: str, model_dir: str):
    input_dir_path = Path(input_dir)
    input_dir_path.mkdir(exist_ok=True, parents=True)
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True, parents=True)

    test = pd.read_csv(input_dir_path / "test.csv", index_col=0)

    with open(model_dir_path / "model.pkl", "rb") as f:
        model = pickle.load(f)
    # Get metrics
    preds = model.predict(test.drop("target", axis=1).values)
    score = { 'f1' : f1_score(test["target"].values, preds, average=None)[1]}

    with open(input_dir_path / "metrics.json", "w") as f:
        json.dump(score, f)


if __name__ == '__main__':
    validate()
