from pathlib import Path
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--input_dir")
@click.option("--output_dir")
def train(input_dir: str, output_dir: str):
    input_dir_path = Path(input_dir)
    input_dir_path.mkdir(exist_ok=True, parents=True)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    model = LogisticRegression(solver='liblinear')
    data = pd.read_csv(input_dir_path / "train.csv", index_col=0)

    model.fit(data.drop("target", axis=1).values, data["target"].values)
    with open(output_dir_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
