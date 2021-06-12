from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import click


@click.command("predict")
@click.option("--input_dir")
@click.option("--output_dir")
@click.option("--model_dir")
def predict(input_dir: str, output_dir: str, model_dir: str):
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    input_dir_path = Path(input_dir)
    input_dir_path.mkdir(exist_ok=True, parents=True)
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(exist_ok=True, parents=True)
	
    data = pd.read_csv(input_dir_path / "features.csv", index_col=0)
    with open(model_dir_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(data).astype(int)
    np.savetxt(output_dir_path / "prediction.csv", preds)


if __name__ == '__main__':
    predict()
