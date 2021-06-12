from pathlib import Path
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input_dir")
@click.option("--output_dir")
def split(input_dir: str, output_dir: str):
    input_dir_path = Path(input_dir)
    input_dir_path.mkdir(exist_ok=True, parents=True)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)

    features = pd.read_csv(input_dir_path / "features.csv")
    target = pd.read_csv(input_dir_path / "target.csv")
    train, test = train_test_split(features.merge(target), test_size=0.2)

    train.to_csv(output_dir_path / "train.csv", index=False)
    test.to_csv(output_dir_path / "test.csv", index=False)


if __name__ == '__main__':
    split()
