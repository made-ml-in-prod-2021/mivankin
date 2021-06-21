from pathlib import Path
import pickle

import click
import pandas as pd


@click.command("load")
@click.option("--input_dir")
@click.option("--output_dir")
def load(input_dir: str, output_dir: str):
	input_dir_path = Path(input_dir)
	input_dir_path.mkdir(exist_ok=True, parents=True)
	output_dir_path = Path(output_dir)
	output_dir_path.mkdir(exist_ok=True, parents=True)

	features = pd.read_csv(input_dir_path / "features.csv", index_col=0)
	target = pd.read_csv(input_dir_path / "target.csv", index_col=0)

	features.to_csv(output_dir_path / "features.csv")
	target.to_csv(output_dir_path / "target.csv")



if __name__ == '__main__':
    load()
