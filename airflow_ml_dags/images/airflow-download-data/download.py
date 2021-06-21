from pathlib import Path
import click
from sklearn.datasets import load_iris


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    X, y = load_iris(return_X_y=True, as_frame=True)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    X.to_csv(output_dir_path / "features.csv")
    y.to_csv(output_dir_path / "target.csv")


if __name__ == '__main__':
    download()