import os

import click



@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    pass


if __name__ == '__main__':
    download()