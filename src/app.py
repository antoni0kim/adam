from pathlib import Path

import typer
from typing import Annotated

from src.training.api import file_pretraining, generate_and_print

app = typer.Typer()


@app.command()
def version():
    print("0.0.1")


@app.command()
def pretrain(
    file: Annotated[str, typer.Argument()],
    directory: Annotated[str, typer.Option("--dir", "-d")] = ".",
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 10,
):
    """Pretrain Adam's base model.

    Requires file's path. If directory is set, trains every text file
    in the directory.
    """
    curr_dir = Path(directory)
    file_path = (curr_dir / file).resolve()
    file_pretraining(file_path, epochs)


@app.command()
def generate(text: Annotated[str, typer.Argument()]):
    generate_and_print(text)


if __name__ == "__main__":
    app()
