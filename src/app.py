from pathlib import Path

import typer
from typing import Annotated, Optional

from src.training.api import (
    file_pretraining,
    instructions_pretraining,
    directory_pretraining,
    generate_and_print,
    ask_question,
)

app = typer.Typer()


@app.command()
def version():
    print("0.0.1")


@app.command()
def pretrain(
    file: Annotated[Optional[str], typer.Option("--file", "-f")] = None,
    directory: Annotated[str, typer.Option("--dir", "-d")] = ".",
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 10,
):
    """Pretrain Adam's base model.

    Requires file's path. If directory is set, trains every text file
    in the directory.
    """
    curr_dir = Path(directory)

    if file:
        file_path = (curr_dir / file).resolve()
        file_pretraining(str(file_path), epochs=epochs)
    else:
        directory_pretraining(str(curr_dir), epochs=epochs)


@app.command()
def train(
    file: Annotated[Optional[str], typer.Option("--file", "-f")] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e")] = 10,
):
    curr_dir = Path.cwd()
    file_path = (curr_dir / file).resolve()
    instructions_pretraining(str(file_path), epochs=epochs)


@app.command()
def generate(text: Annotated[str, typer.Argument()]):
    generate_and_print(text)


@app.command()
def ask(question: Annotated[str, typer.Argument()]):
    ask_question(question)


if __name__ == "__main__":
    app()
