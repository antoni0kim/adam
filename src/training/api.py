from pathlib import Path
import time

import tiktoken
import torch

from src.Adam import AdamModel
from src.adam_config import ADAM_CONFIG
from src.training.ModelTraining import ModelTraining
from src.utils.dataloaders import create_text_dataloader
from src.utils.loss_calculation import loss_loader, loss_batch
from src.utils.token_conversions import token_ids_to_text, text_to_token_ids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdamModel(ADAM_CONFIG).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
tokenizer = tiktoken.get_encoding("gpt2")


def file_pretraining(file: str, epochs: int = 100) -> None:
    _load_model()
    with open(file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    training = ModelTraining(
        model=model,
        CONFIG=ADAM_CONFIG,
        dataset=raw_text,
        data_loader=create_text_dataloader,
        train_ratio=0.9,
        loss_loader=loss_loader,
        loss_batch=loss_batch,
        device=device,
    )

    start_time = time.time()

    train_losses, val_losses, _ = training.model_training(
        optimizer=optimizer, num_epochs=epochs, eval_freq=5, eval_iter=5
    )

    end_time = time.time()
    elapsed = end_time - start_time

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"🕒 Training time: {hours} hr {minutes} min {seconds} sec")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "adam_weights.pth",
    )


def directory_pretraining(directory: str, epochs: int = 100) -> None:
    dir_path = Path(directory).resolve()

    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"{directory} does not exist or is not a directory")

    text_files = list(dir_path.glob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No text files found in {directory}")

    for file in text_files:
        print(f"\n🚀 Starting pretraining for: {file.name}")
        try:
            file_pretraining(str(file), epochs=epochs)
        except Exception as e:
            raise RuntimeError(
                f"❌ Error occurred while training on {file.name}: {e}")


def generate_and_print(start_context):
    _load_model()
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = _generate_text_simple(
            idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def _load_model():
    adam_weights = Path("adam_weights.pth")
    if adam_weights.exists():
        print("Pre-train weights found.")
        weights = torch.load(adam_weights)
        model.load_state_dict(weights["model_state_dict"])
        optimizer.load_state_dict(weights["optimizer_state_dict"])


def _generate_text_simple(idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

            logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
