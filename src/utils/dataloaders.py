import torch
import tiktoken
from torch.utils.data import DataLoader

from .datasets import TextDataset, InstructionDataset


def create_text_dataloader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def create_instruction_dataloader(
    data, collate_fn, batch_size=4, shuffle=False, drop_last=False, num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instruction_dataset = InstructionDataset(data, tokenizer, device=device)
    dataloader = DataLoader(
        instruction_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
