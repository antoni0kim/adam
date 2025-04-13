import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Convert text file(s) into readable data for training in torch

    This dataset processes a text file into series of chunks for model
    training. Each chunk consists of max_length token for input and
    subsequent token as targets

    Args:
        text: location of the text file
        tokenizer: requires in order to convert text into tokens. Tokens
            are set of integers that is translated from words or part
            of words into numbers that the LLM can interpret
        max_length (int): maximum length of token sequence
        stride: interval size to slide the chunk window over tokenized
            text. Stride of 1 means there's no overlap, while higher
            stride allows overlapping chunks
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.inputs_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # convert to tensor and store them in chunks
            self.inputs_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.inputs_ids)

    # Return tokenized input and output pair in given index for training
    def __getitem__(self, idx):
        return self.inputs_ids[idx], self.target_ids[idx]


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024, device="cpu"):
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data

        # Validate data format
        assert all(
            k in entry for entry in data for k in ("instruction", "output")
        ), "Missing keys"

        # Pre-process
        self.encoded_texts = []
        for entry in data:
            text = self._format_text(entry)
            encoded = tokenizer.encode(text)[:max_length]
            self.encoded_texts.append(encoded)

    def _format_text(self, entry):
        parts = [
            "Below is an instruction that describes a task.",
            "Write a response that appropriately completes the request.",
            f"\n\n### Instruction:\n{entry['instruction']}",
            f"\n\n### Input:\n{entry['input']}" if entry["input"] else "",
            f"\n\n### Response:\n{entry['output']}",
        ]
        return "".join(parts)

    def __getitem__(self, index):
        return torch.tensor(
            self.encoded_texts[index], device=self.device, dtype=torch.long
        )

    def __len__(self):
        return len(self.data)
