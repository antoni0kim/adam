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
