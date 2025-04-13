import torch


def instruction_collate(batch, pad_token_id=50256, ignore_index=-100, device="cpu"):
    # Find max length in batch
    batch_max = max(len(x) for x in batch)

    # Process all items
    inputs, targets = [], []
    for item in batch:
        # Convert to tensor and pad
        if isinstance(item, torch.Tensor):
            item_tensor = item.clone().detach().to(device)
        else:
            item_tensor = torch.tensor(item, dtype=torch.long, device=device)
        pad_amount = batch_max - len(item_tensor)
        padded = torch.cat(
            [
                item_tensor,
                torch.full(
                    (pad_amount,), pad_token_id, dtype=torch.long, device=device
                ),
            ]
        )

        # Inputs: all tokens except last
        inputs.append(padded[:-1])

        # Targets: all tokens except first (mask padded positions)
        t = padded[1:].clone()
        t[pad_amount - 1 :] = ignore_index
        targets.append(t)

    return torch.stack(inputs), torch.stack(targets)
