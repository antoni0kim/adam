import torch
import torch.nn.functional as F


def loss_batch(inputs, targets, model, device):
    inputs, targets = inputs.to(device), targets.to(device)
    logits = model(inputs)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    return loss


def loss_loader(dataloader, model, device, num_batches=None):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        num_batches = min(num_batches or len(dataloader), len(dataloader))
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
            total_loss += loss_batch(inputs, targets, model, device)

    return total_loss / num_batches if num_batches > 0 else float("nan")
