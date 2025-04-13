import torch
from torch.utils.data import DataLoader
import tiktoken
from tqdm import tqdm

from src.utils.custom_collate import instruction_collate
from src.utils.datasets import InstructionDataset


class InstructionTraining:
    def __init__(
        self,
        model,
        CONFIG,
        dataset,
        loss_batch,
        loss_loader,
        batch_size=8,
    ):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = CONFIG
        self.batch_size = batch_size
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # Data splitting (85% train, 10% test, 5% val)
        n = len(dataset)
        train_data = dataset[: int(0.85 * n)]
        test_data = dataset[int(0.85 * n) : int(0.95 * n)]
        val_data = dataset[int(0.95 * n) :]

        # Initialize datasets
        self.train_dataset = InstructionDataset(
            train_data,
            self.tokenizer,
            max_length=CONFIG["context_length"],
            device=self.device,
        )
        self.test_dataset = InstructionDataset(
            test_data,
            self.tokenizer,
            max_length=CONFIG["context_length"],
            device=self.device,
        )
        self.val_dataset = InstructionDataset(
            val_data,
            self.tokenizer,
            max_length=CONFIG["context_length"],
            device=self.device,
        )

        # DataLoaders with automatic device placement
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda b: instruction_collate(b, device=self.device),
            shuffle=True,
            num_workers=0,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=lambda b: instruction_collate(b, device=self.device),
            num_workers=0,
        )

        # Loss functions
        self.loss_batch = loss_batch
        self.loss_loader = loss_loader

        # Training state
        self.scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")

    def train_epoch(
        self, optimizer, scheduler=None, eval_freq=50, epoch=1, total_epochs=100
    ):
        self.model.train()
        train_losses, val_losses = [], []

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="step",
        ) as pbar:
            for step, (inputs, targets) in enumerate(self.train_loader):
                with torch.amp.autocast(
                    device_type=(
                        self.device.type if self.device.type == "cuda" else "cpu"
                    )
                ):
                    loss = self.loss_batch(inputs, targets, self.model, self.device)

                # Gradient handling
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(optimizer)
                self.scaler.update()

                # Evaluation
                if (step + 1) % eval_freq == 0:
                    val_loss = self.validate()
                    train_losses.append(loss.item())
                    val_losses.append(val_loss)
                    pbar.set_postfix(
                        {"Training Loss": loss.item(), "Validation Loss": val_loss}
                    )

                pbar.update(1)

        if scheduler:
            scheduler.step()

        return train_losses, val_losses

    def validate(self, num_batches=None):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.val_loader):
                if num_batches and i >= num_batches:
                    break
                loss = self.loss_batch(inputs, targets, self.model, self.device)
                total_loss += loss.item()

        self.model.train()
        return total_loss / (num_batches or len(self.val_loader))

    def generate_sample(self, prompt, max_length=50):
        self.model.eval()
        with torch.no_grad():
            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt)], device=self.device
            )
            output = self.generate_text(
                input_ids, max_length=max_length, temperature=0.7
            )
        return self.tokenizer.decode(output[0].cpu().numpy())
