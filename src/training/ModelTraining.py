import torch


class ModelTraining:
    def __init__(
        self,
        model,
        CONFIG,
        dataset,
        data_loader,
        train_ratio,
        loss_batch,
        loss_loader,
        device,
    ):
        self.model = model
        self.config = CONFIG
        self.device = device

        # Data splitting
        split_index = int(train_ratio * len(dataset))
        train_data = dataset[:split_index]
        val_data = dataset[split_index:]

        # Data loaders
        assert CONFIG["context_length"] > 0, "Context length must be positive"

        self.training_loader = data_loader(
            train_data,
            batch_size=2,
            max_length=CONFIG["context_length"],
            stride=CONFIG["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )
        self.validation_loader = data_loader(
            val_data,
            batch_size=2,
            max_length=CONFIG["context_length"],
            stride=CONFIG["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

        # Loss functions
        self.loss_batch = loss_batch
        self.loss_loader = loss_loader

    def model_training(self, optimizer, num_epochs, eval_freq, eval_iter):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, 0

        eval_freq = max(eval_freq, 1)

        for epoch in range(num_epochs):
            self.model.train()
            for input_batch, target_batch in self.training_loader:
                optimizer.zero_grad()
                loss = self.loss_batch(
                    input_batch, target_batch, self.model, self.device
                )
                if torch.isnan(loss):
                    raise ValueError(f"NaN loss at epoch {epoch}, step {global_step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                # Evaluation
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch+1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, "
                        f"Val loss {val_loss:.3f}"
                    )

        return train_losses, val_losses, track_tokens_seen

    def evaluate_model(self, eval_iter):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.loss_loader(
                self.training_loader, self.model, self.device, num_batches=eval_iter
            )
            val_loss = self.loss_loader(
                self.validation_loader, self.model, self.device, num_batches=eval_iter
            )
        self.model.train()
        return train_loss, val_loss
