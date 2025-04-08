class ModelTraining:
    def __init__(self, CONFIG, dataset, data_loader, train_ratio):
        self.config = CONFIG
        split_index = int(train_ratio * len(dataset))
        train_data = dataset[:split_index]
        val_data = dataset[split_index:]
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
