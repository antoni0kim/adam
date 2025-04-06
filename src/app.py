import torch

from .Adam import AdamModel
from .adam_config import ADAM_CONFIG

device = torch.device("cuda")
model = AdamModel(ADAM_CONFIG)
model = model.to(device)

print(model)
