import torch
from pathlib import Path

# Create models directory

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

# Create model save path
MODEL_NAME = "pytorch_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


