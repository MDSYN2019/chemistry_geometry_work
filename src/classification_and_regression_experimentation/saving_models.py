from pathlib import Path
import torch

def save_model(model_name: str, model) -> None:
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents = True, exist_ok = True)

    MODEL_NAME = model_name
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj = model.state_dict(), f = MODEL_SAVE_PATH)
