import numpy as np

model_metrics = {
    "xgboost": {"AUC": 0.8757, "F1": 0.8097, "Precision": 0.7335, "Recall": 0.9034},
    "lightgbm": {"AUC": 0.8504, "F1": 0.7907, "Precision": 0.7129, "Recall": 0.8876},
    "tabnet": {"AUC": 0.9115, "F1": 0.8425, "Precision": 0.7829, "Recall": 0.9120},
}

def sigmoid(x: float) -> float:
    """Standard sigmoid function to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))