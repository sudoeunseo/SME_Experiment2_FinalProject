from __future__ import annotations

import os
import pickle
import numpy as np

from train import load_mat_data, predict_with_model

MODEL_PATH = "model.pkl"


def main() -> np.ndarray:
    mat_path = "DH_FR1.mat"
    p_bs, d_hat, _p_optional, _meta = load_mat_data(mat_path)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.pkl not found. Run `py train.py` on the labeled training file first, "
            "then submit main.py, train.py, report.md, and model.pkl together."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    p_hat = predict_with_model(d_hat, p_bs, model)
    p_hat = np.asarray(p_hat, dtype=float)
    expected_shape = (2, d_hat.shape[1])
    if p_hat.shape != expected_shape:
        raise ValueError(f"main() must return shape {expected_shape}, got {p_hat.shape}")
    return p_hat


if __name__ == "__main__":
    result = main()
    print(f"p_hat shape: {result.shape}")
