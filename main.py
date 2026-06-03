from __future__ import annotations

import os
import pickle

import numpy as np
import scipy.io as sio

from train import predict_with_model

MODEL_PATH = "model.pkl"

_MODEL = None
_PRED_CACHE = None
_CURRENT_USER = 0


def _load_model():
    global _MODEL

    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model.pkl not found. Submit main.py, train.py, report.md, and model.pkl together."
            )

        with open(MODEL_PATH, "rb") as f:
            _MODEL = pickle.load(f)

    return _MODEL


def _run_my_algorithm_for_all_users(d_hat, p_bs):
    model = _load_model()
    p_hat = predict_with_model(d_hat, p_bs, model)
    p_hat = np.asarray(p_hat, dtype=float)
    return p_hat


def your_algorithm(d_one, p_bs):
    global _PRED_CACHE, _CURRENT_USER

    result = _PRED_CACHE[:, _CURRENT_USER]
    _CURRENT_USER += 1
    return result


def main():
    # 1) 입력 데이터 로드 — 채점기가 같은 폴더에 .mat 파일 자동 배치
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data["BS_positions"], dtype=float)     # (2, 18)
    d_hat = np.asarray(data["d_hat"], dtype=float)                   # (18, num_user)
    p = np.asarray(data["p"], dtype=float)                           # (2, num_user) — GT 위치

    # 2) 본인 알고리즘 — 사용자 수는 입력에서 동적으로 받기
    num_user = d_hat.shape[1]

    global _PRED_CACHE, _CURRENT_USER
    _PRED_CACHE = _run_my_algorithm_for_all_users(d_hat, BS_positions)
    _CURRENT_USER = 0

    if _PRED_CACHE.shape != (2, num_user):
        raise ValueError(f"main() must return shape {(2, num_user)}, got {_PRED_CACHE.shape}")

    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    # 3) 결과 반환 — numpy 배열, 모양 (2, num_user)
    return p_hat


if __name__ == "__main__":
    result = main()
    print(f"p_hat shape: {result.shape}")