from __future__ import annotations

import os
import pickle
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
CV_FOLDS = 5
EPS = 1e-9
MODEL_PATH = "model.pkl"

# V2 핵심: pure physics가 아니라, fingerprint prior를 가진 robust physics refinement.
EXPERT_NAMES = [
    "KNN_rankdiff_k7_p2",
    "KNN_rankdiff_k21_p2",
    "Local_Ridge_KNN40",
    "ExtraTrees_RobustFeatures",
    "HistGBR_RobustFeatures",
    "MAP_Robust_NLS_KNNPrior",
    "MAP_Trimmed_Residual_NLS",
    "Calibration_MAP_NLS",
    "RANSAC_Triplet_Multilateration",
    "Calibrated_RANSAC_Triplet",
    "Residual_Reranked_Top20_Fingerprint",
    "SafeAffine_MAP_NLS",
    "SafeAffine_Trimmed_NLS",
]


def _as_2_by_n(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {a.shape}")
    if a.shape[0] == 2:
        return a
    if a.shape[1] == 2:
        return a.T
    raise ValueError(f"{name} must have one dimension equal to 2, got {a.shape}")


def _as_m_by_n(a: np.ndarray, m: int = 18, name: str = "d_hat") -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {a.shape}")
    if a.shape[0] == m:
        return a
    if a.shape[1] == m:
        return a.T
    raise ValueError(f"{name} must be ({m}, N) or (N, {m}), got {a.shape}")


def _extract_zip_if_needed(zip_path: Path) -> Path:
    out_dir = zip_path.parent
    with zipfile.ZipFile(zip_path, "r") as zf:
        mats = [n for n in zf.namelist() if n.lower().endswith(".mat")]
        if not mats:
            raise FileNotFoundError(f"No .mat file in {zip_path}")
        target_name = mats[0]
        out_path = out_dir / Path(target_name).name
        if not out_path.exists():
            zf.extract(target_name, out_dir)
            extracted = out_dir / target_name
            if extracted != out_path:
                extracted.rename(out_path)
        return out_path


def resolve_mat_path(mat_path: str | os.PathLike[str] = "DH_FR1.mat") -> Path:
    candidates = [Path(mat_path), Path("DH_FR1.mat"), Path("InF_DH_FR1.mat")]
    for p in candidates:
        if p.exists():
            return p
    zip_candidates = [Path("InF_DH_FR1.mat.zip"), Path("DH_FR1.mat.zip")]
    for z in zip_candidates:
        if z.exists():
            return _extract_zip_if_needed(z)
    raise FileNotFoundError("Could not find DH_FR1.mat, InF_DH_FR1.mat, or InF_DH_FR1.mat.zip")


def load_mat_data(mat_path: str | os.PathLike[str] = "DH_FR1.mat") -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, Dict[str, Any]]:
    used_path = resolve_mat_path(mat_path)
    data = loadmat(used_path)
    keys = [k for k in data.keys() if not k.startswith("__")]
    if "d_hat" not in data:
        raise KeyError(f"d_hat not found. variables={keys}")
    # 공식 README §3 기준: BS_positions 또는 p_bs 양쪽 처리
    if "BS_positions" in data:
        p_bs = _as_2_by_n(data["BS_positions"], "BS_positions")
    elif "p_bs" in data:
        p_bs = _as_2_by_n(data["p_bs"], "p_bs")
    else:
        raise KeyError(f"Neither BS_positions nor p_bs found. variables={keys}")
    d_hat = _as_m_by_n(data["d_hat"], p_bs.shape[1], "d_hat")
    p = _as_2_by_n(data["p"], "p") if "p" in data else None
    meta: Dict[str, Any] = {
        "used_path": str(used_path),
        "variables": keys,
    }
    if "indices" in data:
        meta["indices_shape"] = tuple(np.asarray(data["indices"]).shape)
    return p_bs, d_hat, p, meta


def sanitize_distances(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float).copy()
    m, n = d.shape
    for i in range(m):
        row = d[i]
        finite_pos = np.isfinite(row) & (row > 0)
        if not finite_pos.any():
            fill = 1.0
        else:
            fill = float(np.median(row[finite_pos]))
        bad = ~np.isfinite(row) | (row <= 0)
        row[bad] = fill
        q_hi = float(np.percentile(row[finite_pos], 99.5)) if finite_pos.sum() > 5 else float(np.max(row))
        row[:] = np.clip(row, 0.05, max(q_hi, 1.0))
        d[i] = row
    return d


def true_ranges(p: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    diff = p.T[:, None, :] - p_bs.T[None, :, :]
    return np.linalg.norm(diff, axis=2).T


def robust_mad(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return 1.4826 * mad


def fit_calibration(d_train: np.ndarray, p_train: np.ndarray, p_bs: np.ndarray) -> Dict[str, np.ndarray | float]:
    d_train = sanitize_distances(d_train)
    tr = true_ranges(p_train, p_bs)
    err = d_train - tr
    bias = np.median(err, axis=1)
    centered = err - bias[:, None]
    sigma = robust_mad(centered, axis=1)
    sigma = np.clip(sigma, 0.25, np.percentile(sigma, 90) * 2.0 + 0.25)
    d_corr = d_train - bias[:, None]
    qlo = np.percentile(d_corr, 0.5, axis=1)
    qhi = np.percentile(d_corr, 99.5, axis=1)
    qlo = np.maximum(qlo, 0.05)
    span = np.linalg.norm(np.max(p_bs, axis=1) - np.min(p_bs, axis=1))
    qhi = np.maximum(qhi, qlo + 1.0)
    qhi = np.minimum(qhi, max(span * 3.0, np.max(qhi)))
    base_w = 1.0 / (sigma**2 + EPS)
    base_w = np.clip(base_w, np.percentile(base_w, 10), np.percentile(base_w, 90))
    base_w = base_w / (np.median(base_w) + EPS)
    f_scale = float(np.clip(np.median(sigma), 0.3, 8.0))
    tau = float(np.clip(np.median(sigma) * 1.5, 0.5, 12.0))
    return {
        "bias": bias,
        "sigma": sigma,
        "qlo": qlo,
        "qhi": qhi,
        "base_w": base_w,
        "f_scale": f_scale,
        "tau": tau,
    }


def apply_calibration(d: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    d = sanitize_distances(d)
    bias = np.asarray(calib["bias"], dtype=float)
    qlo = np.asarray(calib["qlo"], dtype=float)
    qhi = np.asarray(calib["qhi"], dtype=float)
    dc = d - bias[:, None]
    dc = np.clip(dc, qlo[:, None], qhi[:, None])
    return dc


def fit_safe_affine_calibration(d_train: np.ndarray, p_train: np.ndarray, p_bs: np.ndarray, base_calib: Dict[str, Any]) -> Dict[str, Any]:
    """Safe-A variant: anchor-wise robust affine calibration with fallback."""
    d0 = sanitize_distances(d_train)
    tr = true_ranges(p_train, p_bs)
    base_dc = apply_calibration(d0, base_calib)
    m = d0.shape[0]
    a_arr = np.ones(m, dtype=float)
    b_arr = np.zeros(m, dtype=float)
    use_aff = np.zeros(m, dtype=bool)
    dc_train = base_dc.copy()
    for i in range(m):
        x = d0[i].reshape(-1, 1)
        y = tr[i]
        try:
            huber = HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=500)
            huber.fit(x, y)
            a = float(huber.coef_[0])
            b = float(huber.intercept_)
        except Exception:
            qx25, qx75 = np.percentile(d0[i], [25, 75])
            qy25, qy75 = np.percentile(y, [25, 75])
            a = float((qy75 - qy25) / (qx75 - qx25 + EPS))
            b = float(np.median(y) - a * np.median(d0[i]))
        a = float(np.clip(a, 0.35, 1.35))
        b = float(np.clip(b, -80.0, 80.0))
        aff = a * d0[i] + b
        aff = aff - np.median(aff - y)
        base_err = np.median(np.abs(base_dc[i] - y))
        aff_err = np.median(np.abs(aff - y))
        if aff_err < 0.99 * base_err:
            use_aff[i] = True
            a_arr[i] = a
            b_arr[i] = b - np.median(a * d0[i] + b - y)
            dc_train[i] = aff
    err = dc_train - tr
    sigma = robust_mad(err, axis=1)
    sigma = np.clip(sigma, 0.25, np.percentile(sigma, 90) * 2.0 + 0.25)
    qlo = np.maximum(np.percentile(dc_train, 0.5, axis=1), 0.05)
    qhi = np.maximum(np.percentile(dc_train, 99.5, axis=1), qlo + 1.0)
    base_w = 1.0 / (sigma**2 + EPS)
    base_w = np.clip(base_w, np.percentile(base_w, 10), np.percentile(base_w, 90))
    base_w = base_w / (np.median(base_w) + EPS)
    f_scale = float(np.clip(np.median(sigma), 0.3, 8.0))
    tau = float(np.clip(np.median(sigma) * 1.5, 0.5, 12.0))
    return {
        "base_calib": base_calib,
        "safe_a": a_arr,
        "safe_b": b_arr,
        "safe_use_affine": use_aff,
        "bias": np.zeros(m, dtype=float),
        "sigma": sigma,
        "qlo": qlo,
        "qhi": qhi,
        "base_w": base_w,
        "f_scale": f_scale,
        "tau": tau,
    }


def apply_safe_affine_calibration(d: np.ndarray, safe_calib: Dict[str, Any]) -> np.ndarray:
    d0 = sanitize_distances(d)
    base_dc = apply_calibration(d0, safe_calib["base_calib"])
    a = np.asarray(safe_calib["safe_a"], dtype=float)
    b = np.asarray(safe_calib["safe_b"], dtype=float)
    use = np.asarray(safe_calib["safe_use_affine"], dtype=bool)
    aff_dc = a[:, None] * d0 + b[:, None]
    dc = base_dc.copy()
    dc[use] = aff_dc[use]
    qlo = np.asarray(safe_calib["qlo"], dtype=float)
    qhi = np.asarray(safe_calib["qhi"], dtype=float)
    return np.clip(dc, qlo[:, None], qhi[:, None])


def pairwise_diffs(x: np.ndarray) -> np.ndarray:
    m = x.shape[0]
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            out.append(x[i] - x[j])
    return np.vstack(out) if out else np.empty((0, x.shape[1]))


def weighted_centroid_features(d_corr: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    n = d_corr.shape[1]
    feats = []
    for power in (1.0, 2.0):
        w = 1.0 / np.maximum(d_corr, 0.2) ** power
        xy = (p_bs @ w) / (np.sum(w, axis=0, keepdims=True) + EPS)
        feats.append(xy.T)
    nearest = np.argmin(d_corr, axis=0)
    feats.append(p_bs[:, nearest].T)
    return np.hstack(feats)


def make_features(d: np.ndarray, p_bs: np.ndarray, calib: Dict[str, Any] | None = None) -> np.ndarray:
    d0 = sanitize_distances(d)
    if calib is not None:
        dc = apply_calibration(d0, calib)
        sigma = np.asarray(calib["sigma"], dtype=float)
        z = dc / (sigma[:, None] + EPS)
    else:
        dc = d0
        med = np.median(dc, axis=1)
        sc = robust_mad(dc - med[:, None], axis=1)
        sc = np.clip(sc, 0.25, None)
        z = (dc - med[:, None]) / sc[:, None]

    ranks = np.argsort(np.argsort(dc, axis=0), axis=0).astype(float) / max(1, dc.shape[0] - 1)
    sorted_d = np.sort(dc, axis=0)
    stats = np.vstack([
        np.min(dc, axis=0),
        np.max(dc, axis=0),
        np.mean(dc, axis=0),
        np.median(dc, axis=0),
        np.std(dc, axis=0),
        np.percentile(dc, 10, axis=0),
        np.percentile(dc, 25, axis=0),
        np.percentile(dc, 75, axis=0),
        np.percentile(dc, 90, axis=0),
        np.max(dc, axis=0) - np.min(dc, axis=0),
    ])
    wc = weighted_centroid_features(dc, p_bs).T
    diffs = pairwise_diffs(z)
    logd = np.log1p(dc)
    invd = 1.0 / np.maximum(dc, 0.2)
    feature = np.vstack([dc, logd, invd, ranks, sorted_d[:8], diffs, stats, wc]).T
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    return feature


def fit_feature_state(d_train: np.ndarray, p_train: np.ndarray, p_bs: np.ndarray, calib: Dict[str, Any]) -> Dict[str, Any]:
    x = make_features(d_train, p_bs, calib)
    scaler = RobustScaler(quantile_range=(10, 90))
    xs = scaler.fit_transform(x)
    y = p_train.T
    et = ExtraTreesRegressor(
        n_estimators=450,
        max_depth=14,
        min_samples_leaf=3,
        max_features=0.65,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    et.fit(xs, y)
    ridge = Ridge(alpha=8.0)
    ridge.fit(xs, y)
    hgb = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=170,
            learning_rate=0.045,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=RANDOM_STATE,
        )
    )
    hgb.fit(xs, y)
    return {"scaler": scaler, "x_train": xs, "y_train": y, "et": et, "ridge": ridge, "hgb": hgb}


def transform_features(d: np.ndarray, p_bs: np.ndarray, calib: Dict[str, Any], fs: Dict[str, Any]) -> np.ndarray:
    return fs["scaler"].transform(make_features(d, p_bs, calib))


def knn_predict_from_scaled(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int, power: float = 2.0) -> np.ndarray:
    k = int(min(k, x_train.shape[0]))
    out = np.zeros((x_test.shape[0], 2), dtype=float)
    chunk = 128
    for s in range(0, x_test.shape[0], chunk):
        xt = x_test[s:s + chunk]
        d2 = np.sum((xt[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        local_d = np.take_along_axis(d2, idx, axis=1)
        local_d = np.sqrt(np.maximum(local_d, 0.0))
        w = 1.0 / (local_d + 1e-6) ** power
        w = w / (np.sum(w, axis=1, keepdims=True) + EPS)
        out[s:s + len(xt)] = np.sum(y_train[idx] * w[:, :, None], axis=1)
    return out


def local_ridge_knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, k: int = 40, alpha: float = 2.0) -> np.ndarray:
    k = int(min(k, x_train.shape[0]))
    pred = np.zeros((x_test.shape[0], 2), dtype=float)
    for i, xt in enumerate(x_test):
        d2 = np.sum((x_train - xt[None, :]) ** 2, axis=1)
        idx = np.argpartition(d2, kth=k - 1)[:k]
        xloc = x_train[idx]
        yloc = y_train[idx]
        model = Ridge(alpha=alpha)
        model.fit(xloc - xt[None, :], yloc)
        pred[i] = model.predict(np.zeros((1, x_train.shape[1])))[0]
    return pred


def bounds_from_bs(p_bs: np.ndarray, margin: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.min(p_bs, axis=1) - margin
    hi = np.max(p_bs, axis=1) + margin
    if np.any(hi <= lo):
        hi = lo + 1.0
    return lo, hi


def robust_nls_single(
    d_col: np.ndarray,
    p_bs: np.ndarray,
    weights: np.ndarray,
    x0: np.ndarray,
    f_scale: float,
    prior: np.ndarray | None = None,
    prior_scale: float = 3.0,
    max_nfev: int = 80,
) -> np.ndarray:
    lo, hi = bounds_from_bs(p_bs, margin=10.0)
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)
    w_sqrt = np.sqrt(np.maximum(weights, 1e-6))

    def fun(x: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(p_bs.T - x[None, :], axis=1) - d_col
        rr = w_sqrt * r
        if prior is not None:
            pr = (x - prior) / max(prior_scale, 0.2)
            rr = np.concatenate([rr, pr])
        return rr

    try:
        res = least_squares(fun, x0=x0, bounds=(lo, hi), loss="soft_l1", f_scale=f_scale, max_nfev=max_nfev)
        return res.x.astype(float)
    except Exception:
        return x0.astype(float)


def predict_map_nls(
    d: np.ndarray,
    p_bs: np.ndarray,
    calib: Dict[str, Any],
    priors: np.ndarray,
    mode: str = "map",
) -> np.ndarray:
    dc = apply_calibration(d, calib)
    base_w = np.asarray(calib["base_w"], dtype=float)
    f_scale = float(calib["f_scale"])
    tau = float(calib["tau"])
    n = dc.shape[1]
    pred = np.zeros((n, 2), dtype=float)
    for j in range(n):
        d_col = dc[:, j]
        x0 = priors[j]
        x1 = robust_nls_single(d_col, p_bs, base_w, x0=x0, f_scale=f_scale, prior=x0, prior_scale=3.0)
        residual = np.linalg.norm(p_bs.T - x1[None, :], axis=1) - d_col
        positive_nlos = np.maximum(d_col - np.linalg.norm(p_bs.T - x1[None, :], axis=1), 0.0)
        w2 = base_w / (1.0 + (np.abs(residual) / max(tau, 0.5)) ** 2)
        w2 = w2 / (1.0 + (positive_nlos / max(0.75 * tau, 0.4)) ** 2)
        if mode == "trimmed":
            cutoff = np.percentile(np.abs(residual), 70)
            excess = np.maximum(np.abs(residual) - cutoff, 0.0)
            w2 = w2 / (1.0 + (excess / max(tau, 0.5)) ** 2)
            w2 = np.maximum(w2, 0.03 * np.median(base_w))
            prior_scale = 2.2
        elif mode == "calib":
            prior_scale = 4.0
        else:
            prior_scale = 3.0
        pred[j] = robust_nls_single(d_col, p_bs, w2, x0=x1, f_scale=f_scale, prior=x0, prior_scale=prior_scale)
    return pred


def fit_range_calibrators(d_train: np.ndarray, p_train: np.ndarray, p_bs: np.ndarray, calib: Dict[str, Any]) -> List[Any]:
    d0 = sanitize_distances(d_train)
    tr = true_ranges(p_train, p_bs)
    target_err = tr - d0
    models: List[Any] = []
    m = d0.shape[0]
    global_stats = np.vstack([
        np.mean(d0, axis=0), np.std(d0, axis=0), np.min(d0, axis=0), np.max(d0, axis=0)
    ]).T
    for i in range(m):
        xi = np.column_stack([d0[i], np.log1p(d0[i]), global_stats])
        yi = target_err[i]
        model = ExtraTreesRegressor(
            n_estimators=120,
            max_depth=7,
            min_samples_leaf=6,
            max_features=0.9,
            bootstrap=True,
            random_state=RANDOM_STATE + i,
            n_jobs=-1,
        )
        model.fit(xi, yi)
        models.append(model)
    return models


def apply_range_calibrators(d: np.ndarray, models: List[Any], calib: Dict[str, Any]) -> np.ndarray:
    d0 = sanitize_distances(d)
    global_stats = np.vstack([
        np.mean(d0, axis=0), np.std(d0, axis=0), np.min(d0, axis=0), np.max(d0, axis=0)
    ]).T
    out = d0.copy()
    sigma = np.asarray(calib["sigma"], dtype=float)
    for i, model in enumerate(models):
        xi = np.column_stack([d0[i], np.log1p(d0[i]), global_stats])
        delta = model.predict(xi)
        delta = np.clip(delta, -3.0 * sigma[i], 3.0 * sigma[i])
        out[i] = d0[i] + delta
    qlo = np.asarray(calib["qlo"], dtype=float)
    qhi = np.asarray(calib["qhi"], dtype=float)
    out = np.clip(out, qlo[:, None], qhi[:, None])
    return out


def _triplet_candidate(p_bs_t: np.ndarray, d_col: np.ndarray, triplet: Tuple[int, int, int]) -> np.ndarray | None:
    """Algebraic 2D trilateration from three range circles.

    V12A 최적화: np.linalg.cond (SVD 내부 호출) 대신 2x2 행렬식(determinant)으로
    기하학적 퇴화 여부를 직접 판정한다. SVD가 없으므로 약 3~5배 빠르다.
    """
    i, j, k = triplet
    a0 = p_bs_t[i]
    d0 = d_col[i]
    # 두 anchor에 대한 선형 방정식 계수 계산
    row0 = 2.0 * (a0 - p_bs_t[j])
    rhs0 = (d_col[j] ** 2 - d0 ** 2
            - float(np.dot(p_bs_t[j], p_bs_t[j]))
            + float(np.dot(a0, a0)))
    row1 = 2.0 * (a0 - p_bs_t[k])
    rhs1 = (d_col[k] ** 2 - d0 ** 2
            - float(np.dot(p_bs_t[k], p_bs_t[k]))
            + float(np.dot(a0, a0)))
    # 2×2 행렬식으로 퇴화 여부 판정 (cond 대신)
    det = row0[0] * row1[1] - row0[1] * row1[0]
    if abs(det) < 1e-3:
        return None
    inv_det = 1.0 / det
    x = np.array([
        (row1[1] * rhs0 - row0[1] * rhs1) * inv_det,
        (row0[0] * rhs1 - row1[0] * rhs0) * inv_det,
    ])
    if not np.all(np.isfinite(x)):
        return None
    return x


def predict_ransac_triplet_multilateration(
    d: np.ndarray,
    p_bs: np.ndarray,
    calib: Dict[str, Any],
    use_already_calibrated: bool = False,
) -> np.ndarray:
    """RANSAC-like robust multilateration expert."""
    dc = sanitize_distances(d) if use_already_calibrated else apply_calibration(d, calib)
    base_w = np.asarray(calib["base_w"], dtype=float)
    sigma = np.asarray(calib["sigma"], dtype=float)
    tau = float(calib["tau"])
    f_scale = float(calib["f_scale"])
    pbst = p_bs.T
    n = dc.shape[1]
    out = np.zeros((n, 2), dtype=float)
    lo, hi = bounds_from_bs(p_bs, margin=10.0)
    reliable_order = np.argsort(sigma)[:8]
    for col in range(n):
        d_col = dc[:, col]
        near_order = np.argsort(d_col)[:12]
        anchors = np.unique(np.concatenate([near_order, reliable_order]))
        if len(anchors) < 3:
            anchors = np.arange(dc.shape[0])
        best: list[tuple[float, np.ndarray]] = []
        L = len(anchors)
        for aa in range(L - 2):
            for bb in range(aa + 1, L - 1):
                for cc in range(bb + 1, L):
                    tri = (int(anchors[aa]), int(anchors[bb]), int(anchors[cc]))
                    x = _triplet_candidate(pbst, d_col, tri)
                    if x is None:
                        continue
                    if np.any(x < lo - 5.0) or np.any(x > hi + 5.0):
                        continue
                    residual = np.linalg.norm(pbst - x[None, :], axis=1) - d_col
                    absr = np.abs(residual)
                    positive_nlos = np.maximum(d_col - np.linalg.norm(pbst - x[None, :], axis=1), 0.0)
                    score = (
                        np.median(absr)
                        + 0.25 * (np.percentile(absr, 75) - np.percentile(absr, 25))
                        + 0.15 * np.average(np.minimum(absr, 4.0 * tau), weights=np.maximum(base_w, 1e-6))
                        + 0.15 * np.median(positive_nlos)
                    )
                    best.append((float(score), x))
        if not best:
            w = 1.0 / np.maximum(d_col, 0.2) ** 2
            x0 = (p_bs @ w) / (np.sum(w) + EPS)
        else:
            best.sort(key=lambda z: z[0])
            top = best[: min(8, len(best))]
            scores = np.asarray([t[0] for t in top])
            cand = np.vstack([t[1] for t in top])
            ww = 1.0 / (scores + 1e-3) ** 2
            ww = ww / (np.sum(ww) + EPS)
            x0 = np.sum(cand * ww[:, None], axis=0)
        x0 = np.clip(x0, lo, hi)
        r0 = np.linalg.norm(pbst - x0[None, :], axis=1) - d_col
        pos0 = np.maximum(d_col - np.linalg.norm(pbst - x0[None, :], axis=1), 0.0)
        w2 = base_w / (1.0 + (np.abs(r0) / max(tau, 0.5)) ** 2)
        w2 = w2 / (1.0 + (pos0 / max(0.75 * tau, 0.4)) ** 2)
        w2 = np.maximum(w2, 0.03 * np.median(base_w))
        out[col] = robust_nls_single(d_col, p_bs, w2, x0=x0, f_scale=f_scale, prior=x0, prior_scale=6.0, max_nfev=60)
    return out


def predict_residual_reranked_fingerprint(
    d: np.ndarray,
    p_bs: np.ndarray,
    calib: Dict[str, Any],
    fs: Dict[str, Any],
    k: int = 20,
) -> np.ndarray:
    """Direction D: top-k fingerprint candidate reranking by physical residual."""
    xs = transform_features(d, p_bs, calib, fs)
    xtr = fs["x_train"]
    ytr = fs["y_train"]
    dc = apply_calibration(d, calib)
    base_w = np.asarray(calib["base_w"], dtype=float)
    tau = float(calib["tau"])
    n = xs.shape[0]
    k = int(min(k, xtr.shape[0]))
    out = np.zeros((n, 2), dtype=float)
    for j in range(n):
        fd2 = np.sum((xtr - xs[j][None, :]) ** 2, axis=1)
        idx = np.argpartition(fd2, kth=k - 1)[:k]
        cand_xy = ytr[idx]
        scores = []
        for c, xy in enumerate(cand_xy):
            calc = np.linalg.norm(p_bs.T - xy[None, :], axis=1)
            r = calc - dc[:, j]
            absr = np.abs(r)
            pos = np.maximum(dc[:, j] - calc, 0.0)
            score = (
                np.median(absr)
                + 0.20 * (np.percentile(absr, 75) - np.percentile(absr, 25))
                + 0.15 * np.average(np.minimum(absr, 4.0 * tau), weights=np.maximum(base_w, 1e-6))
                + 0.10 * np.median(pos)
                + 0.02 * np.sqrt(fd2[idx[c]])
            )
            scores.append(float(score))
        scores = np.asarray(scores)
        order = np.argsort(scores)[: min(5, len(scores))]
        ww = 1.0 / (scores[order] + 1e-3) ** 2
        ww = ww / (np.sum(ww) + EPS)
        out[j] = np.sum(cand_xy[order] * ww[:, None], axis=0)
    return out


def fit_expert_state(d_train: np.ndarray, p_train: np.ndarray, p_bs: np.ndarray) -> Dict[str, Any]:
    calib = fit_calibration(d_train, p_train, p_bs)
    fs = fit_feature_state(d_train, p_train, p_bs, calib)
    range_models = fit_range_calibrators(d_train, p_train, p_bs, calib)
    safe_affine_calib = fit_safe_affine_calibration(d_train, p_train, p_bs, calib)
    return {"calib": calib, "feature_state": fs, "range_models": range_models, "safe_affine_calib": safe_affine_calib, "p_bs": p_bs}


def predict_experts(d: np.ndarray, p_bs: np.ndarray, state: Dict[str, Any]) -> np.ndarray:
    calib = state["calib"]
    fs = state["feature_state"]
    xs = transform_features(d, p_bs, calib, fs)
    xtr = fs["x_train"]
    ytr = fs["y_train"]

    pred_knn7 = knn_predict_from_scaled(xtr, ytr, xs, k=7, power=2.0)
    pred_knn21 = knn_predict_from_scaled(xtr, ytr, xs, k=21, power=2.0)
    pred_local = local_ridge_knn_predict(xtr, ytr, xs, k=40, alpha=2.0)
    pred_et = fs["et"].predict(xs)
    pred_hgb = fs["hgb"].predict(xs)

    prior = 0.45 * pred_knn21 + 0.35 * pred_local + 0.20 * pred_et
    pred_map = predict_map_nls(d, p_bs, calib, prior, mode="map")
    pred_trim = predict_map_nls(d, p_bs, calib, prior, mode="trimmed")

    d_cal = apply_range_calibrators(d, state["range_models"], calib)
    calib_cal = dict(calib)
    calib_cal["bias"] = np.zeros_like(np.asarray(calib["bias"]))
    pred_calib = predict_map_nls(d_cal, p_bs, calib_cal, prior, mode="calib")

    pred_ransac = predict_ransac_triplet_multilateration(d, p_bs, calib, use_already_calibrated=False)
    pred_ransac_cal = predict_ransac_triplet_multilateration(d_cal, p_bs, calib_cal, use_already_calibrated=True)
    pred_rerank = predict_residual_reranked_fingerprint(d, p_bs, calib, fs, k=20)

    safe_aff = state.get("safe_affine_calib")
    d_safe = apply_safe_affine_calibration(d, safe_aff)
    safe_for_solver = dict(safe_aff)
    safe_for_solver["bias"] = np.zeros_like(np.asarray(safe_for_solver["bias"]))
    pred_safe_map = predict_map_nls(d_safe, p_bs, safe_for_solver, prior, mode="map")
    pred_safe_trim = predict_map_nls(d_safe, p_bs, safe_for_solver, prior, mode="trimmed")

    preds = np.stack([
        pred_knn7,
        pred_knn21,
        pred_local,
        pred_et,
        pred_hgb,
        pred_map,
        pred_trim,
        pred_calib,
        pred_ransac,
        pred_ransac_cal,
        pred_rerank,
        pred_safe_map,
        pred_safe_trim,
    ], axis=0)
    return preds


def residual_stats_for_pred(d: np.ndarray, p_bs: np.ndarray, xy: np.ndarray, calib: Dict[str, Any]) -> np.ndarray:
    dc = apply_calibration(d, calib)
    calc = np.linalg.norm(xy[:, None, :] - p_bs.T[None, :, :], axis=2)
    r = calc - dc.T
    absr = np.abs(r)
    stats = np.column_stack([
        np.mean(r, axis=1),
        np.median(r, axis=1),
        np.mean(absr, axis=1),
        np.max(absr, axis=1),
        np.percentile(absr, 75, axis=1) - np.percentile(absr, 25, axis=1),
        robust_mad(absr, axis=1),
    ])
    return np.nan_to_num(stats)


def geometry_features(p_bs: np.ndarray, xy: np.ndarray) -> np.ndarray:
    out = np.zeros((xy.shape[0], 4), dtype=float)
    for i, x in enumerate(xy):
        v = x[None, :] - p_bs.T
        dist = np.linalg.norm(v, axis=1) + 1e-6
        h = v / dist[:, None]
        g = h.T @ h
        try:
            cond = np.linalg.cond(g)
            gdop = np.sqrt(np.trace(np.linalg.pinv(g)))
        except Exception:
            cond, gdop = 1e6, 1e3
        out[i, 0] = np.log1p(np.clip(cond, 0, 1e6))
        out[i, 1] = np.log1p(np.clip(gdop, 0, 1e6))
        out[i, 2] = np.min(dist)
        out[i, 3] = np.max(dist) - np.min(dist)
    return out


def make_gating_features(d: np.ndarray, p_bs: np.ndarray, state: Dict[str, Any], preds: np.ndarray) -> np.ndarray:
    calib = state["calib"]
    dc = apply_calibration(d, calib)
    dist_stats = np.vstack([
        np.min(dc, axis=0), np.max(dc, axis=0), np.mean(dc, axis=0), np.median(dc, axis=0),
        np.std(dc, axis=0), np.percentile(dc, 90, axis=0) - np.percentile(dc, 10, axis=0),
    ]).T
    center = np.mean(preds, axis=0)
    e, n, _ = preds.shape
    mean_pred = np.mean(preds, axis=0)
    d_to_mean = np.linalg.norm(preds - mean_pred[None, :, :], axis=2).T
    disagree = np.column_stack([
        np.mean(d_to_mean, axis=1),
        np.max(d_to_mean, axis=1),
        np.std(d_to_mean, axis=1),
        np.linalg.norm(preds[0] - preds[5], axis=1),
        np.linalg.norm(preds[2] - preds[3], axis=1),
        np.linalg.norm(preds[5] - preds[6], axis=1),
    ])
    res_parts = []
    for idx in (0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12):
        if idx < preds.shape[0]:
            res_parts.append(residual_stats_for_pred(d, p_bs, preds[idx], calib))
    res_feat = np.hstack(res_parts)
    geom = geometry_features(p_bs, center)
    lo, hi = bounds_from_bs(p_bs, margin=8.0)
    outside = np.column_stack([
        center[:, 0] < lo[0], center[:, 0] > hi[0], center[:, 1] < lo[1], center[:, 1] > hi[1]
    ]).astype(float)
    gate = np.hstack([dist_stats, disagree, res_feat, geom, outside])
    gate = np.nan_to_num(gate, nan=0.0, posinf=0.0, neginf=0.0)
    return gate


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = np.linalg.norm(y_pred - y_true, axis=1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
        "<=1m": float(np.mean(err <= 1.0) * 100),
        "<=2m": float(np.mean(err <= 2.0) * 100),
        "<=3m": float(np.mean(err <= 3.0) * 100),
    }


def format_markdown_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    cols = ["mean", "median", "rmse", "p90", "p95", "max", "<=1m", "<=2m", "<=3m"]
    lines = []
    lines.append("| model | " + " | ".join(cols) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(cols)) + "|")
    for name, s in rows:
        vals = []
        for c in cols:
            if c.startswith("<="):
                vals.append(f"{s[c]:.1f}%")
            else:
                vals.append(f"{s[c]:.4f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def softmax_negative(err_pred: np.ndarray, temperature: float) -> np.ndarray:
    z = -err_pred / max(temperature, 1e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    w = np.exp(z)
    w = w / (np.sum(w, axis=1, keepdims=True) + EPS)
    return w


def combine_with_weights(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(preds.transpose(1, 0, 2) * weights[:, :, None], axis=1)


def make_super_features(
    preds: np.ndarray,
    gate_x: np.ndarray,
    weights: np.ndarray | None = None,
    pred_err: np.ndarray | None = None,
) -> np.ndarray:
    e, n, _ = preds.shape
    pred_flat = preds.transpose(1, 0, 2).reshape(n, e * 2)
    mean_pred = np.mean(preds, axis=0)
    med_pred = np.median(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    d_to_mean = np.linalg.norm(preds - mean_pred[None, :, :], axis=2).T
    disagree = np.column_stack([
        np.mean(d_to_mean, axis=1),
        np.max(d_to_mean, axis=1),
        np.std(d_to_mean, axis=1),
        np.percentile(d_to_mean, 75, axis=1) - np.percentile(d_to_mean, 25, axis=1),
    ])
    parts = [pred_flat, mean_pred, med_pred, std_pred, disagree]
    if weights is not None:
        parts.append(weights)
        parts.append(combine_with_weights(preds, weights))
    if pred_err is not None:
        parts.append(pred_err)
        parts.append(np.column_stack([np.min(pred_err, axis=1), np.mean(pred_err, axis=1), np.std(pred_err, axis=1)]))
    parts.append(gate_x)
    x = np.hstack(parts)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# V12A: Safe Residual Memory helper
# ──────────────────────────────────────────────────────────────────────────────

def _apply_residual_memory(sx: np.ndarray, slm: Dict[str, Any]) -> np.ndarray:
    """V12A safe residual memory correction.

    Computes a kNN-weighted average of OOF training residuals in scaled
    super-feature space and returns alpha * correction.

    참고: memory_x는 학습 시 super_scaler.transform(super_x) 결과이며,
    memory_residual은 HGB meta correction 직전 단계의 OOF 잔차이다.
    따라서 현재 예측이 OOF 시점과 유사한 error 패턴을 가지면 correction이
    예측을 true position 방향으로 당기는 효과를 낸다. Alpha=0.3으로 보수적.
    """
    alpha  = float(slm["alpha"])   # 0.3
    k      = int(slm["k"])          # 15
    power  = float(slm["power"])    # 2.0
    mem_x  = np.asarray(slm["memory_x"],        dtype=float)  # (700, D)
    mem_r  = np.asarray(slm["memory_residual"], dtype=float)  # (700, 2)

    n = sx.shape[0]
    correction = np.zeros((n, 2), dtype=float)

    # 거리 계산: ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # 대용량 브로드캐스트 없이 행렬 곱으로 처리
    sx_sq  = np.sum(sx    ** 2, axis=1)   # (n,)
    mx_sq  = np.sum(mem_x ** 2, axis=1)   # (700,)

    chunk = 256
    for s in range(0, n, chunk):
        e   = min(s + chunk, n)
        xt  = sx[s:e]                                                    # (c, D)
        d2  = sx_sq[s:e, None] + mx_sq[None, :] - 2.0 * (xt @ mem_x.T) # (c, 700)
        d2  = np.maximum(d2, 0.0)                                        # 수치 안전
        c   = e - s
        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]             # (c, k)
        local_d2 = np.take_along_axis(d2, idx, axis=1)                  # (c, k)
        w = 1.0 / (np.sqrt(local_d2) + 1e-6) ** power                  # (c, k)
        w = w / (np.sum(w, axis=1, keepdims=True) + EPS)
        # mem_r[idx]: (c, k, 2)
        correction[s:e] = np.einsum("ij,ijk->ik", w, mem_r[idx])
    return alpha * correction


# ──────────────────────────────────────────────────────────────────────────────
# V3/V4/V12A OOF meta-superlearner (학습 코드)
# ──────────────────────────────────────────────────────────────────────────────

def fit_residual_superlearner_cv(
    preds_oof: np.ndarray,
    gate_x: np.ndarray,
    gate_weights: np.ndarray,
    pred_err: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
) -> Dict[str, Any]:
    x = make_super_features(preds_oof, gate_x, gate_weights, pred_err)
    scaler = RobustScaler(quantile_range=(10, 90))
    xs = scaler.fit_transform(x)
    target_res = y_true - base_pred
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 777)

    res_oof = np.zeros_like(target_res)
    for tr, va in kf.split(xs):
        model = ExtraTreesRegressor(
            n_estimators=650,
            max_depth=7,
            min_samples_leaf=14,
            max_features=0.70,
            bootstrap=True,
            random_state=RANDOM_STATE + 778,
            n_jobs=-1,
        )
        model.fit(xs[tr], target_res[tr])
        res_oof[va] = model.predict(xs[va])

    best_alpha = 0.0
    best_mean = metric_summary(y_true, base_pred)["mean"]
    best_pred = base_pred.copy()
    for alpha in [0.0, 0.10, 0.15, 0.20, 0.27, 0.35, 0.50, 0.65, 0.80, 1.00]:
        cand = base_pred + alpha * res_oof
        m = metric_summary(y_true, cand)["mean"]
        if m < best_mean:
            best_mean = m
            best_alpha = alpha
            best_pred = cand

    final_model = ExtraTreesRegressor(
        n_estimators=750,
        max_depth=7,
        min_samples_leaf=12,
        max_features=0.70,
        bootstrap=True,
        random_state=RANDOM_STATE + 779,
        n_jobs=-1,
    )
    final_model.fit(xs, target_res)
    return {
        "scaler": scaler,
        "model": final_model,
        "alpha": float(best_alpha),
        "residual_oof": res_oof,
        "corrected_oof": best_pred,
        "base_mean": float(metric_summary(y_true, base_pred)["mean"]),
        "corrected_mean": float(best_mean),
    }


def _fit_predict_meta_model(model_name: str, xs_tr: np.ndarray, target_tr: np.ndarray, xs_va: np.ndarray) -> np.ndarray:
    if model_name == "ridge_residual" or model_name == "ridge_direct":
        model = Ridge(alpha=18.0)
    elif model_name == "et_residual_deep" or model_name == "et_direct_deep":
        model = ExtraTreesRegressor(
            n_estimators=700, max_depth=10, min_samples_leaf=8,
            max_features=0.70, bootstrap=True, random_state=RANDOM_STATE + 900, n_jobs=-1,
        )
    elif model_name == "et_residual_safe" or model_name == "et_direct_safe":
        model = ExtraTreesRegressor(
            n_estimators=850, max_depth=7, min_samples_leaf=16,
            max_features=0.70, bootstrap=True, random_state=RANDOM_STATE + 901, n_jobs=-1,
        )
    elif model_name == "hgb_residual" or model_name == "hgb_direct":
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=140, learning_rate=0.035, max_leaf_nodes=12,
                min_samples_leaf=18, l2_regularization=0.20, random_state=RANDOM_STATE + 902,
            )
        )
    else:
        raise ValueError(model_name)
    model.fit(xs_tr, target_tr)
    return model.predict(xs_va)


def _make_final_meta_model(model_name: str) -> Any:
    if model_name in ("ridge_residual", "ridge_direct"):
        return Ridge(alpha=18.0)
    if model_name in ("et_residual_deep", "et_direct_deep"):
        return ExtraTreesRegressor(
            n_estimators=850, max_depth=10, min_samples_leaf=8,
            max_features=0.70, bootstrap=True, random_state=RANDOM_STATE + 910, n_jobs=-1,
        )
    if model_name in ("et_residual_safe", "et_direct_safe"):
        return ExtraTreesRegressor(
            n_estimators=950, max_depth=7, min_samples_leaf=14,
            max_features=0.70, bootstrap=True, random_state=RANDOM_STATE + 911, n_jobs=-1,
        )
    if model_name in ("hgb_residual", "hgb_direct"):
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=170, learning_rate=0.030, max_leaf_nodes=12,
                min_samples_leaf=18, l2_regularization=0.20, random_state=RANDOM_STATE + 912,
            )
        )
    raise ValueError(model_name)


def fit_meta_superlearner_v4_cv(
    preds_oof: np.ndarray,
    gate_x: np.ndarray,
    gate_weights: np.ndarray,
    pred_err: np.ndarray,
    y_true: np.ndarray,
    base_pred: np.ndarray,
) -> Dict[str, Any]:
    x = make_super_features(preds_oof, gate_x, gate_weights, pred_err)
    scaler = RobustScaler(quantile_range=(10, 90))
    xs = scaler.fit_transform(x)
    target_res = y_true - base_pred
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 888)

    candidate_names = [
        "ridge_residual", "et_residual_safe", "et_residual_deep", "hgb_residual",
        "ridge_direct", "et_direct_safe", "et_direct_deep", "hgb_direct",
    ]
    alpha_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.27, 0.35, 0.50, 0.65, 0.80, 1.00]
    candidate_rows: List[Tuple[str, Dict[str, float]]] = []

    best_name = "base_gated"
    best_mode = "none"
    best_alpha = 0.0
    best_mean = metric_summary(y_true, base_pred)["mean"]
    best_pred = base_pred.copy()
    best_raw = np.zeros_like(base_pred)

    for name in candidate_names:
        is_residual = "residual" in name
        target = target_res if is_residual else y_true
        raw_oof = np.zeros_like(y_true)
        for tr, va in kf.split(xs):
            raw_oof[va] = _fit_predict_meta_model(name, xs[tr], target[tr], xs[va])
        for alpha in alpha_grid:
            if is_residual:
                cand = base_pred + alpha * raw_oof
            else:
                cand = (1.0 - alpha) * base_pred + alpha * raw_oof
            s = metric_summary(y_true, cand)
            candidate_rows.append((f"V4 candidate {name} alpha={alpha:.2f}", s))
            score = s["mean"] + 1e-4 * s["p95"]
            best_score = best_mean + 1e-4 * metric_summary(y_true, best_pred)["p95"]
            if score < best_score:
                best_name = name
                best_mode = "residual" if is_residual else "direct"
                best_alpha = float(alpha)
                best_mean = s["mean"]
                best_pred = cand
                best_raw = raw_oof

    if best_mode == "none":
        final_model = None
        final_target = None
    else:
        final_model = _make_final_meta_model(best_name)
        final_target = target_res if best_mode == "residual" else y_true
        final_model.fit(xs, final_target)

    return {
        "scaler": scaler,
        "model": final_model,
        "name": best_name,
        "mode": best_mode,
        "alpha": float(best_alpha),
        "raw_oof": best_raw,
        "corrected_oof": best_pred,
        "base_mean": float(metric_summary(y_true, base_pred)["mean"]),
        "corrected_mean": float(best_mean),
        "candidate_rows": candidate_rows,
    }


def fit_gating_cv(gate_x: np.ndarray, expert_errors: np.ndarray, preds_oof: np.ndarray, y_true: np.ndarray) -> Tuple[Any, RobustScaler, float, np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(10, 90))
    gx = scaler.fit_transform(gate_x)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 100)
    pred_err_oof = np.zeros_like(expert_errors)
    for tr, va in kf.split(gx):
        model = ExtraTreesRegressor(
            n_estimators=350, max_depth=9, min_samples_leaf=10,
            max_features=0.85, bootstrap=True, random_state=RANDOM_STATE + 200, n_jobs=-1,
        )
        model.fit(gx[tr], expert_errors[tr])
        pred_err_oof[va] = np.maximum(model.predict(gx[va]), 0.05)

    best_t = 2.0
    best_mean = float("inf")
    for t in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]:
        w = softmax_negative(pred_err_oof, t)
        pred = combine_with_weights(preds_oof, w)
        m = metric_summary(y_true, pred)["mean"]
        if m < best_mean:
            best_mean = m
            best_t = t
    final_model = ExtraTreesRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=8,
        max_features=0.85, bootstrap=True, random_state=RANDOM_STATE + 300, n_jobs=-1,
    )
    final_model.fit(gx, expert_errors)
    return final_model, scaler, best_t, pred_err_oof, softmax_negative(pred_err_oof, best_t)


# ──────────────────────────────────────────────────────────────────────────────
# V12A: Safe Residual Memory 학습 (fit_safe_residual_memory)
# ──────────────────────────────────────────────────────────────────────────────

def fit_safe_residual_memory(
    preds_oof: np.ndarray,
    gate_x: np.ndarray,
    gate_weights_oof: np.ndarray,
    pred_err_oof: np.ndarray,
    super_oof: Dict[str, Any],
    y_true: np.ndarray,
    base_pred_after_meta: np.ndarray,
) -> Dict[str, Any]:
    """V12A: OOF 잔차를 meta feature space에 저장하는 safe residual memory.

    학습 시 OOF 예측(base_pred_after_meta)의 잔차를 meta feature 공간(153차원)에
    매핑하여 저장한다. 추론 시에는 kNN으로 가장 유사한 학습 샘플의 OOF 잔차를
    보간하여 conservative correction을 적용한다.

    alpha_grid와 k_grid를 OOF 내부 검색으로 선택하므로 alpha=0.0이면 사용 안 함.
    """
    super_x = make_super_features(preds_oof, gate_x, gate_weights_oof, pred_err_oof)
    sx = super_oof["scaler"].transform(super_x)

    memory_x = sx.astype(np.float32)
    # meta correction 직전 잔차를 메모리에 저장
    oof_residual = y_true - base_pred_after_meta          # true - out_after_hgb (OOF)
    memory_residual = oof_residual.astype(np.float32)

    best_alpha = 0.0
    best_k = 15
    best_mean = metric_summary(y_true, base_pred_after_meta)["mean"]
    best_pred = base_pred_after_meta.copy()

    # LOO-style: 각 샘플에 대해 자기 자신을 제외한 kNN 보간
    # 700×700 거리 행렬을 한 번만 계산
    sx_sq  = np.sum(sx ** 2, axis=1)
    d2_all = sx_sq[:, None] + sx_sq[None, :] - 2.0 * (sx @ sx.T)  # (700, 700)
    d2_all = np.maximum(d2_all, 0.0)
    np.fill_diagonal(d2_all, np.inf)   # self-exclusion

    for k_try in [10, 15, 20, 25]:
        idx_k = np.argpartition(d2_all, kth=k_try - 1, axis=1)[:, :k_try]
        local_d2 = np.take_along_axis(d2_all, idx_k, axis=1)
        w = 1.0 / (np.sqrt(local_d2) + 1e-6) ** 2.0
        w = w / (np.sum(w, axis=1, keepdims=True) + EPS)
        knn_res = np.einsum("ij,ijk->ik", w, oof_residual[idx_k])  # (700, 2)

        for a_try in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            cand = base_pred_after_meta + a_try * knn_res
            m = metric_summary(y_true, cand)["mean"]
            if m < best_mean:
                best_mean = m
                best_alpha = a_try
                best_k = k_try
                best_pred = cand

    # best kNN correction on all training data (without LOO, for final storage)
    idx_best = np.argpartition(d2_all, kth=best_k - 1, axis=1)[:, :best_k]
    local_d2_best = np.take_along_axis(d2_all, idx_best, axis=1)
    w_best = 1.0 / (np.sqrt(local_d2_best) + 1e-6) ** 2.0
    w_best = w_best / (np.sum(w_best, axis=1, keepdims=True) + EPS)
    knn_correction = np.einsum("ij,ijk->ik", w_best, oof_residual[idx_best])

    return {
        "alpha": float(best_alpha),
        "k": int(best_k),
        "power": 2.0,
        "oof_residual": oof_residual,
        "corrected_oof": best_pred,
        "corrected_mean": float(best_mean),
        "memory_x": memory_x,              # (700, D) float32
        "memory_residual": memory_residual,  # (700, 2) float32
    }


def run_oof(d_hat: np.ndarray, p: np.ndarray, p_bs: np.ndarray) -> Dict[str, Any]:
    n = d_hat.shape[1]
    y = p.T
    e = len(EXPERT_NAMES)
    preds_oof = np.zeros((e, n, 2), dtype=float)
    gate_oof = None
    fold_ids = np.zeros(n, dtype=int)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"[train-V12A_SafeResidualMemory] OOF expert generation: N={n}, experts={e}")
    for fold, (tr, va) in enumerate(kf.split(np.arange(n)), start=1):
        t0 = time.time()
        state = fit_expert_state(d_hat[:, tr], p[:, tr], p_bs)
        preds = predict_experts(d_hat[:, va], p_bs, state)
        preds_oof[:, va, :] = preds
        gf = make_gating_features(d_hat[:, va], p_bs, state, preds)
        if gate_oof is None:
            gate_oof = np.zeros((n, gf.shape[1]), dtype=float)
        gate_oof[va] = gf
        fold_ids[va] = fold
        print(f"[train-V12A_SafeResidualMemory] fold {fold}/{CV_FOLDS} done in {time.time() - t0:.1f}s")

    expert_errors = np.linalg.norm(preds_oof - y[None, :, :], axis=2).T
    gate_model, gate_scaler, temp, pred_err_oof, gate_weights_oof = fit_gating_cv(gate_oof, expert_errors, preds_oof, y)
    gated_oof = combine_with_weights(preds_oof, gate_weights_oof)

    super_oof = fit_meta_superlearner_v4_cv(
        preds_oof=preds_oof,
        gate_x=gate_oof,
        gate_weights=gate_weights_oof,
        pred_err=pred_err_oof,
        y_true=y,
        base_pred=gated_oof,
    )
    super_pred = super_oof["corrected_oof"]

    # V12A: Safe Residual Memory correction on top of V4 meta
    safe_mem = fit_safe_residual_memory(
        preds_oof=preds_oof,
        gate_x=gate_oof,
        gate_weights_oof=gate_weights_oof,
        pred_err_oof=pred_err_oof,
        super_oof=super_oof,
        y_true=y,
        base_pred_after_meta=super_pred,
    )
    final_pred = safe_mem["corrected_oof"]

    fixed_inv = 1.0 / (np.mean(expert_errors, axis=0) + EPS)
    fixed_inv = fixed_inv / fixed_inv.sum()
    fixed_pred = np.sum(preds_oof * fixed_inv[:, None, None], axis=0)
    oracle_idx = np.argmin(expert_errors, axis=1)
    oracle_pred = preds_oof[oracle_idx, np.arange(n), :]

    rows: List[Tuple[str, Dict[str, float]]] = []
    for i, name in enumerate(EXPERT_NAMES):
        rows.append((name, metric_summary(y, preds_oof[i])))
    rows.append(("Fixed inverse-OOF-error averaging", metric_summary(y, fixed_pred)))
    rows.append(("Dynamic expected-error gated MoE", metric_summary(y, gated_oof)))
    rows.append(("OOF final V12A_SafeResidualMemory", metric_summary(y, final_pred)))
    rows.append(("Oracle best expert upper bound", metric_summary(y, oracle_pred)))

    selection = np.argmin(pred_err_oof, axis=1)
    return {
        "preds_oof": preds_oof,
        "gate_oof": gate_oof,
        "expert_errors": expert_errors,
        "gate_model": gate_model,
        "gate_scaler": gate_scaler,
        "temperature": temp,
        "pred_err_oof": pred_err_oof,
        "gate_weights_oof": gate_weights_oof,
        "gated_oof": gated_oof,
        "super_oof": super_oof,
        "safe_mem": safe_mem,
        "fixed_weights": fixed_inv,
        "rows": rows,
        "selection": selection,
        "fold_ids": fold_ids,
    }


def fit_final_model(d_hat: np.ndarray, p: np.ndarray, p_bs: np.ndarray, oof_result: Dict[str, Any]) -> Dict[str, Any]:
    print("[train-V12A_SafeResidualMemory] fitting final full-data expert state")
    expert_state = fit_expert_state(d_hat, p, p_bs)
    model = {
        "version": "v12a_safe_residual_memory",   # 버전명 일치
        "expert_names": EXPERT_NAMES,
        "expert_state": expert_state,
        "gate_model": oof_result["gate_model"],
        "gate_scaler": oof_result["gate_scaler"],
        "temperature": oof_result["temperature"],
        "super_model": oof_result["super_oof"]["model"],
        "super_scaler": oof_result["super_oof"]["scaler"],
        "super_alpha": oof_result["super_oof"]["alpha"],
        "super_mode": oof_result["super_oof"]["mode"],
        "super_name": oof_result["super_oof"]["name"],
        "super_local_memory": oof_result["safe_mem"],  # V12A residual memory
        "fixed_weights": oof_result["fixed_weights"],
        "oof_table_rows": oof_result["rows"],
        "random_state": RANDOM_STATE,
    }
    return model


# ──────────────────────────────────────────────────────────────────────────────
# predict_with_model (V12A 대응 버전)
# ──────────────────────────────────────────────────────────────────────────────

def predict_with_model(d_hat: np.ndarray, p_bs: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    """V12A predict: gated MoE → HGB meta → safe residual memory correction."""
    state = model["expert_state"]
    preds = predict_experts(d_hat, p_bs, state)
    gate_x = make_gating_features(d_hat, p_bs, state, preds)
    gx = model["gate_scaler"].transform(gate_x)
    pred_err = np.maximum(model["gate_model"].predict(gx), 0.05)
    weights = softmax_negative(pred_err, float(model["temperature"]))
    out = combine_with_weights(preds, weights)

    # V4 meta correction (HGB direct 또는 residual)
    if "super_model" in model and model.get("super_model") is not None and "super_alpha" in model:
        super_x = make_super_features(preds, gate_x, weights, pred_err)
        sx = model["super_scaler"].transform(super_x)
        meta_pred = model["super_model"].predict(sx)
        alpha = float(model["super_alpha"])
        mode = model.get("super_mode", "residual")
        if mode == "residual":
            out = out + alpha * meta_pred
        elif mode == "direct":
            out = (1.0 - alpha) * out + alpha * meta_pred

        # V12A: Safe Residual Memory correction
        # super_local_memory가 있고 alpha > 0이면 kNN 잔차 보간을 추가 적용한다.
        # sx는 이미 계산된 scaled super features이므로 재사용.
        slm = model.get("super_local_memory")
        if slm is not None and float(slm.get("alpha", 0.0)) > 0.0:
            out = out + _apply_residual_memory(sx, slm)

    return out.T


def save_oof_report(oof: Dict[str, Any], out_path: str = "oof_results.md") -> str:
    lines: List[str] = []
    lines.append("# OOF Results V12A_SafeResidualMemory")
    lines.append("")
    lines.append(_simple_markdown_table(oof["rows"]))
    lines.append("")
    lines.append("# Gating Predicted-Best Statistics")
    lines.append("")
    lines.append("| expert | predicted-best count | ratio |")
    lines.append("|---|---:|---:|")
    sel = oof["selection"]
    total = len(sel)
    for i, name in enumerate(EXPERT_NAMES):
        c = int(np.sum(sel == i))
        lines.append(f"| {name} | {c} | {100*c/total:.1f}% |")
    lines.append("")
    lines.append("# Average Soft Gating Weight")
    lines.append("")
    lines.append("| expert | avg soft weight |")
    lines.append("|---|---:|")
    avg_w = np.mean(oof.get("gate_weights_oof"), axis=0)
    for i, name in enumerate(EXPERT_NAMES):
        lines.append(f"| {name} | {avg_w[i]:.4f} |")
    lines.append("")
    lines.append(f"Temperature selected by OOF gating-CV grid: {oof['temperature']}")
    if "super_oof" in oof:
        lines.append(f"V12A_SafeResidualMemory selected meta learner: {oof['super_oof'].get('name', 'base_gated')} ({oof['super_oof'].get('mode', 'none')})")
        lines.append(f"V12A_SafeResidualMemory meta alpha selected by OOF meta-CV: {oof['super_oof']['alpha']}")
        lines.append(f"Base gated mean before V12A_SafeResidualMemory meta correction: {oof['super_oof']['base_mean']:.4f}")
        lines.append(f"V12A_SafeResidualMemory meta-corrected mean after correction: {oof['super_oof']['corrected_mean']:.4f}")
    if "safe_mem" in oof:
        sm = oof["safe_mem"]
        lines.append(f"V12A safe residual memory alpha: {sm['alpha']}")
        lines.append(f"V12A safe residual memory k: {sm['k']}")
    text = "\n".join(lines)
    Path(out_path).write_text(text, encoding="utf-8")
    return text


def _simple_markdown_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    cols = ["mean", "median", "rmse", "p90", "p95", "max", "<=1m", "<=2m", "<=3m"]
    lines = ["| model | " + " | ".join(cols) + " |",
             "|---|" + "|".join(["---:"] * len(cols)) + "|"]
    for name, s in rows:
        vals = []
        for c in cols:
            if c.startswith("<="):
                vals.append(f"{s[c]:.1f}%")
            else:
                vals.append(f"{s[c]:.4f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    p_bs, d_hat, p, meta = load_mat_data("DH_FR1.mat")
    print(f"[data] {meta}")
    print(f"[data] p_bs={p_bs.shape}, d_hat={d_hat.shape}, p={None if p is None else p.shape}")
    if p is None:
        raise RuntimeError("train.py requires ground-truth p. Put the labeled training .mat file in this folder.")
    oof = run_oof(d_hat, p, p_bs)
    model = fit_final_model(d_hat, p, p_bs, oof)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    report = save_oof_report(oof)
    print("\n" + report)
    print(f"\n[saved] {MODEL_PATH}")
    print("[saved] oof_results.md")


if __name__ == "__main__":
    main()
