"""评估度量（对应 计算光采样.md §8，按场景选用）。"""
from __future__ import annotations

from typing import Sequence, Tuple, Union
import numpy as np


def nmse(x_hat: np.ndarray, x: np.ndarray) -> float:
    """归一化均方误差（复/实通用），含全局复标度对齐（相位/幅度模糊）。"""
    x = x.reshape(-1)
    x_hat = x_hat.reshape(-1)
    denom = np.vdot(x_hat, x_hat)
    if abs(denom) < 1e-20:
        alpha = 0.0
    else:
        alpha = np.vdot(x_hat, x) / denom          # 最优复标度
    err = np.linalg.norm(x - alpha * x_hat) ** 2
    return float(err / (np.linalg.norm(x) ** 2 + 1e-20))


def nmse_db(x_hat: np.ndarray, x: np.ndarray) -> float:
    return 10.0 * np.log10(nmse(x_hat, x) + 1e-20)


def support_f1(est: Sequence[int], true: Sequence[int]) -> Tuple[float, float, float]:
    """返回 (precision, recall, F1)。"""
    est_s, true_s = set(int(i) for i in est), set(int(i) for i in true)
    if not est_s and not true_s:
        return 1.0, 1.0, 1.0
    tp = len(est_s & true_s)
    prec = tp / (len(est_s) + 1e-12)
    rec = tp / (len(true_s) + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def jammer_col_accuracy(jammer_logits: np.ndarray, jam_idx: int,
                        prob_thr: float = 0.5) -> Tuple[float, float]:
    """干扰列检测：返回 (top1 准确率, 阈值门控准确率)。

    jammer_logits: (nC,) 或 (B, nC)
    jam_idx: 真值干扰在候选集中的局部列索引
    """
    logits = np.asarray(jammer_logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = np.argmax(prob, axis=1)
    jam = int(jam_idx)
    top1 = float(np.mean(pred == jam))
    gated = float(np.mean((pred == jam) & (prob[np.arange(prob.shape[0]), pred] > prob_thr)))
    return top1, gated


def binary_f1_from_logits(logits: np.ndarray, target: np.ndarray,
                          thr: float = 0.5) -> float:
    """逐候选二元 F1（用于 jammer/support 头全向量评估）。"""
    prob = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64).reshape(-1)))
    pred = (prob >= thr).astype(np.int64)
    tgt = (np.asarray(target, dtype=np.float64).reshape(-1) > 0.5).astype(np.int64)
    tp = int(np.sum((pred == 1) & (tgt == 1)))
    fp = int(np.sum((pred == 1) & (tgt == 0)))
    fn = int(np.sum((pred == 0) & (tgt == 1)))
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return float(2 * prec * rec / (prec + rec + 1e-12))


def si_sdr(x_hat: np.ndarray, x: np.ndarray) -> float:
    """Scale-Invariant SDR (dB)，实信号；复信号取实部。"""
    x = np.real(x).reshape(-1)
    x_hat = np.real(x_hat).reshape(-1)
    alpha = np.dot(x_hat, x) / (np.dot(x, x) + 1e-12)
    target = alpha * x
    noise = x_hat - target
    return float(10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def evm(sym_hat: np.ndarray, sym: np.ndarray) -> float:
    """EVM (%)：通信符号误差矢量幅度。"""
    sym = sym.reshape(-1)
    sym_hat = sym_hat.reshape(-1)
    alpha = np.vdot(sym_hat, sym) / (np.vdot(sym_hat, sym_hat) + 1e-12)
    err = np.linalg.norm(sym - alpha * sym_hat)
    return float(100.0 * err / (np.linalg.norm(sym) + 1e-12))


def weak_signal_nmse(X_hat: np.ndarray, X_true: np.ndarray,
                     support: Sequence[int]) -> float:
    """仅在弱信号子带（不含干扰）上评估谱 NMSE。"""
    support = list(support)
    if not support:
        return float("nan")
    return nmse(X_hat[support, :], X_true[support, :])


def hop_edge_f1(est_centers: Sequence[int], true_centers: Sequence[int],
                tol_slots: int = 0) -> float:
    """跳频 hop 边界 F1：在相邻窗边上检测载频是否跳变（边索引 r=1..W-1）。"""
    tc = np.asarray(true_centers, dtype=np.int64).reshape(-1)
    ec = np.asarray(est_centers, dtype=np.int64).reshape(-1)
    W = tc.size
    if W < 2:
        return 1.0
    true_edges, est_edges = [], []
    for r in range(1, W):
        if abs(int(tc[r]) - int(tc[r - 1])) > tol_slots:
            true_edges.append(r)
        if abs(int(ec[r]) - int(ec[r - 1])) > tol_slots:
            est_edges.append(r)
    _, _, f1 = support_f1(est_edges, true_edges)
    return float(f1)


def carrier_rmse_slots(est_centers: Sequence[int],
                       true_centers: Sequence[int]) -> float:
    """载频轨迹 RMSE（频率箱 / slots）。"""
    ec = np.asarray(est_centers, dtype=np.float64).reshape(-1)
    tc = np.asarray(true_centers, dtype=np.float64).reshape(-1)
    if ec.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((ec - tc) ** 2)))
