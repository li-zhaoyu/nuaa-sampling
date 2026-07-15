"""重构算法（对应 计算光采样.md §5）。

- somp:        多陪集 MMV 联合支撑恢复（SOMP/CTF），多频带场景基线。
- omp:         单测量向量 OMP。
- fista_mmv:   ℓ2,1 group-lasso 的 FISTA（谱域稀疏，含可选字典）。
- null_jammer: 干扰子空间 U_jam 置零（§4 噪声折叠抑制的线性代数实现）。
"""
from __future__ import annotations

from typing import Optional, Tuple, Sequence
import numpy as np


def _normalize_cols(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(A, axis=0) + 1e-12
    return A / norms, norms


def somp(A: np.ndarray, Y: np.ndarray, k: int,
         force_support: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simultaneous OMP（MMV）。A:(R,N0) Y:(R,J)。返回 (support[k], Xhat[N0,J])。

    force_support: 预先纳入的列（如已知干扰子带），不计入 k 预算。
    """
    R, N0 = A.shape
    J = Y.shape[1] if Y.ndim == 2 else 1
    Y = Y.reshape(R, J)
    An, _ = _normalize_cols(A)
    support = list(force_support) if force_support is not None else []
    resid = Y.copy()
    if support:                                   # 先对强制支撑做一次 LS 去除
        As = A[:, support]
        coef = np.linalg.pinv(As) @ Y
        resid = Y - As @ coef
    for _ in range(k):
        corr = np.sum(np.abs(An.conj().T @ resid), axis=1)   # 行能量
        corr[support] = -np.inf
        j = int(np.argmax(corr))
        support.append(j)
        As = A[:, support]
        coef = np.linalg.pinv(As) @ Y
        resid = Y - As @ coef
    Xhat = np.zeros((N0, J), dtype=np.complex128)
    As = A[:, support]
    Xhat[support, :] = np.linalg.pinv(As) @ Y
    return np.array(support, dtype=int), Xhat


def omp(A: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    support, Xhat = somp(A, y.reshape(-1, 1), k)
    return support, Xhat[:, 0]


def somp_prior(A: np.ndarray, Y: np.ndarray, k: int,
               prior: Optional[np.ndarray] = None, beta: float = 2.0,
               force_support: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """先验加权 SOMP：相关度乘以 (1+beta*pi_tilde)，pi_tilde 为归一化先验幅度。

    用于 NUAA-MU 展开输出后的支撑精修：高 SNR 下保留贪心 SOMP 的 LS 最优性，
    同时利用学习式谱估计作为软偏置（对应 计算光采样.md 重构软偏置注入）。
    """
    R, N0 = A.shape
    J = Y.shape[1] if Y.ndim == 2 else 1
    Y = Y.reshape(R, J)
    An, _ = _normalize_cols(A)
    support = list(force_support) if force_support is not None else []
    resid = Y.copy()
    if support:
        As = A[:, support]
        coef = np.linalg.pinv(As) @ Y
        resid = Y - As @ coef
    w = None
    if prior is not None:
        p = np.asarray(prior, dtype=np.float64).reshape(-1)
        if p.size == N0:
            w = 1.0 + beta * (p / (np.max(p) + 1e-12))
    for _ in range(k):
        corr = np.sum(np.abs(An.conj().T @ resid), axis=1)
        if w is not None:
            corr = corr * w
        corr[support] = -np.inf
        j = int(np.argmax(corr))
        support.append(j)
        As = A[:, support]
        coef = np.linalg.pinv(As) @ Y
        resid = Y - As @ coef
    Xhat = np.zeros((N0, J), dtype=np.complex128)
    As = A[:, support]
    Xhat[support, :] = np.linalg.pinv(As) @ Y
    return np.array(support, dtype=int), Xhat


def refine_with_prior(A: np.ndarray, y: np.ndarray, k: int,
                      prior: np.ndarray, beta: float = 2.0) -> np.ndarray:
    """NUAA-MU 后处理：先验加权 SOMP + LS，返回复谱向量 (nC,)。"""
    y2 = y.reshape(-1, 1)
    _, Xhat = somp_prior(A, y2, k, prior=prior, beta=beta)
    return Xhat[:, 0]


def _soft_group(X: np.ndarray, thr: float) -> np.ndarray:
    """ℓ2,1 行组软阈值：按行 ℓ2 范数收缩。"""
    row_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    scale = np.maximum(0.0, 1.0 - thr / row_norm)
    return X * scale


def fista_mmv(A: np.ndarray, Y: np.ndarray, lam: float, n_iter: int = 200,
              Psi_fwd=None, Psi_inv=None) -> np.ndarray:
    """FISTA 求 min_X 0.5||A X - Y||_F^2 + lam ||Psi X||_{2,1}。

    谱域稀疏时 Psi=单位（默认）。返回 Xhat (N0,J)。
    """
    R, N0 = A.shape
    Y = Y.reshape(R, -1)
    L = np.linalg.norm(A, 2) ** 2 + 1e-9          # Lipschitz
    step = 1.0 / L
    X = np.zeros((N0, Y.shape[1]), dtype=np.complex128)
    Z = X.copy()
    t = 1.0
    AH = A.conj().T
    for _ in range(n_iter):
        grad = AH @ (A @ Z - Y)
        Xn = Z - step * grad
        if Psi_fwd is not None:
            Xn = Psi_inv(_soft_group(Psi_fwd(Xn), lam * step))
        else:
            Xn = _soft_group(Xn, lam * step)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Z = Xn + ((t - 1) / tn) * (Xn - X)
        X, t = Xn, tn
    return X


def null_jammer(A: np.ndarray, Y: np.ndarray, jam_cols: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """U_jam 子空间置零：把 Y、A 投影到干扰列张成子空间的正交补。

    返回投影后的 (A_perp, Y_perp)，用于在抑制强干扰后恢复弱信号。
    """
    if len(jam_cols) == 0:
        return A, Y
    U = A[:, list(jam_cols)]
    # 正交投影 P_perp = I - U (U^H U)^{-1} U^H
    G = U.conj().T @ U + 1e-9 * np.eye(U.shape[1])
    Pinv = np.linalg.inv(G)
    def proj_perp(M):
        return M - U @ (Pinv @ (U.conj().T @ M))
    return proj_perp(A), proj_perp(Y.reshape(A.shape[0], -1))


def detect_and_null_jammer(A: np.ndarray, Y: np.ndarray, ratio_thr: float = 3.0):
    """检测主导干扰（最强相关分量显著高于次强）并做 U_jam 置零。

    返回 (A_use, Y_use, jam_col 或 None)。无主导干扰时原样返回（条件置零，避免误删真信号）。
    """
    An, _ = _normalize_cols(A)
    Y2 = Y.reshape(A.shape[0], -1)
    corr = np.sum(np.abs(An.conj().T @ Y2), axis=1)
    order = np.argsort(corr)[::-1]
    if corr[order[1]] > 1e-12 and corr[order[0]] > ratio_thr * corr[order[1]]:
        jam = int(order[0])
        A_p, Y_p = null_jammer(A, Y, [jam])
        return A_p, Y_p, jam
    return A, Y2, None


def reconstruct_multicoset(A: np.ndarray, Y: np.ndarray, n_bands: int,
                           jam_cols: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """多陪集重构主入口：可选先抑制干扰子空间，再 SOMP 恢复弱信号支撑。"""
    if jam_cols:
        A_p, Y_p = null_jammer(A, Y, jam_cols)
        support, Xhat = somp(A_p, Y_p, n_bands)
    else:
        support, Xhat = somp(A, Y, n_bands)
    return support, Xhat
