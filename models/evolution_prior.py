"""演化先验预测器（对应 计算光采样.md §两级先验 / §8 E4）。

把场景级演化律（chirp-rate 递变 / 跳频中心漂移 / 调制调度）离线预训练进网络参数 θ_class：
给定历史窗的主带中心轨迹（归一化），预测下一窗中心 → 形成预测性支撑先验。

- 容量由 hidden / n_layers 控制（E4 容量扫描）。
- CPU 友好：单层/双层 GRU，纯 PyTorch，无 CUDA 依赖。
- 与随机对照对照：随机序列无可外推规律，任何容量都退化到“持续性”水平。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

import torch
import torch.nn as nn


class EvolutionPredictor(nn.Module):
    """GRU 序列预测器：输入归一化中心轨迹 (B,T,1) -> 预测下一步增量 (B,T,1)。

    预测下一窗中心的“增量” Δc_r = c_{r+1}-c_r（而非绝对值），更利于学线性/二阶漂移。
    """

    def __init__(self, hidden: int = 32, n_layers: int = 1, in_dim: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.gru = nn.GRU(in_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # 增量标准化尺度（slots）：在预训练中据训练集统计设定，使回归目标 O(1)
        self.register_buffer("inc_scale", torch.tensor(1.0))

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x:(B,T,in_dim) 归一化中心(+一阶差分)。返回 (B,T,1) 预测的下一步增量。"""
        h, _ = self.gru(x)
        return self.head(h)


def _featurize(centers: np.ndarray, N0: int, inc_scale: float) -> np.ndarray:
    """(B,T) 中心索引 -> (B,T,2) 特征：归一化中心 + 标准化一阶差分。

    一阶差分按 inc_scale（典型每窗漂移幅度，slots）标准化为 O(1)，
    使「线性外推=直接复用上一窗增量」对网络是恒等映射、易学。
    """
    c = centers.astype(np.float64) / float(N0)
    d = np.diff(centers.astype(np.float64), axis=1, prepend=centers[:, :1].astype(np.float64))
    d = d / float(inc_scale)
    return np.stack([c, d], axis=-1)


@dataclass
class PretrainResult:
    model: EvolutionPredictor
    n_params: int
    train_loss: float
    val_pred_mae_slots: float       # 验证集逐步中心预测 MAE（slots）
    persistence_mae_slots: float    # “持续性”基线（预测=上一窗）MAE


def pretrain_evolution(train_centers: np.ndarray, val_centers: np.ndarray, N0: int,
                       hidden: int = 32, n_layers: int = 1, epochs: int = 60,
                       lr: float = 5e-3, seed: int = 0, device: str = "cpu") -> PretrainResult:
    """在某场景类的演化序列上离线预训练中心预测器。

    train/val_centers: (Ntr, W) / (Nva, W) 主带中心索引轨迹。
    """
    torch.manual_seed(seed)
    model = EvolutionPredictor(hidden=hidden, n_layers=n_layers).to(device)
    # 增量标准化尺度：训练集逐窗增量的 std（slots）
    inc_slots = np.diff(train_centers.astype(np.float64), axis=1)
    inc_scale = float(np.std(inc_slots)) + 1e-6
    model.inc_scale = torch.tensor(inc_scale)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    Xtr = torch.tensor(_featurize(train_centers, N0, inc_scale), dtype=torch.float32, device=device)
    # 目标：下一步增量（teacher forcing），标准化到 O(1)
    Ytr = torch.tensor(inc_slots / inc_scale, dtype=torch.float32, device=device).unsqueeze(-1)

    model.train()
    last = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xtr)[:, :-1, :]          # 对齐 (B,W-1,1)
        loss = lossf(pred, Ytr)
        loss.backward()
        opt.step()
        last = float(loss.item())

    val_mae, pers_mae = _eval_centers(model, val_centers, N0, device)
    return PretrainResult(model=model, n_params=model.n_params, train_loss=last,
                          val_pred_mae_slots=val_mae, persistence_mae_slots=pers_mae)


@torch.no_grad()
def _eval_centers(model: EvolutionPredictor, centers: np.ndarray, N0: int, device: str):
    model.eval()
    sc = float(model.inc_scale)
    X = torch.tensor(_featurize(centers, N0, sc), dtype=torch.float32, device=device)
    pred_inc = model(X).cpu().numpy()[:, :, 0] * sc       # (B,W) 预测增量(slots)
    pred_next = centers.astype(np.float64) + pred_inc      # c_r + Δ̂ -> ĉ_{r+1}
    true_next = centers.astype(np.float64)
    # 逐步：用窗 r 预测窗 r+1
    err = np.abs(pred_next[:, :-1] - true_next[:, 1:])
    pers = np.abs(true_next[:, :-1] - true_next[:, 1:])     # 持续性：ĉ_{r+1}=c_r
    return float(np.mean(err)), float(np.mean(pers))


@torch.no_grad()
def predict_next_centers(model: EvolutionPredictor, centers_hist: np.ndarray, N0: int,
                         device: str = "cpu") -> np.ndarray:
    """给定历史中心 (B,T) -> 预测每步的下一窗中心 (B,T)（ĉ_{r+1}）。"""
    model.eval()
    sc = float(model.inc_scale)
    X = torch.tensor(_featurize(centers_hist, N0, sc), dtype=torch.float32, device=device)
    pred_inc = model(X).cpu().numpy()[:, :, 0] * sc
    return centers_hist.astype(np.float64) + pred_inc
