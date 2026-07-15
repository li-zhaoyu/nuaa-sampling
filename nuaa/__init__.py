"""NUAA: 时域非均匀光采样系统与极稀疏波形重构（论文一）。

模块对应 `计算光采样.md` §1：
- config / cpu_env   系统配置与本地 CPU 线程调优
- layout             τ⇄陪集（前缀和）⇄t_opt/t_sample（§3）
- kernels / strobe   PD/THA 核、strobe 预测、转换窗（§4）
- measurement        Φ、线性化前向算子 A(τ)、逆映射（§3）
- forward            物理前向观测（§4）
- reconstruct        多陪集 SOMP / FISTA（§5）
- policy             NUAA 布局评分与慢扫描（§6）
- streaming          1GHz 事件流、滚动重构与慢速 EDL 控制
- metrics / signals  指标与合成信号（§8）
"""

__all__ = [
    "config",
    "cpu_env",
    "layout",
    "kernels",
    "strobe",
    "measurement",
    "signals",
    "forward",
    "reconstruct",
    "policy",
    "streaming",
    "metrics",
]
