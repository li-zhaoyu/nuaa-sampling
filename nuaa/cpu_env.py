"""CPU 运行环境调优（面向本地 AMD Ryzen / 多核 x86）。

在实验脚本入口调用 `configure()`，统一限制 BLAS/OpenMP/PyTorch 线程数，
避免 24 逻辑核全开导致 FFT 与训练互相争抢、反而变慢。
"""
from __future__ import annotations

import os


_CONFIGURED = False


def configure(threads: int | None = None, *, verbose: bool = False) -> int:
    """设置数值库线程数。默认 threads = max(4, min(12, cpu_count//2))。

    显式传入 ``threads`` 时会覆盖环境变量与 PyTorch 设置，便于训练阶段拉满核数。
    """
    global _CONFIGURED
    n = os.cpu_count() or 8
    if threads is None:
        threads = max(4, min(12, n // 2))
    threads = int(max(1, threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        # Always write: setdefault would freeze the first (often conservative) value.
        os.environ[key] = str(threads)
    try:
        import torch

        torch.set_num_threads(threads)
        if not _CONFIGURED:
            # interop threads can only be set once per process
            torch.set_num_interop_threads(max(1, min(8, threads // 2)))
            _CONFIGURED = True
    except ImportError:
        pass
    if verbose:
        print(f"[cpu_env] threads={threads} (cpu_count={n})")
    return threads
