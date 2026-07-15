"""一键 sanity：映射单元测试 + 极小 E1 + 极小 NUAA-MU 训练。约 ~30s（CPU）。

  python experiments/sanity.py
"""
from __future__ import annotations

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable

sys.path.insert(0, ROOT)
from nuaa.cpu_env import configure

configure(verbose=True)


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main():
    rc = 0
    # 1) §3 映射单元测试
    rc |= sh([PY, "tests/test_mapping.py"])
    # 2) E1 极小
    rc |= sh([PY, "experiments/exp_e1_multiband.py", "--n0", "1000", "--trials", "8",
              "--steps", "1", "2", "3", "--tag", "sanity"])
    # 3) NUAA-MU 极小训练（无极端干扰，验证可训练）
    rc |= sh([PY, "experiments/train_nuaa_mu.py", "--quick", "--no-jammer"])
    print("\n=== SANITY", "PASS ===" if rc == 0 else f"FAIL (rc={rc}) ===")
    sys.exit(rc)


if __name__ == "__main__":
    main()
