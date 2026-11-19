# Stage 0 smoke check: the body of tests/test_gemm_bf16xfp32.py, run over the same
# parameter grid without pytest. Use pytest when it is installed; this script is the
# fallback for an environment without it.
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.path.realpath(list(ROOT.glob("build/lib.*/"))[0]))
sys.path.insert(0, str(ROOT / "tests"))

import hpc  # noqa: E402
import torch  # noqa: E402
from utils import allclose  # noqa: E402

M = [1, 6, 16, 32, 48, 64, 96, 144, 208, 304, 416, 624, 832, 1024, 2048, 4096, 12303, 32768]
N = [192, 512, 1024, 2048]
K = [4096]


def case(m, n, k, use_fp32_output, use_split_flag):
    torch.manual_seed(10086)
    torch.cuda.manual_seed(10086)
    x = torch.randn((m, k), dtype=torch.float, device="cuda").to(torch.bfloat16)
    w = torch.randn((n, k), dtype=torch.float, device="cuda")
    scale = 1 / 256
    w_high = w.to(torch.bfloat16)
    w_low = ((w - w_high.float()) / scale).to(torch.bfloat16)
    split_flag = hpc.get_gemm_bf16xfp32_workspace(n) if use_split_flag else None
    gt = torch.matmul(x.float(), w.t())
    my = hpc.gemm_bf16xfp32(x, w_high, w_low, scale, use_fp32_output, True, split_flag)
    if use_split_flag:
        assert (split_flag == 0).all(), "split_flag not reset"
    assert allclose(gt, my.float(), rtol=0.08, atol=0.01), "mismatch"


def main():
    print("loaded arch:", hpc.__loaded_arch__, "device:", torch.cuda.get_device_name())
    passed, failed = 0, []
    for p in itertools.product(M, N, K, [True, False], [True, False]):
        try:
            case(*p)
            passed += 1
        except AssertionError as e:
            failed.append((p, str(e)))
    print(f"{passed} passed, {len(failed)} failed")
    for p, e in failed:
        print("FAIL", p, e)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
