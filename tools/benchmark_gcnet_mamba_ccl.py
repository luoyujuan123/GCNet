#!/usr/bin/env python3
"""Benchmark Params / FLOPs / Latency for current Mamba+CCL model.

Usage examples:
  python tools/benchmark_gcnet_mamba_ccl.py --segments 8 16 --num-class 51
  python tools/benchmark_gcnet_mamba_ccl.py --segments 8 16 --device cuda:0 --json-out benchmark.json

Notes:
- `counted_gflops` comes from fvcore and may miss custom ops.
- `full_theory_gflops_est` adds:
  1) unsupported elementwise ops captured by torch profiler
  2) analytical FLOPs for `prim::PythonOp.MambaInnerFn`
"""

import argparse
import inspect
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
from torch.profiler import ProfilerActivity, profile
from fvcore.nn import FlopCountAnalysis

# Ensure project root is importable when running from tools/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ops.models import TSN


BINARY_OPS = {"aten::sub", "aten::mul", "aten::add", "aten::add_", "aten::mul_"}
UNARY_OPS = {"aten::sigmoid", "aten::silu", "aten::exp", "aten::neg", "aten::mean"}


@dataclass
class BenchResult:
    segments: int
    params_all: int
    params_trainable: int
    params_m: float
    counted_gflops: float
    extra_elem_gflops: float
    extra_mamba_inner_gflops: float
    full_theory_gflops_est: float
    latency_ms_mean: float
    latency_ms_std: float
    unsupported_ops: Dict[str, int]


def _numel(shape: List[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _mamba_inner_flops(
    mamba_batch: int,
    seq_len: int = 8,
    d_model: int = 6272,
    expand: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
) -> int:
    # Approximate FLOPs of Mamba fast path based on module dimensions.
    d_inner = int(d_model * expand)
    dt_rank = (d_model + 15) // 16

    conv_dw = 2 * mamba_batch * d_inner * seq_len * d_conv
    x_proj = 2 * mamba_batch * seq_len * d_inner * (dt_rank + 2 * d_state)
    dt_proj = 2 * mamba_batch * seq_len * d_inner * dt_rank
    scan_core = (7 * mamba_batch * d_inner * seq_len * d_state) + (2 * mamba_batch * d_inner * seq_len)
    out_proj = 2 * mamba_batch * seq_len * d_inner * d_model
    return conv_dw + x_proj + dt_proj + scan_core + out_proj


def build_model(num_class: int, seg: int, arch: str, dropout: float, img_feature_dim: int, pretrain: str, device: str):
    model = TSN(
        num_class,
        seg,
        "RGB",
        base_model=arch,
        consensus_type="avg",
        dropout=dropout,
        img_feature_dim=img_feature_dim,
        pretrain=pretrain,
    )
    model.to(device)
    model.train(False)
    return model


def _build_forward_inputs(model, x):
    """Build forward inputs compatible with different TSN forward signatures."""
    try:
        sig = inspect.signature(model.forward)
        params = list(sig.parameters.values())
    except (ValueError, TypeError):
        return (x,)

    # Excluding `self`, if forward has an additional required positional arg
    # (usually `target`), provide a dummy target tensor.
    required_after_input = []
    for p in params[1:]:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            required_after_input.append(p)

    if required_after_input:
        dummy_target = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return (x, dummy_target)

    return (x,)


def _forward_once(model, fwd_inputs):
    out = model(*fwd_inputs)
    # Some variants return (loss, logits) when target is provided.
    if isinstance(out, (tuple, list)):
        return out[-1]
    return out


def run_one(
    seg: int,
    num_class: int,
    arch: str,
    dropout: float,
    img_feature_dim: int,
    pretrain: str,
    input_size: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    iters: int,
    device: str,
) -> BenchResult:
    model = build_model(num_class, seg, arch, dropout, img_feature_dim, pretrain, device)

    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    x = torch.randn(batch_size, seg, 15, input_size, input_size, device=device)

    # Capture the true batch entering Mamba, since this code reshapes temporal dims internally.
    mamba_batch = {"v": None}

    def pre_hook(_module, inputs):
        mamba_batch["v"] = int(inputs[0].shape[0])

    fwd_inputs = _build_forward_inputs(model, x)

    hook_target = getattr(model, "bash_model", None)
    if hook_target is None:
        hook_target = getattr(model, "base_model", None)
    hook = None
    if hook_target is not None:
        hook = hook_target.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        _ = _forward_once(model, fwd_inputs)
    if hook is not None:
        hook.remove()

    flops_analyzer = FlopCountAnalysis(model, fwd_inputs)
    counted_flops = flops_analyzer.total()
    unsupported = {str(k): int(v) for k, v in flops_analyzer.unsupported_ops().items()}

    # Warmup for stable latency.
    with torch.no_grad():
        for _ in range(warmup):
            _ = _forward_once(model, fwd_inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    lat_vals = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.time()
            for _ in range(iters):
                _ = _forward_once(model, fwd_inputs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            lat_vals.append((time.time() - t0) * 1000.0 / iters)

    # Profile once and account for unsupported elementwise ops.
    acts = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        acts.append(ProfilerActivity.CUDA)

    with profile(activities=acts, record_shapes=True) as prof:
        with torch.no_grad():
            _ = _forward_once(model, fwd_inputs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    extra_elem_flops = 0
    for item in prof.key_averages(group_by_input_shape=True):
        if not item.input_shapes:
            continue
        shape = item.input_shapes[0]
        if not isinstance(shape, list) or not shape:
            continue
        if item.key in BINARY_OPS or item.key in UNARY_OPS:
            extra_elem_flops += int(item.count) * _numel(shape)

    has_mamba = any("MambaInnerFn" in k for k in unsupported.keys())
    if has_mamba:
        m_batch = mamba_batch["v"] if mamba_batch["v"] is not None else max(1, (batch_size * seg) // 8)
        extra_mamba_flops = _mamba_inner_flops(mamba_batch=m_batch, seq_len=8)
    else:
        extra_mamba_flops = 0

    full_flops_est = counted_flops + extra_elem_flops + extra_mamba_flops

    return BenchResult(
        segments=seg,
        params_all=params_all,
        params_trainable=params_trainable,
        params_m=params_all / 1e6,
        counted_gflops=counted_flops / 1e9,
        extra_elem_gflops=extra_elem_flops / 1e9,
        extra_mamba_inner_gflops=extra_mamba_flops / 1e9,
        full_theory_gflops_est=full_flops_est / 1e9,
        latency_ms_mean=statistics.mean(lat_vals),
        latency_ms_std=statistics.pstdev(lat_vals),
        unsupported_ops=unsupported,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark current Mamba+CCL model")
    parser.add_argument("--segments", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--num-class", type=int, default=51)
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--img-feature-dim", type=int, default=256)
    parser.add_argument("--pretrain", type=str, default="imagenet")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. cuda:0 / cuda:2 / cpu / auto (pick GPU with most free memory)",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    if args.device != "cpu" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")

    if args.device == "auto":
        if not torch.cuda.is_available():
            args.device = "cpu"
        else:
            best_idx = 0
            best_free = -1
            for i in range(torch.cuda.device_count()):
                free_bytes, _ = torch.cuda.mem_get_info(i)
                if free_bytes > best_free:
                    best_free = free_bytes
                    best_idx = i
            args.device = f"cuda:{best_idx}"

    if args.device.startswith("cuda"):
        # Set current CUDA device explicitly to avoid accidental allocation on gpu0.
        if ":" in args.device:
            idx = int(args.device.split(":", 1)[1])
        else:
            idx = 0
        n_gpu = torch.cuda.device_count()
        if idx < 0 or idx >= n_gpu:
            raise RuntimeError(f"Invalid CUDA device index {idx}. Visible GPU count: {n_gpu}")
        torch.cuda.set_device(idx)

    print(f"Using device: {args.device}")

    results = []
    for seg in args.segments:
        res = run_one(
            seg=seg,
            num_class=args.num_class,
            arch=args.arch,
            dropout=args.dropout,
            img_feature_dim=args.img_feature_dim,
            pretrain=args.pretrain,
            input_size=args.input_size,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            iters=args.iters,
            device=args.device,
        )
        results.append(res)

    print("=" * 80)
    print("Mamba+CCL Benchmark Results")
    print("=" * 80)
    for r in results:
        print(f"segments={r.segments}")
        print(f"  params_all               : {r.params_all} ({r.params_m:.6f} M)")
        print(f"  params_trainable         : {r.params_trainable}")
        print(f"  counted_gflops           : {r.counted_gflops:.6f}")
        print(f"  extra_elem_gflops        : {r.extra_elem_gflops:.6f}")
        print(f"  extra_mamba_inner_gflops : {r.extra_mamba_inner_gflops:.6f}")
        print(f"  full_theory_gflops_est   : {r.full_theory_gflops_est:.6f}")
        print(f"  latency_ms_mean/std      : {r.latency_ms_mean:.6f} / {r.latency_ms_std:.6f}")
        print(f"  unsupported_ops          : {r.unsupported_ops}")

    if args.json_out:
        payload = [asdict(r) for r in results]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Saved JSON to {args.json_out}")


if __name__ == "__main__":
    main()
