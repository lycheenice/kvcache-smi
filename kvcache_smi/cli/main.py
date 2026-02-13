from __future__ import annotations

import argparse
import json
import uuid

from kvcache_smi.core.reporter import write_output
from kvcache_smi.core.schemas import AnalysisReport, InferenceConfig, RunMetadata
from kvcache_smi.estimator.calculator import build_recommendations, estimate_kv_cache_memory
from kvcache_smi.estimator.model_config import load_model_config
from kvcache_smi.monitor.collector import KVCacheMonitor


def _cmd_estimate(args: argparse.Namespace) -> int:
    model = load_model_config(args.config)
    inference = InferenceConfig(
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        precision=args.precision,
    )
    estimate = estimate_kv_cache_memory(
        model=model,
        inference=inference,
        memory_budget_bytes=args.memory_budget_bytes,
        overhead_factor=args.overhead_factor,
    )
    report = AnalysisReport(
        metadata=RunMetadata(run_id=str(uuid.uuid4()), device=args.device, backend="cli"),
        metrics={"estimate": estimate},
        summary={
            "model": model.name,
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
        },
        recommendations=build_recommendations(model, inference, estimate),
    )
    print(write_output(report, args.output))
    return 0


def _cmd_monitor(args: argparse.Namespace) -> int:
    monitor = KVCacheMonitor(mode=args.mode)
    layers = json.loads(args.mock_layer_bytes)
    per_layer = {int(k): bytearray(v) for k, v in layers.items()}
    metric = monitor.capture(per_layer, num_cached_tokens=args.num_cached_tokens)
    report = AnalysisReport(
        metadata=RunMetadata(run_id=str(uuid.uuid4()), device=args.device, backend="cli"),
        metrics={"latest": metric, "history": monitor.history()},
        summary=monitor.summary(),
        recommendations=[],
    )
    print(write_output(report, args.output))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvcache-smi")
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser("estimate", help="Estimate KV cache memory.")
    estimate.add_argument("--config", required=True, help="Path to model config JSON.")
    estimate.add_argument("--max-seq-length", type=int, required=True)
    estimate.add_argument("--batch-size", type=int, required=True)
    estimate.add_argument("--precision", default="fp16")
    estimate.add_argument("--memory-budget-bytes", type=int)
    estimate.add_argument("--overhead-factor", type=float, default=1.0)
    estimate.add_argument("--device", default="unknown")
    estimate.add_argument("--output")
    estimate.set_defaults(func=_cmd_estimate)

    monitor = subparsers.add_parser("monitor", help="Collect monitor snapshot (mock in v0.1).")
    monitor.add_argument("--mode", default="balanced", choices=["minimal", "balanced", "diagnostic"])
    monitor.add_argument("--mock-layer-bytes", default='{"0": 1024, "1": 2048}')
    monitor.add_argument("--num-cached-tokens", type=int, default=0)
    monitor.add_argument("--device", default="unknown")
    monitor.add_argument("--output")
    monitor.set_defaults(func=_cmd_monitor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
