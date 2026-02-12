from __future__ import annotations

import time
from dataclasses import asdict

from kvcache_smi.core.schemas import KVCacheMetrics


class KVCacheMonitor:
    def __init__(self, mode: str = "balanced") -> None:
        if mode not in {"minimal", "balanced", "diagnostic"}:
            raise ValueError("mode must be minimal|balanced|diagnostic")
        self.mode = mode
        self._peak = 0
        self._records: list[KVCacheMetrics] = []

    @staticmethod
    def tensor_nbytes(tensor) -> int:
        if tensor is None:
            return 0
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            return int(tensor.numel() * tensor.element_size())
        if hasattr(tensor, "nbytes"):
            return int(tensor.nbytes)
        if hasattr(tensor, "__len__"):
            return int(len(tensor))
        return 0

    def capture(self, per_layer_cache: dict[int, object], num_cached_tokens: int = 0) -> KVCacheMetrics:
        layer_bytes = {layer: self.tensor_nbytes(t) for layer, t in per_layer_cache.items()}
        total = sum(layer_bytes.values())
        self._peak = max(self._peak, total)
        metric = KVCacheMetrics(
            timestamp=time.time(),
            total_memory_bytes=total,
            per_layer_memory=layer_bytes,
            num_cached_tokens=num_cached_tokens,
            peak_memory_bytes=self._peak,
            mode=self.mode,
        )
        self._records.append(metric)
        return metric

    def history(self) -> list[dict]:
        return [asdict(item) for item in self._records]

    def summary(self) -> dict:
        if not self._records:
            return {"samples": 0, "peak_memory_bytes": 0, "growth_rate_bytes_per_sample": 0.0}
        samples = len(self._records)
        first = self._records[0].total_memory_bytes
        last = self._records[-1].total_memory_bytes
        growth = (last - first) / max(samples - 1, 1)
        return {
            "samples": samples,
            "peak_memory_bytes": self._peak,
            "growth_rate_bytes_per_sample": growth,
        }
