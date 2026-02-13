from kvcache_smi.monitor.collector import KVCacheMonitor


def test_monitor_capture_and_summary():
    monitor = KVCacheMonitor(mode="balanced")
    monitor.capture({0: bytearray(128), 1: bytearray(256)}, num_cached_tokens=16)
    monitor.capture({0: bytearray(128), 1: bytearray(512)}, num_cached_tokens=32)

    summary = monitor.summary()
    assert summary["samples"] == 2
    assert summary["peak_memory_bytes"] == 640
    assert summary["growth_rate_bytes_per_sample"] == 256
