#!/usr/bin/env python3
"""
performance.py - System performance monitoring for the Kobe trading system.

Usage:
    python performance.py              # Show current system stats
    python performance.py --live       # Live monitoring mode
    python performance.py --profile    # Profile slow operations
    python performance.py --benchmark  # Run performance benchmarks
    python performance.py --dotenv PATH  # Load environment from .env file
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.env_loader import load_env


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def get_cpu_usage() -> Optional[float]:
    """Get current CPU usage percentage."""
    try:
        import psutil
        return psutil.cpu_percent(interval=1)
    except ImportError:
        # Fallback for Windows without psutil
        try:
            import subprocess
            result = subprocess.run(
                ["wmic", "cpu", "get", "loadpercentage"],
                capture_output=True,
                text=True,
                timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                return float(lines[1].strip())
        except Exception:
            pass
    return None


def get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information."""
    result: Dict[str, Any] = {}

    try:
        import psutil
        mem = psutil.virtual_memory()
        result["total"] = mem.total
        result["available"] = mem.available
        result["used"] = mem.used
        result["percent"] = mem.percent
    except ImportError:
        # Fallback for Windows
        try:
            import subprocess
            cmd = ["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            lines = proc.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 2:
                    free_kb = int(parts[0])
                    total_kb = int(parts[1])
                    result["total"] = total_kb * 1024
                    result["available"] = free_kb * 1024
                    result["used"] = (total_kb - free_kb) * 1024
                    result["percent"] = ((total_kb - free_kb) / total_kb) * 100
        except Exception:
            pass

    # Process-specific memory
    try:
        import psutil
        proc = psutil.Process()
        result["process_rss"] = proc.memory_info().rss
        result["process_vms"] = proc.memory_info().vms
    except Exception:
        pass

    return result


def get_disk_usage() -> Dict[str, Any]:
    """Get disk usage for the project directory."""
    result: Dict[str, Any] = {}

    try:
        import shutil
        total, used, free = shutil.disk_usage(ROOT)
        result["total"] = total
        result["used"] = used
        result["free"] = free
        result["percent"] = (used / total) * 100
    except Exception:
        pass

    # Calculate project directory size
    try:
        project_size = 0
        for path in ROOT.rglob("*"):
            if path.is_file():
                try:
                    project_size += path.stat().st_size
                except OSError:
                    pass
        result["project_size"] = project_size
    except Exception:
        pass

    return result


def measure_api_response_time(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 10
) -> Tuple[Optional[float], Optional[int], Optional[str]]:
    """Measure API response time."""
    try:
        import requests
        start = time.perf_counter()
        response = requests.get(url, headers=headers, timeout=timeout)
        elapsed = time.perf_counter() - start
        return elapsed, response.status_code, None
    except Exception as e:
        return None, None, str(e)


def benchmark_polygon_api() -> Dict[str, Any]:
    """Benchmark Polygon API response time."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        return {"error": "POLYGON_API_KEY not set"}

    url = f"https://api.polygon.io/v1/marketstatus/now?apiKey={api_key}"
    elapsed, status, error = measure_api_response_time(url)

    if error:
        return {"error": error}

    return {
        "endpoint": "marketstatus/now",
        "response_time_ms": round(elapsed * 1000, 2) if elapsed else None,
        "status_code": status,
    }


def benchmark_alpaca_api() -> Dict[str, Any]:
    """Benchmark Alpaca API response time."""
    api_key = os.getenv("ALPACA_API_KEY_ID", "")
    api_secret = os.getenv("ALPACA_API_SECRET_KEY", "")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not api_secret:
        return {"error": "Alpaca credentials not set"}

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    url = f"{base_url}/v2/account"
    elapsed, status, error = measure_api_response_time(url, headers)

    if error:
        return {"error": error}

    return {
        "endpoint": "/v2/account",
        "response_time_ms": round(elapsed * 1000, 2) if elapsed else None,
        "status_code": status,
    }


def profile_operation(
    name: str,
    func: Callable,
    *args: Any,
    iterations: int = 1,
    **kwargs: Any
) -> Dict[str, Any]:
    """Profile a function execution."""
    gc.collect()

    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func(*args, **kwargs)
        except Exception as e:
            return {"name": name, "error": str(e)}
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "name": name,
        "iterations": iterations,
        "total_time_ms": round(sum(times) * 1000, 2),
        "avg_time_ms": round((sum(times) / len(times)) * 1000, 2),
        "min_time_ms": round(min(times) * 1000, 2),
        "max_time_ms": round(max(times) * 1000, 2),
    }


def show_system_stats() -> None:
    """Display current system statistics."""
    print("=" * 60)
    print("  SYSTEM PERFORMANCE - KOBE TRADING SYSTEM")
    print("=" * 60)
    print(f"  Timestamp: {datetime.now().isoformat()}")

    # CPU
    print("\n--- CPU ---")
    cpu = get_cpu_usage()
    if cpu is not None:
        bar_length = int(cpu / 5)
        bar = "[" + "#" * bar_length + "-" * (20 - bar_length) + "]"
        print(f"  Usage: {cpu:5.1f}% {bar}")
    else:
        print("  [INFO] CPU stats unavailable (install psutil)")

    # Memory
    print("\n--- Memory ---")
    mem = get_memory_usage()
    if mem:
        if "percent" in mem:
            bar_length = int(mem["percent"] / 5)
            bar = "[" + "#" * bar_length + "-" * (20 - bar_length) + "]"
            print(f"  Usage: {mem['percent']:5.1f}% {bar}")
        if "total" in mem:
            print(f"  Total:     {format_size(mem['total'])}")
        if "used" in mem:
            print(f"  Used:      {format_size(mem['used'])}")
        if "available" in mem:
            print(f"  Available: {format_size(mem['available'])}")
        if "process_rss" in mem:
            print(f"  Process:   {format_size(mem['process_rss'])} (RSS)")
    else:
        print("  [INFO] Memory stats unavailable")

    # Disk
    print("\n--- Disk ---")
    disk = get_disk_usage()
    if disk:
        if "percent" in disk:
            bar_length = int(disk["percent"] / 5)
            bar = "[" + "#" * bar_length + "-" * (20 - bar_length) + "]"
            print(f"  Usage: {disk['percent']:5.1f}% {bar}")
        if "total" in disk:
            print(f"  Total: {format_size(disk['total'])}")
        if "free" in disk:
            print(f"  Free:  {format_size(disk['free'])}")
        if "project_size" in disk:
            print(f"  Project: {format_size(disk['project_size'])}")
    else:
        print("  [INFO] Disk stats unavailable")


def show_api_stats() -> None:
    """Display API response time statistics."""
    print("\n--- API Response Times ---")

    # Polygon
    print("\n  Polygon API:")
    polygon_stats = benchmark_polygon_api()
    if "error" in polygon_stats:
        print(f"    [ERROR] {polygon_stats['error']}")
    else:
        print(f"    Endpoint: {polygon_stats['endpoint']}")
        print(f"    Response: {polygon_stats['response_time_ms']} ms")
        print(f"    Status:   {polygon_stats['status_code']}")

    # Alpaca
    print("\n  Alpaca API:")
    alpaca_stats = benchmark_alpaca_api()
    if "error" in alpaca_stats:
        print(f"    [ERROR] {alpaca_stats['error']}")
    else:
        print(f"    Endpoint: {alpaca_stats['endpoint']}")
        print(f"    Response: {alpaca_stats['response_time_ms']} ms")
        print(f"    Status:   {alpaca_stats['status_code']}")


def run_live_monitoring(interval: int = 5) -> None:
    """Run live performance monitoring."""
    print("=" * 60)
    print("  LIVE MONITORING - Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")

            print("KOBE TRADING SYSTEM - Live Monitor")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)

            # CPU
            cpu = get_cpu_usage()
            if cpu is not None:
                bar = "#" * int(cpu / 5) + "-" * (20 - int(cpu / 5))
                print(f"CPU:  [{bar}] {cpu:5.1f}%")

            # Memory
            mem = get_memory_usage()
            if mem and "percent" in mem:
                bar = "#" * int(mem["percent"] / 5) + "-" * (20 - int(mem["percent"] / 5))
                print(f"MEM:  [{bar}] {mem['percent']:5.1f}%")
                if "process_rss" in mem:
                    print(f"      Process: {format_size(mem['process_rss'])}")

            # Disk
            disk = get_disk_usage()
            if disk and "percent" in disk:
                bar = "#" * int(disk["percent"] / 5) + "-" * (20 - int(disk["percent"] / 5))
                print(f"DISK: [{bar}] {disk['percent']:5.1f}%")

            print("-" * 40)
            print(f"Refresh: {interval}s | Ctrl+C to exit")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def run_profiling() -> None:
    """Profile common operations."""
    print("=" * 60)
    print("  PROFILING OPERATIONS")
    print("=" * 60)

    results: List[Dict[str, Any]] = []

    # Profile file operations
    print("\n--- File Operations ---")

    # JSON parsing
    settings_file = ROOT / "config" / "settings.json"
    if settings_file.exists():
        def parse_json():
            json.loads(settings_file.read_text(encoding="utf-8"))

        result = profile_operation("Parse settings.json", parse_json, iterations=100)
        results.append(result)
        print(f"  {result['name']}: {result['avg_time_ms']:.3f} ms avg ({result['iterations']} iterations)")

    # Hash chain verification
    try:
        from core.hash_chain import verify_chain
        result = profile_operation("Verify hash chain", verify_chain, iterations=10)
        results.append(result)
        print(f"  {result['name']}: {result['avg_time_ms']:.3f} ms avg ({result['iterations']} iterations)")
    except Exception as e:
        print(f"  [SKIP] Hash chain: {e}")

    # Universe loading
    try:
        from data.universe.loader import load_universe
        universe_files = list((ROOT / "data" / "universe").glob("*.csv"))
        if universe_files:
            def load_uni():
                load_universe(universe_files[0], cap=100)
            result = profile_operation("Load universe", load_uni, iterations=10)
            results.append(result)
            print(f"  {result['name']}: {result['avg_time_ms']:.3f} ms avg ({result['iterations']} iterations)")
    except Exception as e:
        print(f"  [SKIP] Universe loading: {e}")

    # Summary
    print("\n--- Profile Summary ---")
    for result in results:
        if "error" not in result:
            print(f"  {result['name']:30s} : {result['avg_time_ms']:8.3f} ms (min: {result['min_time_ms']:.3f}, max: {result['max_time_ms']:.3f})")


def run_benchmarks() -> None:
    """Run performance benchmarks."""
    print("=" * 60)
    print("  PERFORMANCE BENCHMARKS")
    print("=" * 60)

    results: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "benchmarks": {},
    }

    # API benchmarks
    print("\n--- API Benchmarks ---")

    print("  Testing Polygon API (3 requests)...")
    polygon_times: List[float] = []
    for i in range(3):
        stats = benchmark_polygon_api()
        if "response_time_ms" in stats:
            polygon_times.append(stats["response_time_ms"])
            print(f"    Request {i+1}: {stats['response_time_ms']} ms")
        time.sleep(0.5)

    if polygon_times:
        results["benchmarks"]["polygon_api"] = {
            "avg_ms": round(sum(polygon_times) / len(polygon_times), 2),
            "min_ms": round(min(polygon_times), 2),
            "max_ms": round(max(polygon_times), 2),
        }
        print(f"    Average: {results['benchmarks']['polygon_api']['avg_ms']} ms")

    print("\n  Testing Alpaca API (3 requests)...")
    alpaca_times: List[float] = []
    for i in range(3):
        stats = benchmark_alpaca_api()
        if "response_time_ms" in stats:
            alpaca_times.append(stats["response_time_ms"])
            print(f"    Request {i+1}: {stats['response_time_ms']} ms")
        time.sleep(0.5)

    if alpaca_times:
        results["benchmarks"]["alpaca_api"] = {
            "avg_ms": round(sum(alpaca_times) / len(alpaca_times), 2),
            "min_ms": round(min(alpaca_times), 2),
            "max_ms": round(max(alpaca_times), 2),
        }
        print(f"    Average: {results['benchmarks']['alpaca_api']['avg_ms']} ms")

    # Computation benchmarks
    print("\n--- Computation Benchmarks ---")

    # List comprehension benchmark
    def list_comp():
        return [i * i for i in range(10000)]

    result = profile_operation("List comprehension (10k)", list_comp, iterations=100)
    results["benchmarks"]["list_comprehension"] = result
    print(f"  List comprehension: {result['avg_time_ms']:.3f} ms")

    # JSON serialization benchmark
    sample_data = {"key" + str(i): {"value": i, "nested": {"a": 1, "b": 2}} for i in range(100)}

    def json_dumps():
        return json.dumps(sample_data)

    result = profile_operation("JSON serialize (100 objects)", json_dumps, iterations=100)
    results["benchmarks"]["json_serialize"] = result
    print(f"  JSON serialize: {result['avg_time_ms']:.3f} ms")

    # File I/O benchmark
    test_file = ROOT / "temp" / "benchmark_test.tmp"
    test_file.parent.mkdir(parents=True, exist_ok=True)

    def file_write():
        test_file.write_text("x" * 10000)

    result = profile_operation("File write (10KB)", file_write, iterations=50)
    results["benchmarks"]["file_write"] = result
    print(f"  File write (10KB): {result['avg_time_ms']:.3f} ms")

    # Cleanup
    if test_file.exists():
        test_file.unlink()

    # Summary
    print("\n--- Benchmark Summary ---")
    print(json.dumps(results["benchmarks"], indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Kobe Trading System - Performance Monitoring"
    )
    ap.add_argument(
        "--dotenv",
        type=str,
        default="C:/Users/Owner/OneDrive/Desktop/GAME_PLAN_2K28/.env",
        help="Path to .env file",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Live monitoring mode",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Profile slow operations",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks",
    )
    ap.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval for live mode (seconds)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    args = ap.parse_args()

    # Load environment if specified
    dotenv = Path(args.dotenv)
    if dotenv.exists():
        load_env(dotenv)

    # Execute requested action
    if args.live:
        run_live_monitoring(args.interval)
    elif args.profile:
        run_profiling()
    elif args.benchmark:
        run_benchmarks()
    else:
        # Default: show system stats
        show_system_stats()
        show_api_stats()


if __name__ == "__main__":
    main()
