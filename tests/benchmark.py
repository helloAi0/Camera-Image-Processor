import logging
import time
from typing import Dict, List
import cv2
import numpy as np

from src.pipeline.stream_pipeline import StreamPipeline
from src.stereo.stereo_vision import StereoVision
from src.utils.logger import setup_logger

logger = setup_logger("SystemBenchmark", log_level=logging.INFO)


class VisionSystemBenchmark:
    """Production Benchmark Engine for measuring computer vision pipeline latency and throughput."""

    def __init__(self, iterations: int = 50) -> None:
        """Initializes the benchmarking suite.

        Args:
            iterations (int): Number of test iterations per benchmark stage.
        """
        self.iterations = iterations
        self.resolutions: Dict[str, Tuple[int, int]] = {
            "720p (HD)": (1280, 720),
            "1080p (FHD)": (1920, 1080),
        }

    def benchmark_stereo_vision(self) -> Dict[str, Dict[str, float]]:
        """Benchmarks Stereo Disparity & 3D Reprojection latency across resolutions."""
        logger.info("=== Starting Stereo Vision Subsystem Benchmark ===")
        stereo = StereoVision()
        results = {}

        for res_name, (width, height) in self.resolutions.items():
            left = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            right = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            latencies: List[float] = []

            # Warmup run
            _ = stereo.compute_disparity(left, right, num_disparities=32, block_size=11)

            for _ in range(self.iterations):
                t0 = time.perf_counter()
                _ = stereo.compute_disparity(left, right, num_disparities=32, block_size=11)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

            p50 = float(np.median(latencies))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            throughput_fps = 1000.0 / p50 if p50 > 0 else 0.0

            results[res_name] = {
                "P50_ms": round(p50, 2),
                "P95_ms": round(p95, 2),
                "P99_ms": round(p99, 2),
                "Throughput_FPS": round(throughput_fps, 2),
            }
            logger.info(f" [{res_name}] P50: {p50:.2f} ms | P95: {p95:.2f} ms | FPS: {throughput_fps:.1f}")

        return results

    def print_summary_table(self, stereo_stats: Dict[str, Dict[str, float]]) -> None:
        """Formats and prints markdown performance summary table."""
        print("\n" + "=" * 65)
        print("          PRODUCTION COMPUTER VISION BENCHMARK REPORT          ")
        print("=" * 65)
        print(f"| {'Resolution':<15} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'Est. FPS':<10} |")
        print("|" + "-" * 17 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")

        for res, stats in stereo_stats.items():
            print(
                f"| {res:<15} | {stats['P50_ms']:<10.2f} | {stats['P95_ms']:<10.2f} | {stats['Throughput_FPS']:<10.1f} |"
            )
        print("=" * 65 + "\n")


if __name__ == "__main__":
    bench = VisionSystemBenchmark(iterations=10)
    stereo_results = bench.benchmark_stereo_vision()
    bench.print_summary_table(stereo_results)