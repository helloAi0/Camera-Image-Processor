import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.pipeline.stream_pipeline import StreamPipeline
from src.utils.logger import setup_logger

logger = setup_logger("CLIOrchestrator", log_level=logging.INFO)


class HUDVisualizer:
    """Renders real-time operational metrics and diagnostic HUD overlays onto frame streams."""

    @staticmethod
    def draw_telemetry_hud(
        frame: np.ndarray,
        telemetry: Dict[str, Any],
        latency_ms: float,
        active_mode: str,
    ) -> np.ndarray:
        """Overlays processing performance stats on top-left of the canvas.

        Args:
            frame (np.ndarray): Target video frame.
            telemetry (Dict[str, Any]): Dictionary containing stream pipeline metrics.
            latency_ms (float): End-to-end processing latency for current frame.
            active_mode (str): Name of active vision processing mode.

        Returns:
            np.ndarray: Frame with HUD overlay applied.
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw semi-transparent background panel
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (320, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)

        # HUD Text items: (text, position_y, color)
        fps = telemetry.get("effective_fps", 0.0)
        dropped = telemetry.get("dropped_frames", 0)
        drop_pct = telemetry.get("drop_rate_pct", 0.0)
        queue_size = telemetry.get("queue_unconsumed", 0)

        lines = [
            (f"MODE: {active_mode.upper()}", 30, (0, 255, 255)),
            (f"FPS: {fps:.1f}", 55, (0, 255, 0) if fps > 20 else (0, 0, 255)),
            (f"Latency: {latency_ms:.1f} ms", 80, (255, 255, 255)),
            (f"Queue Size: {queue_size}", 105, (255, 200, 0)),
            (f"Dropped: {dropped} ({drop_pct:.1f}%)", 130, (100, 100, 255)),
        ]

        for text, y_pos, color in lines:
            cv2.putText(
                output,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                1,
                cv2.LINE_AA,
            )

        # Footer interaction guide
        cv2.putText(
            output,
            "[Q] Quit | [S] Save Frame | [T] Toggle HUD",
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return output


class ApplicationOrchestrator:
    """Top-level Vision Engine Application Orchestrator managing pipeline lifecycle and CLI options."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initializes the application runtime with parsed CLI arguments.

        Args:
            args (argparse.Namespace): Parsed command line arguments.
        """
        self.args = args
        self.pipeline = StreamPipeline(queue_capacity=args.queue_capacity)
        self.show_hud = True
        self.snapshot_dir = "snapshots"

        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def _resolve_source(self) -> Any:
        """Resolves argument input source to integer webcam index, video file path, or synthetic generator."""
        source_str = self.args.source
        if source_str.isdigit():
            return int(source_str)
        elif source_str.lower() == "synthetic":
            # Generate synthetic moving square for testing without physical camera hardware
            self.synth_counter = 0

            def synthetic_generator() -> np.ndarray:
                self.synth_counter += 1
                time.sleep(0.015)  # Simulate ~60 FPS source
                canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                x = (self.synth_counter * 5) % 500 + 50
                y = 200 + int(30 * np.sin(self.synth_counter * 0.1))
                cv2.rectangle(canvas, (x, y), (x + 80, y + 80), (0, 255, 120), -1)
                cv2.putText(
                    canvas,
                    f"Frame #{self.synth_counter}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                return canvas

            return synthetic_generator
        else:
            return source_str

    def run(self) -> None:
        """Executes the main application loop, handling processing, HUD rendering, and keyboard inputs."""
        source = self._resolve_source()
        logger.info(f"Starting Vision Pipeline Engine (Source: {self.args.source}, Mode: {self.args.mode})")

        self.pipeline.start_stream(source)
        frame_idx = 0

        try:
            while True:
                success, capture_ts, frame = self.pipeline.read_frame(timeout=0.5)

                if not success or frame is None:
                    # Stream ended or timeout reached
                    if isinstance(source, str) and source != "synthetic":
                        logger.info("End of video stream file reached.")
                        break
                    continue

                t_start = time.perf_counter()

                # Simulated processing stage based on active CLI mode selection
                processed_frame = frame.copy()
                if self.args.mode == "edge":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    processed_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                elif self.args.mode == "invert":
                    processed_frame = cv2.bitwise_not(frame)

                latency_ms = (time.perf_counter() - t_start) * 1000.0
                frame_idx += 1

                # Render Telemetry HUD overlay
                telemetry = self.pipeline.get_telemetry()
                display_canvas = processed_frame
                if self.show_hud:
                    display_canvas = HUDVisualizer.draw_telemetry_hud(
                        processed_frame, telemetry, latency_ms, self.args.mode
                    )

                # Show frame if display flag is enabled
                if not self.args.headless:
                    cv2.imshow("Production Vision Engine Dashboard", display_canvas)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        logger.info("Quit command received from user keyboard input.")
                        break
                    elif key == ord("t"):
                        self.show_hud = not self.show_hud
                        logger.info(f"Toggled Telemetry HUD Overlay: {self.show_hud}")
                    elif key == ord("s"):
                        filename = os.path.join(self.snapshot_dir, f"snapshot_{int(time.time())}.png")
                        cv2.imwrite(filename, display_canvas)
                        logger.info(f"Saved frame snapshot to: {filename}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Cleaning up...")

        finally:
            self.pipeline.stop_stream()
            cv2.destroyAllWindows()
            
            # Output Final Performance Summary Report
            final_stats = self.pipeline.get_telemetry()
            logger.info("================ FINAL EXECUTION SUMMARY ================")
            logger.info(f" Total Elapsed Time : {final_stats['elapsed_sec']} seconds")
            logger.info(f" Frames Processed   : {final_stats['processed_frames']}")
            logger.info(f" Effective Pipeline : {final_stats['effective_fps']} FPS")
            logger.info(f" Dropped Frames     : {final_stats['dropped_frames']} ({final_stats['drop_rate_pct']}%)")
            logger.info("========================================================")


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs command-line argument options parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser object.
    """
    parser = argparse.ArgumentParser(
        description="Production Real-Time Computer Vision & Video Stream Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--source",
        type=str,
        default="synthetic",
        help="Video source input: camera index ('0'), file path ('video.mp4'), or 'synthetic'.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="passthrough",
        choices=["passthrough", "edge", "invert"],
        help="Active vision processing mode pipeline.",
    )
    parser.add_argument(
        "-q",
        "--queue-capacity",
        type=int,
        default=4,
        help="Maximum thread-safe ring buffer size for stream pipeline.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run engine in headless mode without GUI rendering windows.",
    )
    return parser


def main() -> None:
    """CLI entry point function."""
    parser = build_arg_parser()
    args = parser.parse_args()
    orchestrator = ApplicationOrchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()