import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("StreamPipeline", log_level=logging.DEBUG)


class StreamPipeline(BaseProcessor):
    """Production Multi-Threaded Stream Pipeline decoupling frame acquisition from processing workers."""

    def __init__(self, queue_capacity: int = 4) -> None:
        """Initializes the StreamPipeline processor.

        Args:
            queue_capacity (int): Maximum ring buffer size before dropping stale frames.
        """
        super().__init__(name="StreamPipeline")
        self.queue_capacity = queue_capacity
        self.frame_queue: queue.Queue = queue.Queue(maxsize=queue_capacity)
        
        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        
        # Telemetry metrics
        self.total_frames_captured: int = 0
        self.total_frames_processed: int = 0
        self.total_frames_dropped: int = 0
        self.start_time: float = 0.0

    def start_stream(self, source: Union[int, str, Callable[[], Optional[np.ndarray]]]) -> None:
        """Spawns the background producer thread for video stream acquisition.

        Args:
            source (Union[int, str, Callable]): Video source index (int for webcam), 
                                                file path (str), or custom frame generator function.
        """
        if self._producer_thread is not None and self._producer_thread.is_alive():
            logger.warning("Stream producer thread is already running.")
            return

        self._stop_event.clear()
        self.total_frames_captured = 0
        self.total_frames_processed = 0
        self.total_frames_dropped = 0
        self.start_time = time.perf_counter()

        self._producer_thread = threading.Thread(
            target=self._frame_producer, args=(source,), daemon=True, name="FrameProducerThread"
        )
        self._producer_thread.start()
        logger.info(f"Started Frame Producer Thread (Queue Capacity: {self.queue_capacity})")

    def _frame_producer(self, source: Union[int, str, Callable[[], Optional[np.ndarray]]]) -> None:
        """Background thread target reading frames and populating the frame queue non-blockingly."""
        cap = None
        is_generator = callable(source)

        if not is_generator:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                self._stop_event.set()
                return

        while not self._stop_event.is_set():
            if is_generator:
                frame = source()
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("Video stream ended or frame read failed.")
                    break

            if frame is None:
                continue

            self.total_frames_captured += 1
            timestamp = time.perf_counter()

            # Non-blocking enqueue: drop oldest frame if buffer capacity is reached
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.total_frames_dropped += 1
                except queue.Empty:
                    pass

            self.frame_queue.put((timestamp, frame))

        if cap is not None:
            cap.release()
        logger.debug("Frame Producer Thread terminated gracefully.")

    def read_frame(self, timeout: float = 0.5) -> Tuple[bool, Optional[float], Optional[np.ndarray]]:
        """Consumes the newest frame from the thread-safe frame queue.

        Args:
            timeout (float): Maximum wait time in seconds for a frame to become available.

        Returns:
            Tuple[bool, Optional[float], Optional[np.ndarray]]:
                - success (bool): True if frame consumed, False if queue timed out.
                - capture_timestamp (Optional[float]): Perf counter timestamp at frame capture.
                - frame (Optional[np.ndarray]): Captured image array.
        """
        try:
            timestamp, frame = self.frame_queue.get(block=True, timeout=timeout)
            self.total_frames_processed += 1
            return True, timestamp, frame
        except queue.Empty:
            return False, None, None

    def stop_stream(self) -> None:
        """Signals background threads to stop and waits for thread join completion."""
        self._stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=2.0)
            logger.info("Stopped Frame Producer Thread.")

    def get_telemetry(self) -> Dict[str, Any]:
        """Calculates real-time pipeline performance telemetry.

        Returns:
            Dict[str, Any]: Metrics dictionary including FPS, dropped frames, and drop rate %.
        """
        elapsed = time.perf_counter() - self.start_time if self.start_time > 0 else 1e-6
        fps = self.total_frames_processed / elapsed
        drop_rate = (self.total_frames_dropped / max(1, self.total_frames_captured)) * 100.0

        return {
            "elapsed_sec": round(elapsed, 2),
            "captured_frames": self.total_frames_captured,
            "processed_frames": self.total_frames_processed,
            "dropped_frames": self.total_frames_dropped,
            "drop_rate_pct": round(drop_rate, 2),
            "effective_fps": round(fps, 2),
            "queue_unconsumed": self.frame_queue.qsize(),
        }

    def process(self, image: np.ndarray, mode: str = "passthrough", **kwargs: Any) -> Any:
        """Applies operational mode transformations or pipeline passthrough.

        Args:
            image (np.ndarray): Input frame array.
            mode (str): Execution mode ('passthrough').
            **kwargs (Any): Processor parameters.

        Returns:
            Any: Processed output image.
        """
        self.validate_image(image)
        return image


if __name__ == "__main__":
    pipeline = StreamPipeline(queue_capacity=3)

    # Synthetic frame generator function simulating a 100 FPS camera stream
    frame_counter = 0

    def mock_camera_stream() -> np.ndarray:
        global frame_counter
        frame_counter += 1
        time.sleep(0.01)  # Simulate ~100 FPS camera sensor interval (10 ms)
        canvas = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.putText(
            canvas,
            f"Frame #{frame_counter}",
            (30, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        return canvas

    # 1. Start asynchronous stream producer thread
    pipeline.start_stream(source=mock_camera_stream)

    # 2. Simulate consumer worker processing frames with artificial heavy delay (30 ms per frame)
    logger.info("Simulating consumer loop processing frames with 30ms heavy workload...")
    processed_count = 0

    try:
        while processed_count < 20:
            success, ts_capture, frame = pipeline.read_frame(timeout=1.0)
            if not success or frame is None:
                break

            # Measure latency from camera capture to processing ingestion
            latency_ms = (time.perf_counter() - ts_capture) * 1000.0

            # Simulate heavy downstream vision workload (e.g., Detection + Tracking)
            time.sleep(0.03)  # 30 ms workload (~33 FPS max consumer limit)
            processed_count += 1

            if processed_count % 5 == 0:
                logger.debug(f"Consumer processed frame latency: {latency_ms:.2f} ms")

    finally:
        pipeline.stop_stream()

    # 3. Print Telemetry Report
    telemetry = pipeline.get_telemetry()
    logger.info("=== STREAM PIPELINE TELEMETRY REPORT ===")
    for key, val in telemetry.items():
        logger.info(f"  {key}: {val}")