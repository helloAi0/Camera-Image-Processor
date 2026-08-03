import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ObjectTracker", log_level=logging.DEBUG)


class ObjectTracker(BaseProcessor):
    """Production Object Tracking & Motion Analytics Engine supporting Lucas-Kanade, Farneback, and Kalman Filtering."""

    def __init__(self) -> None:
        """Initializes the ObjectTracker processor."""
        super().__init__(name="ObjectTracker")

    def calc_sparse_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates Lucas-Kanade sparse optical flow using multi-level image pyramids.

        Args:
            prev_gray (np.ndarray): Previous frame grayscale image array (uint8).
            curr_gray (np.ndarray): Current frame grayscale image array (uint8).
            prev_points (np.ndarray): Array of 2D feature coordinates to track (shape: [N, 1, 2], float32).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - curr_points (np.ndarray): Tracked 2D feature coordinates in current frame.
                - status (np.ndarray): Vector of status flags (1 if flow found, 0 otherwise).
                - err (np.ndarray): Tracking error vector.
        """
        self.validate_image(prev_gray)
        self.validate_image(curr_gray)

        if prev_points is None or len(prev_points) == 0:
            raise ValueError("prev_points array cannot be empty for sparse optical flow tracking.")

        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        curr_points, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points.astype(np.float32), None, **lk_params
        )

        logger.debug(f"Tracked {np.sum(status == 1)}/{len(prev_points)} features via Lucas-Kanade Flow")
        return curr_points, status, err

    def calc_dense_optical_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates Farneback dense optical flow and generates HSV motion visualization.

        Args:
            prev_gray (np.ndarray): Previous frame grayscale image array (uint8).
            curr_gray (np.ndarray): Current frame grayscale image array (uint8).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - flow (np.ndarray): Motion vector field of shape (H, W, 2) containing (dx, dy).
                - hsv_visual (np.ndarray): BGR image encoding motion direction (Hue) and speed (Value).
        """
        self.validate_image(prev_gray)
        self.validate_image(curr_gray)

        # Farneback polynomial expansion parameters
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Convert motion vectors to polar coordinates (magnitude and angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Map motion direction to HSV Hue [0, 179] and magnitude to Value [0, 255]
        hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        hsv_visual = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        logger.debug("Computed Farneback Dense Optical Flow field")
        return flow, hsv_visual

    def create_kalman_filter(self, initial_position: Tuple[float, float], dt: float = 1.0) -> cv2.KalmanFilter:
        """Instantiates a 2D constant-velocity Kalman Filter.

        State Vector X = [x, y, vx, vy]^T
        Measurement Vector Z = [x, y]^T

        Args:
            initial_position (Tuple[float, float]): Initial target coordinates (x, y).
            dt (float): Time step delta between state predictions. Default is 1.0.

        Returns:
            cv2.KalmanFilter: Initialized OpenCV Kalman Filter instance.
        """
        kf = cv2.KalmanFilter(4, 2)

        # State transition matrix F
        kf.transitionMatrix = np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # Measurement matrix H
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # Process noise covariance Q
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Measurement noise covariance R
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Posteriori error covariance P
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        # Initial state setup
        x0, y0 = initial_position
        kf.statePost = np.array([[x0], [y0], [0.0], [0.0]], dtype=np.float32)

        logger.debug(f"Initialized Kalman Filter at position ({x0}, {y0})")
        return kf

    def update_kalman(
        self,
        kf: cv2.KalmanFilter,
        measurement: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Executes Kalman Filter Prediction and Measurement Correction steps.

        Args:
            kf (cv2.KalmanFilter): Active Kalman Filter instance.
            measurement (Optional[Tuple[float, float]]): Observed position (x, y). If None, relies solely on prediction.

        Returns:
            Tuple[Tuple[float, float], Tuple[float, float]]:
                - position (x, y): Estimated position.
                - velocity (vx, vy): Estimated velocity.
        """
        # 1. Predict state transition
        prediction = kf.predict()

        # 2. Correct state estimate if measurement is available
        if measurement is not None:
            meas_matrix = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            state = kf.correct(meas_matrix)
        else:
            state = prediction

        pos_x, pos_y = float(state[0, 0]), float(state[1, 0])
        vel_x, vel_y = float(state[2, 0]), float(state[3, 0])

        return (pos_x, pos_y), (vel_x, vel_y)

    def process(self, image: np.ndarray, mode: str = "dense_flow", **kwargs: Any) -> Any:
        """Executes motion tracking pipeline algorithms.

        Args:
            image (np.ndarray): Current frame array (grayscale or BGR).
            mode (str): Execution mode ('sparse_flow', 'dense_flow', 'kalman').
            **kwargs (Any): Mode-specific parameters (prev_image, prev_points, kf_instance, measurement).

        Returns:
            Any: Motion vectors, visualization matrix, or state estimates depending on mode.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        if mode_lower in ["sparse_flow", "lk"]:
            prev_img = kwargs.get("prev_image")
            prev_pts = kwargs.get("prev_points")
            if prev_img is None or prev_pts is None:
                raise ValueError("Mode 'sparse_flow' requires 'prev_image' and 'prev_points' in kwargs.")

            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) if prev_img.ndim == 3 else prev_img.copy()
            return self.calc_sparse_optical_flow(prev_gray, curr_gray, prev_pts)

        elif mode_lower in ["dense_flow", "farneback"]:
            prev_img = kwargs.get("prev_image")
            if prev_img is None:
                raise ValueError("Mode 'dense_flow' requires 'prev_image' in kwargs.")

            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY) if prev_img.ndim == 3 else prev_img.copy()
            return self.calc_dense_optical_flow(prev_gray, curr_gray)

        elif mode_lower == "kalman":
            kf = kwargs.get("kf_instance")
            meas = kwargs.get("measurement")
            if kf is None:
                initial_pos = meas if meas is not None else (0.0, 0.0)
                kf = self.create_kalman_filter(initial_pos)

            return self.update_kalman(kf, measurement=meas)

        else:
            logger.warning(f"Unknown tracking mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    tracker = ObjectTracker()

    # Generate synthetic sequential 400x400 frames with a moving target circle
    frame1 = np.zeros((400, 400, 3), dtype=np.uint8)
    frame2 = np.zeros((400, 400, 3), dtype=np.uint8)

    # Circle shifts down-right by (+20 px, +15 px) in frame 2
    cv2.circle(frame1, (100, 100), 25, (255, 255, 255), -1)
    cv2.circle(frame2, (120, 115), 25, (255, 255, 255), -1)

    # 1. Test Lucas-Kanade Sparse Optical Flow (tracking feature point at initial target center)
    initial_pts = np.array([[[100.0, 100.0]]], dtype=np.float32)
    (tracked_pts, status, err), t_sparse = tracker.execute_with_timing(
        frame2, mode="sparse_flow", prev_image=frame1, prev_points=initial_pts
    )

    # 2. Test Farneback Dense Optical Flow
    (flow_field, flow_visual), t_dense = tracker.execute_with_timing(
        frame2, mode="dense_flow", prev_image=frame1
    )

    # 3. Test Kalman Filter state estimation & trajectory prediction
    kf_filter = tracker.create_kalman_filter(initial_position=(100.0, 100.0))
    
    # Observe trajectory over 3 timesteps (including an occlusion step with no measurement)
    obs_1, t_k1 = tracker.execute_with_timing(frame2, mode="kalman", kf_instance=kf_filter, measurement=(120.0, 115.0))
    obs_2_occluded, t_k2 = tracker.execute_with_timing(frame2, mode="kalman", kf_instance=kf_filter, measurement=None)

    logger.info(f"Sparse LK Tracked Point: {tracked_pts[0][0].round(1)} (completed in {t_sparse:.3f} ms)")
    logger.info(f"Dense Optical Flow visual matrix shape: {flow_visual.shape} (completed in {t_dense:.3f} ms)")
    logger.info(f"Kalman Filter Correction Step 1 Position: {obs_1[0][0]:.1f}, {obs_1[0][1]:.1f}")
    logger.info(f"Kalman Filter Occlusion Prediction Step Position: {obs_2_occluded[0][0]:.1f}, {obs_2_occluded[0][1]:.1f}")