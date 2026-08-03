import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("StereoVision", log_level=logging.DEBUG)


class StereoVision(BaseProcessor):
    """Production Stereo Vision Engine supporting Disparity Map Estimation and 3D Point Cloud Reprojection."""

    def __init__(self) -> None:
        """Initializes the StereoVision processor."""
        super().__init__(name="StereoVision")

    def compute_disparity(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        num_disparities: int = 64,
        block_size: int = 15,
        algorithm: str = "sgbm",
    ) -> np.ndarray:
        """Computes dense pixel-wise disparity map between rectified stereo image pair.

        Args:
            img_left (np.ndarray): Rectified left camera grayscale image (uint8).
            img_right (np.ndarray): Rectified right camera grayscale image (uint8).
            num_disparities (int): Search range for disparity (must be divisible by 16).
            block_size (int): Matched block size (odd integer >= 3).
            algorithm (str): Matching algorithm ('sgbm' for Semi-Global or 'bm' for Block Matching).

        Returns:
            np.ndarray: Disparity map (float32, with values in pixel displacement).
        """
        self.validate_image(img_left)
        self.validate_image(img_right)

        left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY) if img_left.ndim == 3 else img_left.copy()
        right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY) if img_right.ndim == 3 else img_right.copy()

        algo_lower = algorithm.lower()

        if algo_lower == "sgbm":
            # Semi-Global Block Matching parameter selection
            p1 = 8 * 1 * block_size ** 2
            p2 = 32 * 1 * block_size ** 2

            matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=p1,
                P2=p2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        elif algo_lower == "bm":
            matcher = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size,
            )
        else:
            raise ValueError(f"Unsupported stereo algorithm '{algorithm}'. Choose 'sgbm' or 'bm'.")

        # StereoSGBM outputs disparity scaled by 16
        raw_disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        logger.debug(f"Computed stereo disparity map using '{algo_lower}' algorithm")
        return raw_disparity

    def create_reprojection_matrix(
        self,
        focal_length: float,
        baseline: float,
        principal_point: Tuple[float, float],
    ) -> np.ndarray:
        """Constructs 4x4 Disparity-to-Depth Reprojection Matrix (Q).

        Args:
            focal_length (float): Camera focal length in pixels (f).
            baseline (float): Physical distance between left and right camera centers in meters (B).
            principal_point (Tuple[float, float]): Optical center coordinates (cx, cy).

        Returns:
            np.ndarray: 4x4 Q reprojection matrix.
        """
        cx, cy = principal_point
        Q = np.array(
            [
                [1.0, 0.0, 0.0, -cx],
                [0.0, 1.0, 0.0, -cy],
                [0.0, 0.0, 0.0, focal_length],
                [0.0, 0.0, -1.0 / baseline, 0.0],
            ],
            dtype=np.float32,
        )
        return Q

    def reproject_to_3d(
        self,
        disparity: np.ndarray,
        Q: np.ndarray,
        max_depth: float = 10.0,
    ) -> np.ndarray:
        """Reprojects 2D disparity map into 3D spatial point cloud coordinates (X, Y, Z).

        Args:
            disparity (np.ndarray): Disparity map array (float32).
            Q (np.ndarray): 4x4 Reprojection matrix.
            max_depth (float): Maximum valid metric depth threshold in meters.

        Returns:
            np.ndarray: 3D point cloud matrix of shape (H, W, 3) containing metric (X, Y, Z).
        """
        # Reproject to 3D spatial points
        points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)

        # Mask out non-positive disparities and out-of-range depths
        invalid_mask = (disparity <= 0.0) | (points_3d[:, :, 2] > max_depth) | np.isinf(points_3d[:, :, 2])
        points_3d[invalid_mask] = np.nan

        logger.debug(f"Reprojected disparity map to 3D point cloud (Max valid depth: {max_depth} m)")
        return points_3d

    def process(self, image: np.ndarray, mode: str = "disparity", **kwargs: Any) -> Any:
        """Executes stereo vision depth estimation processing.

        Args:
            image (np.ndarray): Rectified left camera image array.
            mode (str): Execution mode ('disparity', 'point_cloud').
            **kwargs (Any): Mode parameters (right_image, Q_matrix, focal_length, baseline, principal_point).

        Returns:
            Any: Disparity map or 3D point cloud tensor.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        right_img = kwargs.get("right_image")
        if right_img is None:
            raise ValueError("Stereo processing requires 'right_image' in kwargs.")

        num_disp = kwargs.get("num_disparities", 64)
        block_sz = kwargs.get("block_size", 15)
        algo = kwargs.get("algorithm", "sgbm")

        disparity = self.compute_disparity(
            image, right_img, num_disparities=num_disp, block_size=block_sz, algorithm=algo
        )

        if mode_lower in ["disparity", "disp"]:
            return disparity

        elif mode_lower in ["point_cloud", "3d"]:
            Q = kwargs.get("Q_matrix")
            if Q is None:
                f = kwargs.get("focal_length", 800.0)
                b = kwargs.get("baseline", 0.1)
                cx = kwargs.get("cx", image.shape[1] / 2.0)
                cy = kwargs.get("cy", image.shape[0] / 2.0)
                Q = self.create_reprojection_matrix(focal_length=f, baseline=b, principal_point=(cx, cy))

            max_d = kwargs.get("max_depth", 10.0)
            return self.reproject_to_3d(disparity, Q, max_depth=max_d)

        else:
            logger.warning(f"Unknown mode '{mode}'. Returning original left image.")
            return image


if __name__ == "__main__":
    stereo = StereoVision()

    # Generate synthetic 400x400 rectified left and right image pair
    left_img = np.zeros((400, 400, 3), dtype=np.uint8)
    right_img = np.zeros((400, 400, 3), dtype=np.uint8)

    # Place target square at x=150 in Left image and x=130 in Right image (Disparity = 20 pixels)
    cv2.rectangle(left_img, (150, 150), (250, 250), (255, 255, 255), -1)
    cv2.rectangle(right_img, (130, 150), (230, 250), (255, 255, 255), -1)

    # Camera Calibration Parameters: Focal Length = 500 px, Baseline = 0.2 meters (20 cm)
    focal_length = 500.0
    baseline = 0.2
    principal_point = (200.0, 200.0)

    # 1. Compute Disparity Map
    disparity_map, t_disp = stereo.execute_with_timing(
        left_img,
        mode="disparity",
        right_image=right_img,
        num_disparities=64,
        block_size=15,
        algorithm="sgbm",
    )

    # 2. Reproject to 3D Point Cloud
    points_3d, t_cloud = stereo.execute_with_timing(
        left_img,
        mode="point_cloud",
        right_image=right_img,
        focal_length=focal_length,
        baseline=baseline,
        cx=principal_point[0],
        cy=principal_point[1],
    )

    # Inspect reconstructed 3D position of target center pixel (x=200, y=200)
    target_3d = points_3d[200, 200]

    logger.info(f"Computed disparity map in {t_disp:.3f} ms (Target Disparity: {disparity_map[200, 200]:.1f} px)")
    logger.info(f"Generated 3D point cloud in {t_cloud:.3f} ms")
    logger.info(f"Target Center 3D Position -> X: {target_3d[0]:.2f}m, Y: {target_3d[1]:.2f}m, Depth Z: {target_3d[2]:.2f}m")