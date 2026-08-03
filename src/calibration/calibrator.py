import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("CameraCalibrator", log_level=logging.DEBUG)


class CameraCalibrator(BaseProcessor):
    """Production Camera Calibration and Geometric Transformation Engine."""

    def __init__(self) -> None:
        """Initializes the CameraCalibrator processor."""
        super().__init__(name="CameraCalibrator")

    def calibrate_from_images(
        self,
        images: List[np.ndarray],
        pattern_size: Tuple[int, int] = (9, 6),
        square_size: float = 1.0,
    ) -> Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Computes camera intrinsic matrix and distortion coefficients using chessboard targets.

        Args:
            images (List[np.ndarray]): List of calibration target images (BGR/Gray).
            pattern_size (Tuple[int, int]): Internal corners per chessboard grid (cols, rows). Default is (9, 6).
            square_size (float): Real-world physical size of a chessboard square (e.g., millimeters or meters).

        Returns:
            Tuple[bool, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
                - ret (bool): Success status of calibration.
                - camera_matrix (np.ndarray): 3x3 Intrinsic camera matrix K.
                - dist_coeffs (np.ndarray): Distortion coefficients (k1, k2, p1, p2, k3).
                - rvecs (List[np.ndarray]): Rotation vectors per calibration image.
                - tvecs (List[np.ndarray]): Translation vectors per calibration image.
        """
        if not images:
            raise ValueError("Image list for calibration cannot be empty.")

        # Prepare 3D object points in target coordinate space (Z=0)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        objpoints: List[np.ndarray] = []  # 3d points in real world space
        imgpoints: List[np.ndarray] = []  # 2d points in image plane

        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        image_shape: Optional[Tuple[int, int]] = None

        for idx, img in enumerate(images):
            self.validate_image(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

            if image_shape is None:
                image_shape = gray.shape[::-1]

            # Find interior chessboard corners
            found, corners = cv2.findChessboardCorners(
                gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                objpoints.append(objp)
                # Refine corner locations to sub-pixel accuracy
                corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
                imgpoints.append(corners_subpix)
                logger.debug(f"Chessboard pattern detected in calibration frame {idx}")

        if not objpoints:
            logger.error("Failed to detect chessboard corners in any provided image.")
            return False, np.eye(3), np.zeros(5), [], []

        # Perform camera calibration optimization
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_shape, None, None
        )

        logger.info(f"Camera Calibration Complete. RMS Error = {ret:.4f}")
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs

    def undistort_image(
        self,
        image: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Removes radial and tangential lens distortion from an input image.

        Args:
            image (np.ndarray): Input BGR/Gray image array.
            camera_matrix (np.ndarray): 3x3 intrinsic matrix.
            dist_coeffs (np.ndarray): Distortion coefficients.

        Returns:
            np.ndarray: Lens-rectified output image array.
        """
        self.validate_image(image)
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
        logger.debug("Applied lens undistortion correction.")
        return undistorted

    def warp_perspective(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Applies a 3x3 Homography perspective transformation (e.g., Bird's Eye View).

        Args:
            image (np.ndarray): Input image array.
            src_points (np.ndarray): Float32 array of 4 source coordinates (shape: [4, 2]).
            dst_points (np.ndarray): Float32 array of 4 target coordinates (shape: [4, 2]).
            output_size (Optional[Tuple[int, int]]): Desired output (width, height). Defaults to input dimension.

        Returns:
            np.ndarray: Perspective-warped image output.
        """
        self.validate_image(image)
        if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
            raise ValueError("src_points and dst_points must both have shape (4, 2)")

        h, w = image.shape[:2]
        out_w, out_h = output_size if output_size else (w, h)

        # Compute 3x3 perspective homography matrix H
        homography_matrix = cv2.getPerspectiveTransform(
            src_points.astype(np.float32), dst_points.astype(np.float32)
        )

        # Execute projective transformation
        warped = cv2.warpPerspective(image, homography_matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
        logger.debug("Applied perspective transformation warp.")
        return warped

    def process(self, image: np.ndarray, mode: str = "undistort", **kwargs: Any) -> Any:
        """Executes geometric calibration operations.

        Args:
            image (np.ndarray): Input image array.
            mode (str): Execution mode ('undistort', 'warp_perspective').
            **kwargs (Any): Mode-specific calibration parameters.

        Returns:
            Any: Rectified or warped image output.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "undistort":
            h, w = image.shape[:2]
            # Fallback synthetic intrinsic matrix if not provided
            default_matrix = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1.0]], dtype=np.float32)
            default_dist = np.array([-0.2, 0.05, 0.0, 0.0, 0.0], dtype=np.float32)

            k_matrix = kwargs.get("camera_matrix", default_matrix)
            d_coeffs = kwargs.get("dist_coeffs", default_dist)
            return self.undistort_image(image, camera_matrix=k_matrix, dist_coeffs=d_coeffs)

        elif mode_lower in ["warp_perspective", "bev"]:
            h, w = image.shape[:2]
            # Default trapezoid-to-rectangle perspective warp (Bird's Eye View simulation)
            default_src = np.array(
                [[w * 0.2, h * 0.8], [w * 0.8, h * 0.8], [w * 0.6, h * 0.4], [w * 0.4, h * 0.4]],
                dtype=np.float32,
            )
            default_dst = np.array(
                [[w * 0.2, h * 0.9], [w * 0.8, h * 0.9], [w * 0.8, h * 0.1], [w * 0.2, h * 0.1]],
                dtype=np.float32,
            )

            src = kwargs.get("src_points", default_src)
            dst = kwargs.get("dst_points", default_dst)
            out_size = kwargs.get("output_size", (w, h))

            return self.warp_perspective(image, src_points=src, dst_points=dst, output_size=out_size)

        else:
            logger.warning(f"Unknown calibration mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    calibrator = CameraCalibrator()

    # Create synthetic test canvas with grid pattern
    test_canvas = np.zeros((400, 400, 3), dtype=np.uint8)
    for x in range(0, 400, 40):
        cv2.line(test_canvas, (x, 0), (x, 400), (255, 255, 255), 1)
    for y in range(0, 400, 40):
        cv2.line(test_canvas, (0, y), (400, y), (255, 255, 255), 1)

    # 1. Test Undistortion execution timing
    undistorted_img, t_undistort = calibrator.execute_with_timing(test_canvas, mode="undistort")

    # 2. Test Bird's Eye View Perspective Transformation timing
    warped_bev, t_bev = calibrator.execute_with_timing(test_canvas, mode="warp_perspective")

    logger.info(f"Lens Undistortion executed in {t_undistort:.3f} ms")
    logger.info(f"Perspective Warp executed in {t_bev:.3f} ms")