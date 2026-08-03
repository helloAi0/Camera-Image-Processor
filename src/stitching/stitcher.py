import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ImageStitcher", log_level=logging.DEBUG)


class ImageStitcher(BaseProcessor):
    """Production Image Registration & Panoramic Stitching Engine supporting RANSAC Homography and Linear Alpha Blending."""

    def __init__(self) -> None:
        """Initializes the ImageStitcher processor."""
        super().__init__(name="ImageStitcher")

    def estimate_homography(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        ransac_threshold: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimates the 3x3 projective Homography matrix using RANSAC.

        Args:
            src_points (np.ndarray): Array of source 2D point coordinates (N, 2) or (N, 1, 2).
            dst_points (np.ndarray): Array of destination 2D point coordinates (N, 2) or (N, 1, 2).
            ransac_threshold (float): Maximum allowed reprojection error in pixels to treat a point as an inlier.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - H (np.ndarray): Estimated 3x3 projective transformation matrix.
                - mask (np.ndarray): Inlier binary mask array (1 for inliers, 0 for outliers).
        """
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError("At least 4 point correspondences are required to compute a Homography matrix.")

        src_pts = src_points.reshape(-1, 1, 2).astype(np.float32)
        dst_pts = dst_points.reshape(-1, 1, 2).astype(np.float32)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)

        if H is None:
            raise RuntimeError("Homography estimation failed. Point correspondences may be degenerate.")

        inliers_count = int(np.sum(mask)) if mask is not None else 0
        logger.debug(f"Computed Homography matrix with {inliers_count}/{len(src_points)} RANSAC inliers")
        return H, mask

    def warp_perspective(
        self,
        image: np.ndarray,
        H: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Warps an image using a 3x3 Homography projective transformation matrix.

        Args:
            image (np.ndarray): Input image array.
            H (np.ndarray): 3x3 Homography transformation matrix.
            output_size (Tuple[int, int]): Output dimensions as (width, height).

        Returns:
            np.ndarray: Perspective-warped image array.
        """
        self.validate_image(image)
        warped = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_LINEAR)
        logger.debug(f"Warped image shape {image.shape[:2]} to target canvas size {output_size}")
        return warped

    def blend_linear(
        self,
        warped_img1: np.ndarray,
        img2_canvas: np.ndarray,
    ) -> np.ndarray:
        """Blends two overlapping image layers using linear alpha ramp smooth blending.

        Args:
            warped_img1 (np.ndarray): Warped source image mapped to the canvas coordinate space.
            img2_canvas (np.ndarray): Destination image expanded to the identical canvas coordinate space.

        Returns:
            np.ndarray: Blended output panorama image.
        """
        # Create non-zero binary masks for overlapping content
        mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0) if warped_img1.ndim == 3 else (warped_img1 > 0)
        mask2 = (cv2.cvtColor(img2_canvas, cv2.COLOR_BGR2GRAY) > 0) if img2_canvas.ndim == 3 else (img2_canvas > 0)

        overlap = np.logical_and(mask1, mask2)
        blended = np.zeros_like(warped_img1, dtype=np.uint8)

        # 1. Regions with content only in warped_img1
        only_1 = np.logical_and(mask1, np.logical_not(overlap))
        blended[only_1] = warped_img1[only_1]

        # 2. Regions with content only in img2_canvas
        only_2 = np.logical_and(mask2, np.logical_not(overlap))
        blended[only_2] = img2_canvas[only_2]

        # 3. Overlap regions: Apply smooth 50/50 linear blending
        if np.any(overlap):
            if warped_img1.ndim == 3:
                overlap_3d = np.repeat(overlap[:, :, np.newaxis], 3, axis=2)
                blended[overlap_3d] = cv2.addWeighted(
                    warped_img1, 0.5, img2_canvas, 0.5, 0.0
                )[overlap_3d]
            else:
                blended[overlap] = cv2.addWeighted(
                    warped_img1, 0.5, img2_canvas, 0.5, 0.0
                )[overlap]

        logger.debug("Successfully performed linear alpha region blending")
        return blended

    def stitch_pair(
        self,
        img_src: np.ndarray,
        img_dst: np.ndarray,
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> np.ndarray:
        """Stitches a source image onto a destination image frame given feature point correspondences.

        Args:
            img_src (np.ndarray): Source image array to warp.
            img_dst (np.ndarray): Reference destination image array.
            src_pts (np.ndarray): Source matching keypoint coordinates.
            dst_pts (np.ndarray): Destination matching keypoint coordinates.

        Returns:
            np.ndarray: Stitched combined panorama output canvas.
        """
        self.validate_image(img_src)
        self.validate_image(img_dst)

        # Estimate Homography mapping src coordinates to dst coordinates
        H, _ = self.estimate_homography(src_pts, dst_pts)

        h_src, w_src = img_src.shape[:2]
        h_dst, w_dst = img_dst.shape[:2]

        # Compute output bounding canvas size to accommodate both images
        # Define corner coordinates of the source image
        src_corners = np.array(
            [[0, 0], [0, h_src], [w_src, h_src], [w_src, 0]], dtype=np.float32
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(src_corners, H)

        dst_corners = np.array(
            [[0, 0], [0, h_dst], [w_dst, h_dst], [w_dst, 0]], dtype=np.float32
        ).reshape(-1, 1, 2)

        all_corners = np.concatenate((warped_corners, dst_corners), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation_dist = [-x_min, -y_min]
        H_translation = np.array(
            [[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]],
            dtype=np.float32,
        )

        canvas_width = x_max - x_min
        canvas_height = y_max - y_min

        # Warp source image onto canvas space incorporating translational offset
        warped_src = cv2.warpPerspective(img_src, H_translation.dot(H), (canvas_width, canvas_height))

        # Position reference destination image on the identical expanded canvas space
        dst_canvas = np.zeros_like(warped_src)
        dst_canvas[
            translation_dist[1] : h_dst + translation_dist[1],
            translation_dist[0] : w_dst + translation_dist[0],
        ] = img_dst

        # Blend both canvas layers
        panorama = self.blend_linear(warped_src, dst_canvas)
        return panorama

    def process(self, image: np.ndarray, mode: str = "stitch", **kwargs: Any) -> Any:
        """Executes image registration and stitching routines.

        Args:
            image (np.ndarray): Source input image array.
            mode (str): Execution mode ('homography', 'warp', 'stitch').
            **kwargs (Any): Mode arguments (dst_image, src_points, dst_points, H_matrix, output_size).

        Returns:
            Any: Homography matrix, warped array, or final panorama.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower in ["homography", "ransac"]:
            src_pts = kwargs.get("src_points")
            dst_pts = kwargs.get("dst_points")
            if src_pts is None or dst_pts is None:
                raise ValueError("Mode 'homography' requires 'src_points' and 'dst_points'.")
            return self.estimate_homography(src_pts, dst_pts)

        elif mode_lower in ["warp", "perspective"]:
            H = kwargs.get("H")
            output_size = kwargs.get("output_size", (image.shape[1], image.shape[0]))
            if H is None:
                raise ValueError("Mode 'warp' requires Homography matrix 'H'.")
            return self.warp_perspective(image, H, output_size)

        elif mode_lower == "stitch":
            dst_img = kwargs.get("dst_image")
            src_pts = kwargs.get("src_points")
            dst_pts = kwargs.get("dst_points")
            if dst_img is None or src_pts is None or dst_pts is None:
                raise ValueError("Mode 'stitch' requires 'dst_image', 'src_points', and 'dst_points'.")
            return self.stitch_pair(image, dst_img, src_pts, dst_pts)

        else:
            logger.warning(f"Unknown mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    stitcher = ImageStitcher()

    # Generate synthetic camera pair with synthetic feature point correspondences
    img_left = np.zeros((300, 300, 3), dtype=np.uint8)
    img_right = np.zeros((300, 300, 3), dtype=np.uint8)

    # Draw geometric features on left target
    cv2.rectangle(img_left, (50, 50), (250, 250), (0, 255, 0), 2)
    cv2.circle(img_left, (150, 150), 40, (255, 0, 0), -1)

    # Draw identical geometric features shifted on right target
    cv2.rectangle(img_right, (20, 50), (220, 250), (0, 255, 0), 2)
    cv2.circle(img_right, (120, 150), 40, (255, 0, 0), -1)

    # Synthetic point correspondences (Right perspective translated -30px on X)
    pts_left = np.array([[50, 50], [250, 50], [250, 250], [50, 250]], dtype=np.float32)
    pts_right = np.array([[20, 50], [220, 50], [220, 250], [20, 250]], dtype=np.float32)

    # 1. Estimate Homography matrix using synthetic point pairs
    (H_mat, inlier_mask), t_homo = stitcher.execute_with_timing(
        img_left, mode="homography", src_points=pts_left, dst_points=pts_right
    )

    # 2. Execute full image pair registration & stitching pipeline
    panorama_result, t_stitch = stitcher.execute_with_timing(
        img_left, mode="stitch", dst_image=img_right, src_points=pts_left, dst_points=pts_right
    )

    logger.info(f"Estimated Homography Matrix in {t_homo:.3f} ms:\n{H_mat}")
    logger.info(f"Stitched Panorama shape: {panorama_result.shape} in {t_stitch:.3f} ms")