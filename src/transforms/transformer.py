import logging
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.core.base_processor import BaseProcessor
from src.utils.logger import setup_logger

logger = setup_logger("ImageTransformer", log_level=logging.DEBUG)


class ImageTransformer(BaseProcessor):
    """Production Image Transformation Processor for spatial scaling, rotation, flipping, and perspective warping."""

    def __init__(self) -> None:
        """Initializes the ImageTransformer processor."""
        super().__init__(name="ImageTransformer")

    def resize(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation_mode: str = "bilinear",
    ) -> np.ndarray:
        """Resizes an image array to target (Width, Height) using mathematically optimal interpolation.

        Args:
            image (np.ndarray): Input NumPy image matrix (H, W, C) or (H, W).
            target_size (Tuple[int, int]): Target dimensions as (Width, Height).
            interpolation_mode (str): Interpolation method: 'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'.

        Returns:
            np.ndarray: Resized image matrix with target dimensions.
        """
        self.validate_image(image)

        width, height = target_size
        if width <= 0 or height <= 0:
            raise ValueError(f"Target width and height must be > 0. Got ({width}, {height})")

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }

        cv2_interp = interp_map.get(interpolation_mode.lower(), cv2.INTER_LINEAR)
        resized = cv2.resize(image, (width, height), interpolation=cv2_interp)
        return resized

    def crop(
        self,
        image: np.ndarray,
        crop_box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crops a rectangular sub-region from an image using zero-copy NumPy array slicing.

        Args:
            image (np.ndarray): Input image array.
            crop_box (Tuple[int, int, int, int]): Bounding box as (x_min, y_min, x_max, y_max).

        Returns:
            np.ndarray: Cropped image sub-matrix.
        """
        self.validate_image(image)

        img_height, img_width = image.shape[:2]
        x_min, y_min, x_max, y_max = crop_box

        # Clamp bounding box boundaries safely to image dimensions
        x_min = max(0, min(x_min, img_width - 1))
        y_min = max(0, min(y_min, img_height - 1))
        x_max = max(x_min + 1, min(x_max, img_width))
        y_max = max(y_min + 1, min(y_max, img_height))

        # Perform fast NumPy 2D array slicing: [row_start:row_end, col_start:col_end]
        cropped = image[y_min:y_max, x_min:x_max].copy()
        return cropped

    def rotate(
        self,
        image: np.ndarray,
        angle_degrees: float,
        center: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Rotates an image around a designated center coordinate via affine transformation matrix.

        Args:
            image (np.ndarray): Input image array.
            angle_degrees (float): Rotation angle in degrees (counter-clockwise).
            center (Optional[Tuple[int, int]]): Anchor point (X, Y). Defaults to image center.
            scale (float): Isotropic scaling factor. Default is 1.0.

        Returns:
            np.ndarray: Rotated image matrix.
        """
        self.validate_image(image)

        img_height, img_width = image.shape[:2]

        if center is None:
            center = (img_width // 2, img_height // 2)

        # Compute 2x3 Affine Rotation Matrix: M = [ [alpha, beta, (1-alpha)*cx - beta*cy], [-beta, alpha, beta*cx + (1-alpha)*cy] ]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, scale)

        # Warp image coordinates using affine matrix transformation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (img_width, img_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return rotated

    def flip(self, image: np.ndarray, flip_code: int = 1) -> np.ndarray:
        """Flips an image matrix along horizontal, vertical, or dual axes.

        Args:
            image (np.ndarray): Input image matrix.
            flip_code (int): 1 = Horizontal flip (Y-axis mirror),
                             0 = Vertical flip (X-axis mirror),
                            -1 = Dual-axis flip (Both axes).

        Returns:
            np.ndarray: Flipped image array.
        """
        self.validate_image(image)

        if flip_code not in (-1, 0, 1):
            logger.warning(f"Invalid flip_code {flip_code}. Defaulting to horizontal flip (1).")
            flip_code = 1

        flipped = cv2.flip(image, flipCode=flip_code)
        return flipped

    def warp_perspective(
        self,
        image: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """Performs a 4-point perspective transform (homography) to project quadrilateral views.

        Args:
            image (np.ndarray): Input image matrix.
            src_points (np.ndarray): Float32 array of 4 source coordinates [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
            dst_points (np.ndarray): Float32 array of 4 target coordinates [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
            output_size (Tuple[int, int]): Output dimensions as (Width, Height).

        Returns:
            np.ndarray: Perspective-corrected orthographic image array.
        """
        self.validate_image(image)

        if src_points.shape != (4, 2) or dst_points.shape != (4, 2):
            raise ValueError("Perspective transform requires source and destination points of shape (4, 2).")

        # Convert coordinates to float32 precision required by OpenCV C++ bindings
        src_pts = src_points.astype(np.float32)
        dst_pts = dst_points.astype(np.float32)

        # Calculate 3x3 Homography Matrix H
        homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        out_w, out_h = output_size
        warped = cv2.warpPerspective(
            image,
            homography_matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return warped

    def process(self, image: np.ndarray, mode: str = "resize", **kwargs: Any) -> np.ndarray:
        """Executes spatial transformation based on mode selection.

        Args:
            image (np.ndarray): Input NumPy image matrix.
            mode (str): Mode: 'resize', 'crop', 'rotate', 'flip', 'perspective'.
            **kwargs (Any): Arguments passed to specific transformation routines.

        Returns:
            np.ndarray: Transformed image matrix.
        """
        self.validate_image(image)
        mode_lower = mode.lower()

        if mode_lower == "resize":
            target_size = kwargs.get("target_size", (640, 480))
            interp = kwargs.get("interpolation_mode", "bilinear")
            return self.resize(image, target_size=target_size, interpolation_mode=interp)

        elif mode_lower == "crop":
            h, w = image.shape[:2]
            crop_box = kwargs.get("crop_box", (0, 0, w // 2, h // 2))
            return self.crop(image, crop_box=crop_box)

        elif mode_lower == "rotate":
            angle = kwargs.get("angle_degrees", 45.0)
            center = kwargs.get("center", None)
            scale = kwargs.get("scale", 1.0)
            return self.rotate(image, angle_degrees=angle, center=center, scale=scale)

        elif mode_lower == "flip":
            code = kwargs.get("flip_code", 1)
            return self.flip(image, flip_code=code)

        elif mode_lower == "perspective":
            src = kwargs.get("src_points")
            dst = kwargs.get("dst_points")
            out_sz = kwargs.get("output_size", (400, 400))
            if src is None or dst is None:
                logger.error("Perspective mode requires 'src_points' and 'dst_points' parameters.")
                return image
            return self.warp_perspective(image, src_points=src, dst_points=dst, output_size=out_sz)

        else:
            logger.warning(f"Unknown transform mode '{mode}'. Returning original image.")
            return image


if __name__ == "__main__":
    # Module self-verification test
    transformer = ImageTransformer()

    # Generate synthetic RGB test matrix (400x400)
    test_img = np.zeros((400, 400, 3), dtype=np.uint8)
    test_img[100:300, 100:300] = [0, 255, 0]  # Green central block

    # 1. Test Resizing (Area vs Lanczos)
    resized_img, t_resize = transformer.execute_with_timing(
        test_img, mode="resize", target_size=(200, 200), interpolation_mode="area"
    )

    # 2. Test Cropping
    cropped_img, t_crop = transformer.execute_with_timing(
        test_img, mode="crop", crop_box=(100, 100, 300, 300)
    )

    # 3. Test Rotation (45 degrees)
    rotated_img, t_rot = transformer.execute_with_timing(
        test_img, mode="rotate", angle_degrees=45.0, scale=1.0
    )

    # 4. Test Flipping
    flipped_img, t_flip = transformer.execute_with_timing(
        test_img, mode="flip", flip_code=1
    )

    # 5. Test Perspective Warping
    src_quad = np.array([[50, 50], [350, 30], [380, 370], [20, 350]], dtype=np.float32)
    dst_quad = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)
    warped_img, t_warp = transformer.execute_with_timing(
        test_img, mode="perspective", src_points=src_quad, dst_points=dst_quad, output_size=(300, 300)
    )

    logger.info(f"Resize (Area) completed in: {t_resize:.3f} ms - Output Shape: {resized_img.shape}")
    logger.info(f"Crop completed in: {t_crop:.3f} ms - Output Shape: {cropped_img.shape}")
    logger.info(f"Rotation (45 deg) completed in: {t_rot:.3f} ms - Output Shape: {rotated_img.shape}")
    logger.info(f"Flip completed in: {t_flip:.3f} ms - Output Shape: {flipped_img.shape}")
    logger.info(f"Perspective Warp completed in: {t_warp:.3f} ms - Output Shape: {warped_img.shape}")