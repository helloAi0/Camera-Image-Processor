from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

from src.utils.logger import setup_logger

logger = setup_logger("ConfigParser")


@dataclass(frozen=True)
class AppConfig:
    """Application metadata and logging configuration."""
    name: str = "Camera Image Processor"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_file: Path = Path("output/app.log")


@dataclass(frozen=True)
class IOConfig:
    """Input/Output file system and extension parameters."""
    input_dir: Path = Path("sample_images")
    output_dir: Path = Path("output")
    supported_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


@dataclass(frozen=True)
class EnhancementConfig:
    """Image enhancement and histogram processing parameters."""
    contrast_alpha: float = 1.0
    brightness_beta: int = 0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class EdgeDetectionConfig:
    """Edge detection algorithm kernel sizes and threshold parameters."""
    canny_threshold1: float = 100.0
    canny_threshold2: float = 200.0
    sobel_ksize: int = 3
    laplacian_ksize: int = 3


@dataclass(frozen=True)
class SegmentationConfig:
    """Thresholding and image segmentation parameters."""
    global_threshold: float = 127.0
    max_value: float = 255.0
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0


@dataclass(frozen=True)
class PipelineConfig:
    """Master configuration container uniting all module configurations."""
    app: AppConfig = field(default_factory=AppConfig)
    io: IOConfig = field(default_factory=IOConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    edge_detection: EdgeDetectionConfig = field(default_factory=EdgeDetectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)


class ConfigParser:
    """Parses, validates, and manages YAML application configurations."""

    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        """Initializes ConfigParser with a target YAML file path.

        Args:
            config_path (str | Path): Path to the config.yaml file.
        """
        self.config_path = Path(config_path)
        self.config: PipelineConfig = self._load_and_parse()

    def _load_and_parse(self) -> PipelineConfig:
        """Reads YAML from disk and builds strongly typed configuration objects.

        Returns:
            PipelineConfig: Fully parsed and populated configuration object.
        """
        if not self.config_path.exists():
            logger.warning(
                f"Config file not found at '{self.config_path}'. Falling back to default settings."
            )
            return PipelineConfig()

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_data: Dict[str, Any] = yaml.safe_load(file) or {}

            logger.info(f"Successfully loaded configuration from '{self.config_path}'")
            return self._build_pipeline_config(raw_data)

        except yaml.YAMLError as err:
            logger.error(f"Error parsing YAML config file '{self.config_path}': {err}")
            logger.warning("Falling back to default pipeline configuration.")
            return PipelineConfig()
        except Exception as err:
            logger.error(f"Unexpected error loading config '{self.config_path}': {err}")
            return PipelineConfig()

    def _build_pipeline_config(self, raw_data: Dict[str, Any]) -> PipelineConfig:
        """Maps raw YAML dictionaries to strongly typed frozen dataclasses.

        Args:
            raw_data (Dict[str, Any]): Dictionary loaded from YAML file.

        Returns:
            PipelineConfig: Instantiated pipeline configuration.
        """
        app_data = raw_data.get("app", {})
        io_data = raw_data.get("io", {})
        enh_data = raw_data.get("enhancement", {})
        edge_data = raw_data.get("edge_detection", {})
        seg_data = raw_data.get("segmentation", {})

        app_cfg = AppConfig(
            name=app_data.get("name", "Camera Image Processor"),
            version=app_data.get("version", "1.0.0"),
            log_level=app_data.get("log_level", "INFO"),
            log_file=Path(app_data.get("log_file", "output/app.log")),
        )

        io_cfg = IOConfig(
            input_dir=Path(io_data.get("input_dir", "sample_images")),
            output_dir=Path(io_data.get("output_dir", "output")),
            supported_extensions=tuple(
                io_data.get(
                    "supported_extensions",
                    [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                )
            ),
        )

        enh_cfg = EnhancementConfig(
            contrast_alpha=float(enh_data.get("contrast_alpha", 1.0)),
            brightness_beta=int(enh_data.get("brightness_beta", 0)),
            clahe_clip_limit=float(enh_data.get("clahe_clip_limit", 2.0)),
            clahe_tile_grid_size=tuple(enh_data.get("clahe_tile_grid_size", [8, 8])),
        )

        edge_cfg = EdgeDetectionConfig(
            canny_threshold1=float(edge_data.get("canny_threshold1", 100.0)),
            canny_threshold2=float(edge_data.get("canny_threshold2", 200.0)),
            sobel_ksize=int(edge_data.get("sobel_ksize", 3)),
            laplacian_ksize=int(edge_data.get("laplacian_ksize", 3)),
        )

        seg_cfg = SegmentationConfig(
            global_threshold=float(seg_data.get("global_threshold", 127.0)),
            max_value=float(seg_data.get("max_value", 255.0)),
            adaptive_block_size=int(seg_data.get("adaptive_block_size", 11)),
            adaptive_c=float(seg_data.get("adaptive_c", 2.0)),
        )

        return PipelineConfig(
            app=app_cfg,
            io=io_cfg,
            enhancement=enh_cfg,
            edge_detection=edge_cfg,
            segmentation=seg_cfg,
        )


if __name__ == "__main__":
    # Test execution module verification
    parser = ConfigParser("config.yaml")
    cfg = parser.config

    logger.debug(f"App Name: {cfg.app.name}")
    logger.debug(f"Input Directory: {cfg.io.input_dir.resolve()}")
    logger.debug(f"CLAHE Tile Grid Size: {cfg.enhancement.clahe_tile_grid_size}")
    logger.debug(f"Canny Threshold 1: {cfg.edge_detection.canny_threshold1}")
    logger.debug(f"Global Threshold: {cfg.segmentation.global_threshold}")