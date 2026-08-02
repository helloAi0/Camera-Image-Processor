import logging
import sys
from pathlib import Path
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom ANSI color formatter for terminal log output."""

    # ANSI Escape Sequences for Terminal Colors
    GREY: str = "\x1b[38;20m"
    GREEN: str = "\x1b[32;20m"
    YELLOW: str = "\x1b[33;20m"
    RED: str = "\x1b[31;20m"
    BOLD_RED: str = "\x1b[31;1m"
    RESET: str = "\x1b[0m"

    LOG_FORMAT: str = (
        "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: GREY + LOG_FORMAT + RESET,
        logging.INFO: GREEN + LOG_FORMAT + RESET,
        logging.WARNING: YELLOW + LOG_FORMAT + RESET,
        logging.ERROR: RED + LOG_FORMAT + RESET,
        logging.CRITICAL: BOLD_RED + LOG_FORMAT + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Formats log record with ANSI color coding based on severity level.

        Args:
            record (logging.LogRecord): The log record event.

        Returns:
            str: Colorized log string for terminal display.
        """
        log_fmt = self.FORMATS.get(record.levelno, self.LOG_FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str = "CameraProcessor",
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Creates, configures, and returns a thread-safe logger instance.

    Args:
        name (str): The name identifier for the logger instance.
        log_level (int): Logging severity threshold (e.g., logging.INFO, logging.DEBUG).
        log_file (Optional[Path]): File system path to write log files to.

    Returns:
        logging.Logger: Fully configured Python Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if the logger has already been initialized
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)

    # 1. Terminal Console Handler (Colorized Output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # 2. File Handler (Clean Uncolored Output for Disk Persistence)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Internal module validation test
    test_log_path = Path("output/app.log")
    log = setup_logger("TestLogger", log_level=logging.DEBUG, log_file=test_log_path)

    log.debug("DEBUG: Initializing camera driver pipeline...")
    log.info("INFO: Camera image processor started successfully.")
    log.warning("WARNING: Frame rate dropped below 30 FPS!")
    log.error("ERROR: Failed to acquire matrix handle from image buffer.")
    log.critical("CRITICAL: Hardware sensor thermal shutdown triggered!")