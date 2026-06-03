import logging
from datetime import UTC, datetime
from pathlib import Path

from tqdm.std import tqdm as std_tqdm

EXTERNAL_LOGGERS = {
    "aicsfiles.client.http.http_client": logging.WARNING,
    "cyto_dl": logging.ERROR,
    "fontTools.subset": logging.WARNING,
    "fsspec.local": logging.WARNING,
    "git.cmd": logging.WARNING,
    "h5py._conv": logging.WARNING,
    "lightning.pytorch": logging.WARNING,
    "lightning.pytorch.accelerators.cuda": logging.WARNING,
    "lightning.pytorch.utilities.rank_zero": logging.WARNING,
    "lightning.fabric.utilities": logging.WARNING,
    "numcodecs": logging.WARNING,
    "matplotlib": logging.ERROR,
    "torch": logging.WARNING,
    "urllib3.connectionpool": logging.WARNING,
}


class CustomStreamLoggingFormatter(logging.Formatter):
    """Custom class for formatting stream logging with colored levels."""

    def __init__(self) -> None:
        super().__init__()
        self.format_template = (
            "%(asctime)s - %(name)s - \033[COLORm%(levelname)s\033[0m - %(message)s"
        )
        self.formats = {
            logging.DEBUG: self.format_template.replace("COLOR", "37;1"),
            logging.INFO: self.format_template.replace("COLOR", "34;1"),
            logging.WARNING: self.format_template.replace("COLOR", "33;1"),
            logging.ERROR: self.format_template.replace("COLOR", "31;1"),
            logging.CRITICAL: self.format_template.replace("COLOR", "31;1;4"),
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format logging record with colored levels."""

        log_format = self.formats.get(record.levelno)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class CustomStreamHandler(logging.StreamHandler):
    """Custom class for routing stream through tqdm."""

    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.tqdm_class = std_tqdm

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, end=self.terminator, file=self.stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def setup_logging(level: int, create_directory: bool = True) -> None:
    """Set up logging handlers and assign logging levels."""

    logger = logging.getLogger("")
    logger.setLevel(logging.DEBUG)

    if create_directory:
        log_path = Path(__file__).resolve().parents[3] / "logs"
        log_path.mkdir(exist_ok=True)
        file_name = log_path / f"endo_pipeline_{datetime.now(tz=UTC).strftime('%Y%m%d')}.log"

        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    stream_formatter = CustomStreamLoggingFormatter()
    stream_handler = CustomStreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(stream_handler)


def silence_external_loggers(external_loggers: dict = EXTERNAL_LOGGERS) -> None:
    """
    Set external logger to a specific logging level to avoid excessive logging outputs.

    Parameters
    ----------
    external_loggers
        Dictionary of external loggers and their respective logging levels.
    """

    for logger_name, logging_level in external_loggers.items():
        external_logger = logging.getLogger(logger_name)
        external_logger.setLevel(logging_level)
