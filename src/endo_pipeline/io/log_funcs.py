import logging
from pathlib import Path


def configure_logging(out_dir: Path, logger: logging.Logger, verbose: bool = True) -> None:
    logging.basicConfig(
        filename=out_dir / f"{out_dir.name}.log",
        filemode="a",  # append to the log file instead of rewriting it
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,  # sets the level to be recorded in log file
    )

    # Add the output handler to the logger
    output_handler = logging.StreamHandler()
    # Set the verbosity to be printed
    if verbose:
        output_handler.setLevel(logging.INFO)
    else:
        output_handler.setLevel(logging.WARNING)
    logger.addHandler(output_handler)  # needed to print in ipython
