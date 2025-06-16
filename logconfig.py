import logging

logger = logging.getLogger("giqpy")

def setup_logging(logfile: str) -> None:
    """Configure logging with both console and file handlers."""
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)


def write_to_log(message: str, is_error: bool = False, is_warning: bool = False) -> None:
    """Wrapper used by the main code to record log messages."""
    if is_error:
        logger.error(message)
    elif is_warning:
        logger.warning(message)
    else:
        logger.info(message)
