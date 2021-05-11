import logging
import logging.config

ALL_LOGER = 'all_logger'#yaml
INFO_LOGGER = 'info_logger'#yaml


lgr = logging.getLogger(ALL_LOGER)
lgr_info = logging.getLogger(INFO_LOGGER)

def setup_logging():
    """Function for setup logging
    """
    simple_formatter = logging.Formatter(
        fmt="%(levelname)s: %(message)s"
    )

    file_handler = logging.FileHandler(
        filename="outputs\logs\heart_dissease_uci.log"
    )

    info_handler = logging.FileHandler(
        filename="outputs\logs\heart_dissease_uci.info"
    )

    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(simple_formatter)

    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(simple_formatter)

    lgr_info = logging.getLogger(INFO_LOGGER)
    lgr_info.setLevel(logging.INFO)
    lgr_info.addHandler(info_handler)

    lgr = logging.getLogger(ALL_LOGER)
    lgr.setLevel(logging.ERROR)
    lgr.addHandler(file_handler)

    return [lgr.level, lgr_info.level]