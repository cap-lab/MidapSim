import logging


def init_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter("[ %(levelname)8s | %(filename)s:%(lineno)3d ]  %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
