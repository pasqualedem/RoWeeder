import colorlog
import logging


def get_logger(name):
    name = name.split('.')[-1]
    # Create a StreamHandler with colorlog formatter
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-2s%(reset)s %(blue)s%(asctime)s%(reset)s %(purple)s[%(name)s]%(reset)s %(message)s",
        datefmt='[%m-%d %H:%M:%S]',
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        style='%'
    )
    handler.setFormatter(formatter)

    # Create a logger with the specified name
    logger = colorlog.getLogger(name)

    # Check if the logger already has handlers, if yes, clear them
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set logging level to DEBUG
    logger.setLevel(logging.DEBUG)

    # Add the StreamHandler to the logger
    logger.addHandler(handler)

    return logger

