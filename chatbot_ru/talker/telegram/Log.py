import logging
from logging.handlers import RotatingFileHandler


def get_logger(logfile='/var/log/telegram_bot.log'):
    """
	Provide logger object bind to a file.
	"""
    logging.basicConfig(
        level=logging.INFO,
        format="[TALKER][%(asctime)s][module=%(module)s][thread=%(threadName)s] %(levelname)s: %(message)s",
    )
    # Different level for POST calls
    logging.getLogger("requests").setLevel(logging.WARNING)
    # For every module write to the same logfile
    logger = logging.getLogger(__name__)
    handler = RotatingFileHandler(logfile, maxBytes=2000000,
                                  backupCount=5)
    formatter = logging.Formatter(
        "[TALKER][%(asctime)s][module=%(module)s][thread=%(threadName)s] %(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def debug_mode_on():
    logging.root.setLevel(logging.DEBUG)
    logging.getLogger("requests").setLevel(logging.WARNING)


def debug_mode_off():
    logging.root.setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Add in every function this
# logging.debug('Function called: %s\n
# 					Values passed: (%s)' % (f.__name__, parameters))
