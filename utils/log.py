# utils/log.py
# Debug-only output, switched on with [Logging] debug = true in hajiwo.ini.

DEBUG = False


def set_debug(enabled: bool) -> None:
    global DEBUG
    DEBUG = bool(enabled)


def dbg(*args, **kwargs) -> None:
    """print() only when debug mode is on."""
    if DEBUG:
        print(*args, **kwargs)
