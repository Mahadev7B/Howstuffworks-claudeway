"""Gunicorn error logger that does not write request query strings.

Worker.handle_error logs req.uri, and a kid's question can sit in ?q=.
"""
import re

from gunicorn.glogging import Logger

_QUERY = re.compile(r"\?[^\s\"']*")


def strip_query(value):
    """Drop a request query string from a log message or argument.

    Gunicorn logs req.uri. A question in ?q= is removed, including when the
    question contains spaces. The path is kept.
    """
    if not isinstance(value, str) or "?" not in value:
        return value
    if " HTTP/" in value:
        return re.sub(r"\?.*?(?= HTTP/)", "", value)
    head, _, _ = value.partition("?")
    if " " not in head:
        return head
    return _QUERY.sub("", value)


def _redact(msg, args):
    return strip_query(msg), tuple(strip_query(arg) for arg in args)


class RedactingLogger(Logger):
    """Same as Gunicorn's logger, with query strings removed from error lines."""

    def critical(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().critical(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().error(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().warning(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().info(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().debug(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().exception(msg, *args, **kwargs)

    def log(self, lvl, msg, *args, **kwargs):
        msg, args = _redact(msg, args)
        super().log(lvl, msg, *args, **kwargs)
