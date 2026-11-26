"""Keep raw network addresses out of logs and error-tracker events.

The address is allowed in memory only long enough to hash it and to decide a
country. These helpers redact address-shaped text from log records and from
an error-tracker event payload. The app does not ship an error tracker; the
scrubber is here so one can be attached without writing the address through.
"""
import logging
import re

_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
_IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\b")
_ADDRESS_HEADERS = {
    "x-forwarded-for",
    "cf-connecting-ip",
    "true-client-ip",
    "x-real-ip",
    "forwarded",
    "remote-addr",
    "remote_addr",
}

_installed = False
_previous_factory = logging.getLogRecordFactory()


def redact_text(value: str) -> str:
    return _IPV6.sub("[redacted]", _IPV4.sub("[redacted]", value))


def _redact_record(record: logging.LogRecord) -> logging.LogRecord:
    if isinstance(record.msg, str):
        record.msg = redact_text(record.msg)
    if record.args:
        if isinstance(record.args, dict):
            record.args = {key: redact_text(val) if isinstance(val, str) else val for key, val in record.args.items()}
        else:
            record.args = tuple(redact_text(arg) if isinstance(arg, str) else arg for arg in record.args)
    return record


def install_address_redaction() -> None:
    """Redact addresses in every log record as it is created."""
    global _installed
    if _installed:
        return

    def factory(*args, **kwargs):
        return _redact_record(_previous_factory(*args, **kwargs))

    logging.setLogRecordFactory(factory)
    _installed = True


def _redact_value(value):
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            name = str(key).lower()
            if name in _ADDRESS_HEADERS or name.replace("_", "-") in _ADDRESS_HEADERS:
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = _redact_value(item)
        return cleaned
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def scrub_sentry_event(event: dict, hint=None) -> dict:
    """Drop address headers and address-shaped strings from an error-tracker event."""
    if not isinstance(event, dict):
        return event
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("env", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                if str(key).lower().replace("_", "-") in _ADDRESS_HEADERS or str(key).lower() in _ADDRESS_HEADERS:
                    headers[key] = "[redacted]"
    user = event.get("user")
    if isinstance(user, dict):
        for key in ("ip_address", "ip", "geo"):
            if key in user:
                user[key] = "[redacted]"
    return _redact_value(event)
