"""Send plain emails through any standard SMTP mail service.

Settings come from environment variables set in the Render dashboard, for
example a Gmail address with a Google "app password":

  SMTP_HOST      smtp.gmail.com
  SMTP_PORT      587 (STARTTLS) or 465 (SSL)
  SMTP_USER      Asklilowl@gmail.com
  SMTP_PASSWORD  the app password (never committed)
  MAIL_FROM      optional; defaults to SMTP_USER
"""
import logging
import os
import smtplib
from email.message import EmailMessage
from email.utils import formataddr

logger = logging.getLogger(__name__)

_TIMEOUT_S = 15


class MailUnavailable(RuntimeError):
    """Email isn't configured, or the mail service refused the message."""


def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def enabled() -> bool:
    return bool(_setting("SMTP_HOST") and _setting("SMTP_USER") and _setting("SMTP_PASSWORD"))


def send(to: str, subject: str, text: str) -> None:
    if not enabled():
        raise MailUnavailable("Email isn't set up yet.")
    host, user = _setting("SMTP_HOST"), _setting("SMTP_USER")
    port = int(_setting("SMTP_PORT", "587") or 587)
    message = EmailMessage()
    message["From"] = formataddr(("AskLilOwl", _setting("MAIL_FROM") or user))
    message["To"] = to
    message["Subject"] = subject
    message.set_content(text)
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=_TIMEOUT_S) as smtp:
                smtp.login(user, _setting("SMTP_PASSWORD"))
                smtp.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=_TIMEOUT_S) as smtp:
                smtp.starttls()
                smtp.login(user, _setting("SMTP_PASSWORD"))
                smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("Email could not be sent: %s", type(exc).__name__)
        raise MailUnavailable("We couldn't send the email just now. Please try again.") from exc
