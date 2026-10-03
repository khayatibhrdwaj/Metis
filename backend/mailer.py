"""
Metis Backend - SMTP Mailer

Sends messages from the "Contact Us" chat box via real SMTP. This is a
REQUIRED feature, not decorative - if SMTP isn't configured, /api/contact
returns a clear setup error instead of silently pretending the message
was sent.
"""

import os
import re
import smtplib
import socket
from email.mime.text import MIMEText

CONTACT_RECIPIENT = os.environ.get("METIS_CONTACT_RECIPIENT", "khayati.bhrdwaj15@gmail.com")
MAX_MESSAGE_LENGTH = 5000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class MailerConfigError(Exception):
    """Raised when the mailer can't send because SMTP isn't configured."""


def _smtp_config():
    def _clean(v):
        return v.strip().strip('"').strip("'") if v else v

    host = _clean(os.environ.get("SMTP_HOST"))
    port = _clean(os.environ.get("SMTP_PORT", "587"))
    user = _clean(os.environ.get("SMTP_USER"))
    password = _clean(os.environ.get("SMTP_PASSWORD"))
    use_ssl = os.environ.get("SMTP_USE_SSL", "false").lower() in ("1", "true", "yes")
    if not all([host, user, password]):
        raise MailerConfigError(
            "Email isn't configured yet. Set SMTP_HOST, SMTP_USER, and "
            "SMTP_PASSWORD (in backend/.env or as real environment "
            "variables) and restart the backend - see README.md."
        )
    try:
        port_int = int(port)
    except ValueError:
        raise MailerConfigError(f"SMTP_PORT must be a number, got: {port!r}")
    return host, port_int, user, password, use_ssl


def send_contact_message(name: str, email: str, message: str) -> dict:
    name = (name or "").strip()[:200]
    email = (email or "").strip()[:200]
    message = (message or "").strip()

    if not message:
        return {"success": False, "error": "Message can't be empty."}
    if len(message) > MAX_MESSAGE_LENGTH:
        return {"success": False, "error": f"Message is too long (max {MAX_MESSAGE_LENGTH} characters)."}
    if email and not _EMAIL_RE.match(email):
        return {"success": False, "error": "That doesn't look like a valid email address."}

    try:
        host, port, user, password, use_ssl = _smtp_config()
    except MailerConfigError as e:
        return {"success": False, "error": str(e)}

    body = (
        f"New message from the Metis Contact Us page:\n\n"
        f"Name: {name or '(not provided)'}\n"
        f"Reply-to email: {email or '(not provided)'}\n\n"
        f"Message:\n{message}"
    )
    msg = MIMEText(body)
    msg["Subject"] = f"Metis contact form: {name or 'Anonymous visitor'}"
    msg["From"] = user
    msg["To"] = CONTACT_RECIPIENT
    if email:
        msg["Reply-To"] = email

    try:
        smtp_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
        with smtp_cls(host, port, timeout=10) as server:
            if not use_ssl:
                server.starttls()
            server.login(user, password)
            server.sendmail(user, [CONTACT_RECIPIENT], msg.as_string())
        return {"success": True, "error": None}
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "error": "SMTP login failed - check SMTP_USER/SMTP_PASSWORD. For Gmail this must "
                     "be an app password, not your normal account password.",
        }
    except (socket.timeout, ConnectionRefusedError, socket.gaierror) as e:
        return {"success": False, "error": f"Could not reach the SMTP server ({host}:{port}): {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to send email: {e}"}
