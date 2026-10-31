"""One-shot official-origin HTTPS upload with read-only uncertain-outcome recovery."""
import http.client
import json
from urllib.parse import urlsplit

from .private_import_protocol import MAX_PACKET_BYTES, valid_fingerprint, valid_receipt, valid_slug, valid_token


class PrivateImportError(RuntimeError):
    """A safe local message; never includes provider bodies, credentials or URLs."""


class UploadUncertain(PrivateImportError):
    """The single POST was not confirmed; do not blindly submit it again."""


class PrivateImportClient:
    def __init__(self, base_url, token):
        if not isinstance(base_url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in base_url):
            raise ValueError("Use an official HTTPS origin")
        try:
            parsed = urlsplit(base_url)
            valid = (parsed.scheme == "https" and parsed.hostname in ("asklilowl.com", "www.asklilowl.com")
                     and parsed.port in (None, 443) and parsed.username is None and parsed.password is None
                     and parsed.path in ("", "/") and not parsed.query and not parsed.fragment
                     and "?" not in base_url and "#" not in base_url and "\\" not in base_url)
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("Use an official HTTPS origin")
        if not valid_token(token):
            raise ValueError("A valid dedicated import token is required")
        self._host = parsed.hostname
        self._token = token

    def _request(self, method, path, raw=None):
        connection = http.client.HTTPSConnection(self._host, 443, timeout=60)
        try:
            headers = {"Authorization": "Bearer " + self._token, "Accept": "application/json"}
            if raw is not None:
                headers["Content-Type"] = "application/json"
                headers["Content-Length"] = str(len(raw))
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            status = response.status
            # No redirect handling exists here: the credential is sent to this exact origin once.
            if status in (404, 400, 401, 403, 409, 411, 413, 415):
                return status, None
            if not 200 <= status < 300:
                raise PrivateImportError("Private import service did not confirm the request")
            body = response.read(64 * 1024 + 1)
            if len(body) > 64 * 1024:
                raise PrivateImportError("Private import service returned an invalid receipt")
            return status, json.loads(body)
        except PrivateImportError:
            raise
        except Exception:
            raise PrivateImportError("Private import connection did not confirm the request") from None
        finally:
            connection.close()

    def lookup(self, slug):
        if not valid_slug(slug):
            raise ValueError("Use a valid private lesson slug")
        status, receipt = self._request("GET", "/api/admin/private-lessons/" + slug)
        if status == 404:
            return None
        if status != 200 or not valid_receipt(receipt, slug):
            raise PrivateImportError("Private import service did not return a complete private receipt")
        return receipt

    def upload(self, raw, slug, expected_fingerprint):
        if not isinstance(raw, bytes) or not raw or len(raw) > MAX_PACKET_BYTES:
            raise ValueError("Use one JSON package no larger than 12 MiB")
        if not valid_slug(slug) or not valid_fingerprint(expected_fingerprint):
            raise ValueError("Use a valid slug and expected content fingerprint")
        try:
            status, receipt = self._request("POST", "/api/admin/private-lessons", raw)
        except PrivateImportError:
            status, receipt = None, None
        if status in (400, 401, 403, 409, 411, 413, 415):
            raise PrivateImportError("Private import was rejected; no automatic retry was attempted")
        try:
            confirmed = self.lookup(slug)
            if valid_receipt(confirmed, slug, expected_fingerprint):
                return confirmed
        except PrivateImportError:
            pass
        raise UploadUncertain("Upload outcome is unconfirmed. Reconcile the private receipt before any explicit retry.") from None
