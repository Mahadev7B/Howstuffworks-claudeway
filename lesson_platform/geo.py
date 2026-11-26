"""Best-effort IP geolocation via ipwho.is (free, no API key, HTTPS)."""
import ipaddress
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_TIMEOUT_S = 2.0


def _looks_like_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _client_ip(forwarded_for: str | None, remote_addr: str | None,
               cf_connecting_ip: str | None = None) -> str | None:
    """The visitor address, not a client-supplied earlier hop.

    Cloudflare overwrites CF-Connecting-IP, so that wins when the request came
    through Cloudflare. Otherwise the rightmost valid X-Forwarded-For entry is
    the one Render appended. The leftmost entry is whatever the visitor sent.
    """
    cf = (cf_connecting_ip or "").strip()
    if _looks_like_ip(cf):
        return cf
    parts = [part.strip() for part in (forwarded_for or "").split(",") if part.strip()]
    for candidate in reversed(parts):
        if _looks_like_ip(candidate):
            return candidate
    if remote_addr and _looks_like_ip(remote_addr):
        return remote_addr
    return None


def _is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
        return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved)
    except ValueError:
        return False


def lookup(ip: str | None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fresh lookup every call — returns (city, region, country). Any field may be None."""
    if not ip or not _is_public(ip):
        return (None, None, None)
    try:
        req = urllib.request.Request(
            f"https://ipwho.is/{ip}?fields=success,city,region,country",
            headers={"User-Agent": "asklilowl/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("success"):
            return (data.get("city"), data.get("region"), data.get("country"))
    except Exception:
        logger.warning("Geo lookup failed", exc_info=False)
    return (None, None, None)


def extract_and_lookup(
    forwarded_for: str | None, remote_addr: str | None, cf_connecting_ip: str | None = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Returns (ip, city, region, country)."""
    ip = _client_ip(forwarded_for, remote_addr, cf_connecting_ip)
    city, region, country = lookup(ip)
    return (ip, city, region, country)
