"""Best-effort IP geolocation via ipwho.is (free, no API key, HTTPS)."""
import ipaddress
import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_TIMEOUT_S = 2.0


def _client_ip(forwarded_for: str | None, remote_addr: str | None) -> str | None:
    """Pick the leftmost IP from X-Forwarded-For, else remote_addr."""
    if forwarded_for:
        for part in forwarded_for.split(","):
            ip = part.strip()
            if ip:
                return ip
    return remote_addr or None


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
            headers={"User-Agent": "howstuffworks-claudeway/1.0"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("success"):
            return (data.get("city"), data.get("region"), data.get("country"))
    except Exception:
        logger.warning("Geo lookup failed for %s", ip, exc_info=False)
    return (None, None, None)


def extract_and_lookup(
    forwarded_for: str | None, remote_addr: str | None
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Returns (ip, city, region, country)."""
    ip = _client_ip(forwarded_for, remote_addr)
    city, region, country = lookup(ip)
    return (ip, city, region, country)
