"""Visitor address, as Render's proxy records it. Country is not looked up on the network.

Render appends the connecting client to X-Forwarded-For. The rightmost valid
address is the one this process trusts. Addresses further left were supplied
before the request reached Render. The CF-Connecting-IP header is read only
when X-Forwarded-For contains no valid address, so a supplied header cannot
replace the address Render appended.

No country table is bundled. One registry's terms forbid redistributing a
modified copy of its delegation file. See data/geo/README.md. country_code
therefore returns None, and nothing on the network is asked.
"""
import ipaddress
from typing import Optional


def _looks_like_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def _client_ip(forwarded_for: str | None, remote_addr: str | None,
               cf_connecting_ip: str | None = None) -> str | None:
    """The visitor address Render appended, not an earlier supplied hop.

    The rightmost valid X-Forwarded-For entry wins. CF-Connecting-IP is a
    fallback for when that header has no valid address, then the socket peer.
    """
    parts = [part.strip() for part in (forwarded_for or "").split(",") if part.strip()]
    for candidate in reversed(parts):
        if _looks_like_ip(candidate):
            return candidate
    supplied = (cf_connecting_ip or "").strip()
    if _looks_like_ip(supplied):
        return supplied
    if remote_addr and _looks_like_ip(remote_addr):
        return remote_addr
    return None


def _is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if hasattr(addr, "is_global"):
        return bool(addr.is_global)
    return not (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_reserved or addr.is_multicast or addr.is_unspecified
    )


def country_code(ip: str | None) -> str | None:
    """ISO country for a public address, or None.

    Private, loopback, and unroutable addresses are None. A public address is
    also None: no offline table is shipped, and this function does not call
    out to the network.
    """
    if not ip or not _is_public(ip):
        return None
    return None


def lookup(ip: str | None) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (city, region, country). City and region are never derived."""
    return (None, None, country_code(ip))


def extract_and_lookup(
    forwarded_for: str | None, remote_addr: str | None, cf_connecting_ip: str | None = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Return (ip, city, region, country). The address is for the caller to hash, then drop."""
    ip = _client_ip(forwarded_for, remote_addr, cf_connecting_ip)
    _city, _region, country = lookup(ip)
    return (ip, None, None, country)
