"""$4.99 monthly subscription through Stripe Checkout.

Card details never touch this server: parents pay on Stripe's own page, and
Stripe tells us about the subscription through a signed webhook.
Settings come from environment variables set in the Render dashboard:

  STRIPE_SECRET_KEY      sk_test_... while testing, sk_live_... when real
  STRIPE_PRICE_ID        price_... for the $4.99 monthly price
  STRIPE_WEBHOOK_SECRET  whsec_... from the webhook endpoint
  STRIPE_PORTAL_URL      optional customer-portal login link for cancelling
"""
import json
import logging
import os

import stripe

logger = logging.getLogger(__name__)

SITE_URL = os.getenv("SITE_URL", "https://www.asklilowl.com").rstrip("/")
# Stripe subscription states that unlock questions; everything else (canceled, unpaid…) does not.
ACTIVE_STATES = ("active", "trialing")
_SUBSCRIPTION_EVENTS = ("customer.subscription.created", "customer.subscription.updated",
                        "customer.subscription.deleted")


class BillingUnavailable(RuntimeError):
    """Stripe isn't configured yet, or Stripe couldn't be reached."""


def _secret_key() -> str:
    return os.getenv("STRIPE_SECRET_KEY", "").strip()


def enabled() -> bool:
    return bool(_secret_key() and os.getenv("STRIPE_PRICE_ID", "").strip())


def test_mode() -> bool:
    return _secret_key().startswith("sk_test_")


def portal_url() -> str:
    url = os.getenv("STRIPE_PORTAL_URL", "").strip()
    return url if url.startswith("https://") else ""


def create_checkout_url(email: str) -> str:
    """Start a Stripe Checkout page for this parent's email and return its address."""
    if not enabled():
        raise BillingUnavailable("Subscriptions aren't open yet.")
    try:
        session = stripe.checkout.Session.create(
            api_key=_secret_key(),
            mode="subscription",
            line_items=[{"price": os.getenv("STRIPE_PRICE_ID", "").strip(), "quantity": 1}],
            customer_email=email,
            client_reference_id=email[:200],
            metadata={"asker_email": email},
            subscription_data={"metadata": {"asker_email": email}},
            success_url=f"{SITE_URL}/?subscribed=1",
            cancel_url=f"{SITE_URL}/?subscribe=cancelled",
        )
    except stripe.StripeError as exc:
        logger.warning("Stripe checkout could not be created: %s", type(exc).__name__)
        raise BillingUnavailable("Payments are unavailable right now. Please try again.") from exc
    return session["url"]


def subscription_change(payload: bytes, signature: str) -> dict | None:
    """Verify a Stripe webhook and return {email, status, customer_id, subscription_id}.

    Returns None for events we don't act on. Raises ValueError when the
    signature or payload is bad, so forged requests are rejected.
    """
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise BillingUnavailable("Webhook secret is not set.")
    try:
        stripe.Webhook.construct_event(payload, signature, secret)
    except (stripe.SignatureVerificationError, ValueError) as exc:
        raise ValueError("Invalid Stripe webhook") from exc
    event = json.loads(payload)  # Verified above; plain dicts are easier to read safely.
    kind = event.get("type", "")
    obj = (event.get("data") or {}).get("object") or {}
    if kind == "checkout.session.completed":
        if obj.get("mode") != "subscription":
            return None
        email = (obj.get("metadata") or {}).get("asker_email") or (obj.get("customer_details") or {}).get("email")
        status = "active" if obj.get("payment_status") in ("paid", "no_payment_required") else "incomplete"
        return _change(email, status, obj.get("customer"), obj.get("subscription"))
    if kind in _SUBSCRIPTION_EVENTS:
        email = (obj.get("metadata") or {}).get("asker_email")
        status = "canceled" if kind.endswith("deleted") else obj.get("status")
        return _change(email, status, obj.get("customer"), obj.get("id"))
    return None


def _change(email, status, customer_id, subscription_id) -> dict | None:
    email = (email or "").strip().lower()
    if not email or not status:
        logger.warning("Stripe event without an email or status; ignored")
        return None
    return {"email": email, "status": str(status)[:40],
            "customer_id": (customer_id or None), "subscription_id": (subscription_id or None)}
