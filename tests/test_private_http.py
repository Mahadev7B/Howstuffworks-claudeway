"""Offline transport tests prevent credentials following redirects or POST retries."""
import importlib
import importlib.util
import json

import pytest


TOKEN = "fixture-import-token-" + "x" * 32
FINGERPRINT = "a" * 64
RECEIPT = {"slug": "fixture-only", "fingerprint": FINGERPRINT,
    "active": False, "approved_for_publish": False, "slides": 4, "audio_clips": 4,
    "preview_url": "/admin/published-lessons/fixture-only/preview"}


@pytest.fixture
def http():
    assert importlib.util.find_spec("lesson_platform.private_http"), "scoped HTTP client is missing"
    return importlib.import_module("lesson_platform.private_http")


class Response:
    def __init__(self, status=200, body=None):
        self.status = status
        self.body = json.dumps(RECEIPT if body is None else body).encode()
    def read(self, size): return self.body[:size]


def transport(monkeypatch, http, outcomes):
    calls = []
    class Connection:
        def __init__(self, host, port, **kwargs):
            assert host in ("asklilowl.com", "www.asklilowl.com") and port == 443
        def request(self, method, path, body=None, headers=None):
            calls.append((method, path, body, headers))
        def getresponse(self):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception): raise outcome
            return outcome
        def close(self): pass
    monkeypatch.setattr(http.http.client, "HTTPSConnection", Connection)
    return calls


@pytest.mark.parametrize("url", ["http://asklilowl.com", "https://evil.test", "https://asklilowl.com.evil.test",
    "https://user:pass@asklilowl.com", "https://asklilowl.com:444", "https://asklilowl.com?key=secret",
    "https://asklilowl.com#fragment", "https://asklilowl.com/path", "https://asklilowl.com\\@evil.test",
    "https://asklilowl.com\n", "https://asklilowl.com?", "https://asklilowl.com#"])
def test_rejects_nonofficial_or_credential_bearing_origins(http, url):
    with pytest.raises(ValueError): http.PrivateImportClient(url, TOKEN)


def test_upload_sends_one_json_request_and_validates_receipt(http, monkeypatch):
    calls = transport(monkeypatch, http, [Response(201), Response()])
    client = http.PrivateImportClient("https://www.asklilowl.com:443/", TOKEN)
    assert client.upload(b"{}", "fixture-only", FINGERPRINT)["fingerprint"] == FINGERPRINT
    assert [c[0] for c in calls] == ['POST', 'GET']
    assert calls[0][:3] == ("POST", "/api/admin/private-lessons", b"{}")
    assert calls[0][3]["Authorization"] == "Bearer " + TOKEN


@pytest.mark.parametrize('stored', [Response(404), Response(200, dict(RECEIPT, fingerprint='b'*64)),
                                  Response(200, dict(RECEIPT, slides=3)),
                                  Response(200, dict(RECEIPT, active=True))])
def test_successful_post_requires_one_matching_stored_receipt(http, monkeypatch, stored):
    calls = transport(monkeypatch, http, [Response(201), stored])
    with pytest.raises(http.UploadUncertain):
        http.PrivateImportClient('https://asklilowl.com', TOKEN).upload(b'{}', 'fixture-only', FINGERPRINT)
    assert [c[0] for c in calls] == ['POST', 'GET']


def test_uncertain_post_is_reconciled_with_read_only_lookup(http, monkeypatch):
    calls = transport(monkeypatch, http, [TimeoutError("secret transport detail"), Response()])
    result = http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", "fixture-only", FINGERPRINT)
    assert result["fingerprint"] == FINGERPRINT
    assert [c[0] for c in calls] == ["POST", "GET"]


def test_failed_reconciliation_reports_safe_uncertainty_without_retry(http, monkeypatch):
    calls = transport(monkeypatch, http, [OSError("private secret"), TimeoutError("private secret")])
    with pytest.raises(http.UploadUncertain) as caught:
        http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", "fixture-only", FINGERPRINT)
    assert [c[0] for c in calls] == ["POST", "GET"]
    assert "private secret" not in str(caught.value)


@pytest.mark.parametrize("token", ["", "short", "é" * 40, "x" * 40 + "\r\nInjected: header"])
def test_invalid_token_rejected_before_network(http, monkeypatch, token):
    calls = transport(monkeypatch, http, [])
    with pytest.raises(ValueError): http.PrivateImportClient("https://asklilowl.com", token)
    assert calls == []


@pytest.mark.parametrize("post", [Response(302), Response(503), Response(200, {"unexpected": "secret"})])
def test_redirect_and_bad_response_never_repost_or_exfiltrate(http, monkeypatch, post):
    calls = transport(monkeypatch, http, [post, Response(404)])
    with pytest.raises(http.UploadUncertain) as caught:
        http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", "fixture-only", FINGERPRINT)
    assert [c[0] for c in calls] == ["POST", "GET"]
    assert "secret" not in str(caught.value) and TOKEN not in str(caught.value)


@pytest.mark.parametrize("change", [{"fingerprint": "b" * 64}, {"active": True}, {"approved_for_publish": True},
    {"audio_clips": 3}, {"preview_url": "https://evil.test"}, {"slug": "different"}])
def test_mismatched_reconciliation_never_claims_success(http, monkeypatch, change):
    receipt = dict(RECEIPT, **change)
    calls = transport(monkeypatch, http, [TimeoutError(), Response(200, receipt)])
    with pytest.raises(http.UploadUncertain):
        http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", "fixture-only", FINGERPRINT)
    assert [c[0] for c in calls].count("POST") == 1


def test_confirmed_rejection_does_not_retry(http, monkeypatch):
    calls = transport(monkeypatch, http, [Response(409, {"error": "provider secret"})])
    with pytest.raises(http.PrivateImportError) as caught:
        http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", "fixture-only", FINGERPRINT)
    assert len(calls) == 1 and "provider secret" not in str(caught.value)


def test_lookup_missing_and_redirect_are_bounded(http, monkeypatch):
    calls = transport(monkeypatch, http, [Response(404), Response(302)])
    client = http.PrivateImportClient("https://asklilowl.com", TOKEN)
    assert client.lookup("fixture-only") is None
    with pytest.raises(http.PrivateImportError): client.lookup("fixture-only")
    assert [c[0] for c in calls] == ["GET", "GET"]


@pytest.mark.parametrize("slug,fingerprint", [("../admin", FINGERPRINT), ("fixture-only", "wrong")])
def test_upload_arguments_rejected_before_network(http, monkeypatch, slug, fingerprint):
    calls = transport(monkeypatch, http, [])
    with pytest.raises(ValueError):
        http.PrivateImportClient("https://asklilowl.com", TOKEN).upload(b"{}", slug, fingerprint)
    assert calls == []


def test_cube_transport_confirms_sixteen_private_clips_with_one_stored_get(http, monkeypatch):
    receipt = dict(RECEIPT, slug='solve-a-2x2-rubiks-cube', slides=16, audio_clips=16, sort_order=28,
                   preview_url='/admin/published-lessons/solve-a-2x2-rubiks-cube/preview')
    calls = transport(monkeypatch, http, [Response(201, receipt), Response(200, receipt)])
    assert http.PrivateImportClient('https://asklilowl.com', TOKEN).upload(
        b'{}', receipt['slug'], FINGERPRINT)['slides'] == 16
    assert [call[0] for call in calls] == ['POST', 'GET']


@pytest.mark.parametrize('change', [dict(slides=4), dict(audio_clips=15), dict(sort_order=29),
                                  dict(sort_order=None), dict(active=True), dict(approved_for_publish=True)])
def test_cube_receipt_cannot_accept_wrong_count_position_or_private_state(http, monkeypatch, change):
    receipt = dict(RECEIPT, slug='solve-a-2x2-rubiks-cube', slides=16, audio_clips=16, sort_order=28,
                   preview_url='/admin/published-lessons/solve-a-2x2-rubiks-cube/preview')
    receipt.update(change)
    calls = transport(monkeypatch, http, [Response(201), Response(200, receipt)])
    with pytest.raises(http.UploadUncertain):
        http.PrivateImportClient('https://asklilowl.com', TOKEN).upload(b'{}', receipt['slug'], FINGERPRINT)
    assert [call[0] for call in calls] == ['POST', 'GET']
