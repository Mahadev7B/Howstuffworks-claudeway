"""Scoped bearer imports exercise the real validator and transaction fixture."""
import copy
import importlib
import importlib.util
import json

from flask import Flask
import pytest

from test_library_import import SQLiteDB, packet, cube_packet


TOKEN = "fixture-import-token-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}
URL = "/api/admin/private-lessons"


@pytest.fixture
def api(monkeypatch):
    assert importlib.util.find_spec("lesson_platform.private_api"), "scoped private API is missing"
    module = importlib.import_module("lesson_platform.private_api")
    monkeypatch.setenv("ASKLILOWL_IMPORT_TOKEN", TOKEN)
    monkeypatch.setattr(module.library_import, "_validate_mp3", lambda *a: None)
    storage = SQLiteDB()
    monkeypatch.setattr(module.library, "_connect", lambda: storage)
    app = Flask(__name__)
    app.secret_key = "fixture-session-key"
    app.config["TESTING"] = True
    module.register_private_api(app)
    yield module, app.test_client(), storage
    storage.sql.close()


@pytest.mark.parametrize("configured", [None, "", "short", "x" * 32 + "\n", "é" * 40])
def test_disabled_or_malformed_configuration_fails_closed(api, monkeypatch, configured):
    _, client, storage = api
    if configured is None:
        monkeypatch.delenv("ASKLILOWL_IMPORT_TOKEN")
    else:
        monkeypatch.setenv("ASKLILOWL_IMPORT_TOKEN", configured)
    assert client.post(URL, headers=HEADERS, data=b"{}").status_code == 503
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (0,)


@pytest.mark.parametrize("auth", [None, "Basic " + TOKEN, "Bearer wrong", "Bearer " + TOKEN + " extra"])
def test_bearer_required_even_for_admin_cookie(api, auth):
    _, client, _ = api
    with client.session_transaction() as session:
        session["admin"] = True
    headers = {"Content-Type": "application/json"}
    if auth is not None:
        headers["Authorization"] = auth
    for suffix in ("", "/capabilities", "/fixture-only"):
        response = client.post(URL, headers=headers, data=b"{}") if not suffix else client.get(URL + suffix, headers=headers)
        assert response.status_code == 401


def test_size_and_content_type_rejected_before_any_body_read(api):
    _, client, _ = api
    class Unreadable:
        def read(self, *args): pytest.fail("body read before size/content-type guard")
        def readinto(self, *args): pytest.fail("body read before size/content-type guard")
    response = client.open(URL, method="POST", headers=HEADERS, environ_overrides={
        "CONTENT_LENGTH": str(12 * 1024 * 1024 + 1), "wsgi.input": Unreadable()})
    assert response.status_code == 413
    response = client.open(URL, method="POST", headers={"Authorization": "Bearer " + TOKEN,
        "Content-Type": "text/plain"}, environ_overrides={"CONTENT_LENGTH": "1", "wsgi.input": Unreadable()})
    assert response.status_code == 415
    response = client.open(URL, method="POST", headers=HEADERS,
        environ_overrides={"CONTENT_LENGTH": "", "wsgi.input": Unreadable()})
    assert response.status_code == 411


def test_exact_size_boundary_reaches_validator_and_default_decoder_is_not_bypassed(api, monkeypatch):
    module, client, _ = api
    observed = []
    def validate(raw):
        observed.append(len(raw))
        raise ValueError("fixture invalid package")
    with monkeypatch.context() as patch:
        patch.setattr(module.library_import, "prepare_import", validate)
        response = client.post(URL, headers=HEADERS, data=b" " * (12 * 1024 * 1024))
    assert response.status_code == 400 and observed == [12 * 1024 * 1024]
    decoded = []
    def decoder(audio, index):
        decoded.append(index)
        raise module.library_import.ImportUnavailable("fixture decoder unavailable")
    monkeypatch.setattr(module.library_import, "_validate_mp3", decoder)
    response = client.post(URL, headers=HEADERS, data=json.dumps(packet()))
    assert response.status_code == 503 and decoded == [0]


def test_private_insert_idempotence_receipt_and_no_copy_leak(api):
    module, client, storage = api
    raw = json.dumps(packet()).encode()
    first = client.post(URL, headers=HEADERS, data=raw)
    assert first.status_code == 201
    receipt = first.json
    assert receipt["slug"] == "fixture-only"
    assert receipt["active"] is False and receipt["approved_for_publish"] is False
    assert (receipt["slides"], receipt["audio_clips"]) == (4, 4)
    assert receipt["preview_url"] == "/admin/published-lessons/fixture-only/preview"
    protocol = importlib.import_module("lesson_platform.private_import_protocol")
    assert receipt["fingerprint"] == protocol.fingerprint_prepared(module.library_import.prepare_import(raw))
    before = list(storage.sql.iterdump())
    again = client.post(URL, headers=HEADERS, data=raw)
    assert again.status_code == 200 and again.json["created"] is False
    lookup = client.get(URL + "/fixture-only", headers=HEADERS)
    assert lookup.status_code == 200 and lookup.json["fingerprint"] == receipt["fingerprint"]
    assert b"Original body" not in lookup.data and b"test decoder boundary" not in lookup.data
    assert list(storage.sql.iterdump()) == before
    assert client.get(URL + "/unknown", headers=HEADERS).status_code == 404


def test_cube_private_import_receipt_and_lookups_require_all_sixteen(api):
    module, client, storage = api
    raw = json.dumps(cube_packet())
    assert client.post(URL, headers={'Content-Type': 'application/json'}, data=raw).status_code == 401
    response = client.post(URL, headers=HEADERS, data=raw)
    assert response.status_code == 201
    receipt = response.json
    assert (receipt['slides'], receipt['audio_clips'], receipt['sort_order']) == (16, 16, 28)
    assert receipt['active'] is False and receipt['approved_for_publish'] is False
    before = list(storage.sql.iterdump())
    assert client.post(URL, headers=HEADERS, data=raw).status_code == 200
    lookup = client.get(URL+'/solve-a-2x2-rubiks-cube', headers=HEADERS)
    assert lookup.status_code == 200 and lookup.json['fingerprint'] == receipt['fingerprint']
    assert lookup.json['slides'] == lookup.json['audio_clips'] == 16
    assert list(storage.sql.iterdump()) == before
    protocol = importlib.import_module('lesson_platform.private_import_protocol')
    changed = copy.deepcopy(module.library_import.prepare_import(raw.encode()))
    changed.clips[15]['mp3'] += b'changed final clip'
    assert protocol.fingerprint_prepared(changed) != receipt['fingerprint']
    storage.sql.execute('DELETE FROM published_lesson_audio WHERE slide_index=15')
    storage.sql.commit()
    assert client.get(URL+'/solve-a-2x2-rubiks-cube', headers=HEADERS).status_code == 409


def test_cube_receipt_rejects_changed_stored_position(api):
    _, client, storage = api
    assert client.post(URL, headers=HEADERS, data=json.dumps(cube_packet())).status_code == 201
    storage.sql.execute('UPDATE published_lessons SET sort_order=29')
    storage.sql.commit()
    assert client.get(URL+'/solve-a-2x2-rubiks-cube', headers=HEADERS).status_code == 409


def test_cube_partial_storage_failure_is_never_reported_as_success(api):
    _, client, storage = api
    storage.fail_audio = True
    storage.fail_audio_index = 15
    response = client.post(URL, headers=HEADERS, data=json.dumps(cube_packet()))
    assert response.status_code == 503
    assert storage.sql.execute('SELECT COUNT(*) FROM published_lessons').fetchone() == (0,)
    assert storage.sql.execute('SELECT COUNT(*) FROM published_lesson_audio').fetchone() == (0,)


@pytest.mark.parametrize("change", ["slug", "title", "summary", "category", "question", "sort_order",
    "rank", "thumbnail_mime", "thumbnail", "lesson", "audio", "model", "voice", "source"])
def test_fingerprint_covers_all_stored_content(api, change):
    module, _, _ = api
    protocol = importlib.import_module("lesson_platform.private_import_protocol")
    ready = module.library_import.prepare_import(json.dumps(packet()).encode())
    original = protocol.fingerprint_prepared(ready)
    changed = copy.deepcopy(ready)
    if change in ("slug", "title", "summary", "category", "question"):
        changed.entry[change] += " different"
    elif change == "sort_order": changed.entry["sort_order"] += 1
    elif change == "rank": changed.entry["featured_rank"] = 7
    elif change == "thumbnail_mime": object.__setattr__(changed.prepared, "thumbnail_mime", "image/png")
    elif change == "thumbnail": object.__setattr__(changed.prepared, "thumbnail", b"different")
    elif change == "lesson": changed.prepared.lesson["slides"][0]["narration"] += " different"
    elif change == "audio": changed.clips[0]["mp3"] += b"different"
    else: changed.clips[0][{"source": "source_text_hash"}.get(change, change)] += "different"
    assert protocol.fingerprint_prepared(changed) != original


def test_conflicts_cannot_change_existing_or_approved_rows(api):
    _, client, storage = api
    value = packet()
    assert client.post(URL, headers=HEADERS, data=json.dumps(value)).status_code == 201
    value["entry"]["summary"] = "Changed summary"
    assert client.post(URL, headers=HEADERS, data=json.dumps(value)).status_code == 409
    storage.sql.execute("UPDATE published_lessons SET approved_for_publish=1,active=1")
    storage.sql.commit()
    before = list(storage.sql.iterdump())
    assert client.post(URL, headers=HEADERS, data=json.dumps(packet())).status_code == 409
    assert client.get(URL + "/fixture-only", headers=HEADERS).status_code == 409
    assert list(storage.sql.iterdump()) == before


def test_validation_and_storage_errors_are_sanitized(api, monkeypatch):
    module, client, storage = api
    assert client.post(URL, headers=HEADERS, data=b"{}").status_code == 400
    monkeypatch.setattr(module.library_import, "_validate_mp3", lambda *a: (_ for _ in ()).throw(
        module.library_import.ImportUnavailable("secret provider detail")))
    result = client.post(URL, headers=HEADERS, data=json.dumps(packet()))
    assert result.status_code == 503 and b"secret provider detail" not in result.data
    monkeypatch.setattr(module.library_import, "_validate_mp3", lambda *a: None)
    storage.fail_audio = True
    result = client.post(URL, headers=HEADERS, data=json.dumps(packet()))
    assert result.status_code == 503 and "unconfirmed" in result.json["error"]
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (0,)


def test_capabilities_distinguish_configuration_from_verification(api, monkeypatch, tmp_path):
    _, client, storage = api
    missing = tmp_path / "missing.onnx"
    voices = tmp_path / "voices.bin"
    voices.write_bytes(b"fixture")
    monkeypatch.setenv("KOKORO_MODEL_PATH", str(missing))
    monkeypatch.setenv("KOKORO_VOICES_PATH", str(voices))
    response = client.get(URL + "/capabilities", headers=HEADERS)
    assert response.status_code == 200
    assert response.json["server_tts"]["provider"] == "openai"
    kokoro = response.json["kokoro"]
    assert kokoro["model_configured"] is True and kokoro["model_file_present"] is False
    assert kokoro["voices_file_present"] is True and kokoro["verified"] is False
    assert str(tmp_path).encode() not in response.data
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (0,)
