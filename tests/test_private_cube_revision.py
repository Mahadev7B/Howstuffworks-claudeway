"""The cube-only formatting route updates real storage atomically and fails closed."""
import base64
import hashlib
import io
import json

from flask import Flask
from PIL import Image
import pytest

from lesson_platform import library_import, private_api
from test_library_import import SQLiteCursor, SQLiteDB, cube_packet, packet


TOKEN = "fixture-cube-revision-token-" + "x" * 32
URL = "/api/admin/private-lessons/solve-a-2x2-rubiks-cube/formatting-revision"
SLUG = "solve-a-2x2-rubiks-cube"
HEADERS = {"Authorization": "Bearer " + TOKEN, "Content-Type": "application/json"}


class RevisionCursor(SQLiteCursor):
    def execute(self, sql, params=()):
        self.db.statements.append(sql)
        if self.db.fail_revision and "UPDATE published_lesson_audio" in sql and params[-1] == 15:
            raise RuntimeError("simulated final revision clip failure")
        return super().execute(sql, params)


class RevisionDB(SQLiteDB):
    def __init__(self):
        super().__init__()
        self.fail_revision = False
        self.statements = []

    def cursor(self):
        return RevisionCursor(self)


def reset_scripts(value, *, new_audio=False):
    for index, (slide, clip) in enumerate(zip(value["entry"]["lesson"]["slides"], value["clips"])):
        script = slide["narration"].strip() + " Fun fact: " + slide["fun_fact"].strip()
        clip["script"] = script
        clip["source_text_hash"] = hashlib.sha256(script.encode()).hexdigest()
        if new_audio:
            audio = b"replacement decoder fixture " + bytes([index])
            clip["mp3"] = base64.b64encode(audio).decode()
            clip["mp3_sha256"] = hashlib.sha256(audio).hexdigest()


def old_packet():
    value = cube_packet()
    for slide in value["entry"]["lesson"]["slides"]:
        slide["narration"] = slide["explanation"] = "Turn **the top** once."
    reset_scripts(value)
    return value


def revised_packet():
    value = old_packet()
    for slide in value["entry"]["lesson"]["slides"]:
        slide["narration"] = slide["explanation"] = "Turn the top once."
    reset_scripts(value, new_audio=True)
    return value


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("ASKLILOWL_IMPORT_TOKEN", TOKEN)
    # The external decoder alone is stubbed, as in existing importer integration tests.
    monkeypatch.setattr(library_import, "_validate_mp3", lambda *args: None)
    storage = RevisionDB()
    monkeypatch.setattr(private_api.library, "_connect", lambda: storage)
    app = Flask(__name__)
    app.config.update(TESTING=True, SECRET_KEY="fixture-only")
    private_api.register_private_api(app)
    yield app.test_client(), storage
    storage.sql.close()


def seed(client):
    response = client.post("/api/admin/private-lessons", headers=HEADERS, data=json.dumps(old_packet()))
    assert response.status_code == 201
    return response.json["fingerprint"]


def revise(client, fingerprint, value=None, **kwargs):
    headers = {**HEADERS, "If-Match": fingerprint}
    headers.update(kwargs.pop("headers", {}))
    return client.post(URL, headers=headers, data=json.dumps(value or revised_packet()), **kwargs)


def dump(storage):
    return list(storage.sql.iterdump())


def test_revision_replaces_only_plain_scripts_and_all_sixteen_audio_rows(api):
    client, storage = api
    old = seed(client)
    assert client.post("/api/admin/private-lessons", headers=HEADERS, data=json.dumps(packet())).status_code == 201
    metadata = storage.sql.execute("SELECT slug,source_question_hash,question,title,summary,category,sort_order,featured_rank,cover_slide_index,thumbnail,thumbnail_mime,active,approved_for_publish FROM published_lessons ORDER BY slug").fetchall()
    unrelated = storage.sql.execute("SELECT * FROM published_lessons WHERE slug='fixture-only'").fetchall()
    unrelated_audio = storage.sql.execute("SELECT * FROM published_lesson_audio WHERE slug='fixture-only'").fetchall()
    storage.statements.clear()
    response = revise(client, old)
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json["revised"] is True and response.json["previous_fingerprint"] == old
    assert (response.json["slides"], response.json["audio_clips"], response.json["sort_order"]) == (16, 16, 28)
    assert response.json["active"] is False and response.json["approved_for_publish"] is False
    lesson = json.loads(storage.sql.execute("SELECT lesson FROM published_lessons WHERE slug=?", (SLUG,)).fetchone()[0])
    for slide in lesson["slides"]:
        assert slide["explanation"] == slide["narration"] == slide["publication_narration"] == "Turn the top once."
        assert slide["publication_script"] == "Turn the top once. Fun fact: Separate fact."
        assert slide["fun_fact"] == slide["publication_fun_fact"] == "Separate fact."
    clips = storage.sql.execute("SELECT slide_index,mp3,source_text_hash FROM published_lesson_audio WHERE slug=? ORDER BY slide_index", (SLUG,)).fetchall()
    assert len(clips) == 16
    assert all(row[0] == i and row[1] == b"replacement decoder fixture " + bytes([i]) for i, row in enumerate(clips))
    assert {row[2] for row in clips} == {hashlib.sha256(b"Turn the top once. Fun fact: Separate fact.").hexdigest()}
    assert storage.sql.execute("SELECT slug,source_question_hash,question,title,summary,category,sort_order,featured_rank,cover_slide_index,thumbnail,thumbnail_mime,active,approved_for_publish FROM published_lessons ORDER BY slug").fetchall() == metadata
    assert storage.sql.execute("SELECT * FROM published_lessons WHERE slug='fixture-only'").fetchall() == unrelated
    assert storage.sql.execute("SELECT * FROM published_lesson_audio WHERE slug='fixture-only'").fetchall() == unrelated_audio
    assert not any("DELETE" in sql or "INSERT" in sql or "ALTER" in sql for sql in storage.statements)
    lookup = client.get("/api/admin/private-lessons/" + SLUG, headers=HEADERS)
    assert lookup.json["fingerprint"] == response.json["fingerprint"] != old
    assert b"Turn the top" not in response.data and b"replacement decoder" not in response.data


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Bearer " + "z" * 40])
def test_revision_requires_dedicated_bearer_even_with_admin_cookie(api, authorization):
    client, storage = api
    old = seed(client)
    with client.session_transaction() as session:
        session["admin"] = True
    headers = {"Content-Type": "application/json", "If-Match": old}
    if authorization is not None:
        headers["Authorization"] = authorization
    before = dump(storage)
    response = client.post(URL, headers=headers, data=json.dumps(revised_packet()))
    assert response.status_code == 401 and response.headers["Cache-Control"] == "no-store"
    assert dump(storage) == before


@pytest.mark.parametrize("fingerprint,status", [(None, 428), ("", 428), ("wrong", 400), ('"' + "a" * 64 + '"', 400), ("A" * 64, 400), ("0" * 64, 409)])
def test_if_match_is_required_exact_and_current_before_writes(api, fingerprint, status):
    client, storage = api
    seed(client)
    headers = dict(HEADERS)
    if fingerprint is not None:
        headers["If-Match"] = fingerprint
    before = dump(storage)
    response = client.post(URL, headers=headers, data=json.dumps(revised_packet()))
    assert response.status_code == status and response.headers["Cache-Control"] == "no-store"
    assert dump(storage) == before


def test_only_exact_cube_post_route_is_registered(api):
    client, storage = api
    old = seed(client)
    before = dump(storage)
    assert client.get(URL, headers=HEADERS).status_code == 405
    assert client.post(URL.replace(SLUG, "fixture-only"), headers=HEADERS, data=b"{}").status_code == 404
    assert client.patch(URL, headers=HEADERS, data=b"{}").status_code == 405
    assert dump(storage) == before


@pytest.mark.parametrize("defect", ["active", "approved_for_publish", "missing", "missing_clip", "sort", "cover_index", "stale_source"])
def test_existing_cube_must_be_complete_private_and_unchanged(api, defect):
    client, storage = api
    old = seed(client)
    if defect in ("active", "approved_for_publish"):
        storage.sql.execute("UPDATE published_lessons SET " + defect + "=1")
    elif defect == "missing": storage.sql.execute("DELETE FROM published_lessons")
    elif defect == "missing_clip": storage.sql.execute("DELETE FROM published_lesson_audio WHERE slide_index=15")
    elif defect == "sort": storage.sql.execute("UPDATE published_lessons SET sort_order=29")
    elif defect == "cover_index": storage.sql.execute("UPDATE published_lessons SET cover_slide_index=1")
    else: storage.sql.execute("UPDATE published_lesson_audio SET source_text_hash=? WHERE slide_index=15", ("0" * 64,))
    storage.sql.commit()
    before = dump(storage)
    assert revise(client, old).status_code == 409
    assert dump(storage) == before


@pytest.mark.parametrize("defect", ["words", "explanation", "facts", "image", "cover", "title", "slide_title", "summary", "question", "category", "rank", "subtitle", "old_asterisks"])
def test_revision_rejects_every_change_except_removing_text_asterisks(api, defect):
    client, storage = api
    old = seed(client)
    value = revised_packet()
    entry, slide = value["entry"], value["entry"]["lesson"]["slides"][0]
    if defect == "words": slide["narration"] = "Turn the bottom once."
    elif defect == "explanation": slide["explanation"] = "Turn the bottom once."
    elif defect == "facts": slide["fun_fact"] = "A changed fact."
    elif defect in ("image", "cover"):
        output = io.BytesIO(); Image.new("RGB", (640, 400), "red").save(output, "PNG")
        art = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
        if defect == "image": slide["image_data_url"] = art
        else: entry["cover_data_url"] = art
    elif defect == "title": entry["title"] = entry["lesson"]["title"] = "Other title"
    elif defect == "slide_title": slide["title"] = "Changed slide"
    elif defect == "rank": entry["featured_rank"] = 4
    elif defect == "subtitle": slide["subtitle"] = "New subtitle"
    elif defect == "old_asterisks": value = old_packet()
    else: entry[defect] += " changed"
    reset_scripts(value)
    before = dump(storage)
    assert revise(client, old, value).status_code == 409
    assert dump(storage) == before


@pytest.mark.parametrize("defect", ["count", "slides", "source", "slug", "index", "derived"])
def test_invalid_revision_packages_fail_before_storage(api, defect):
    client, storage = api
    old = seed(client)
    value = revised_packet()
    if defect == "count": value["clips"].pop()
    elif defect == "slides": value["entry"]["lesson"]["slides"].pop()
    elif defect == "source": value["clips"][-1]["script"] = "Wrong script."
    elif defect == "slug": value["entry"]["slug"] = "fixture-only"
    elif defect == "index": value["clips"][-1]["slide_index"] = 14
    else: value["entry"]["lesson"]["slides"][0]["publication_script"] = "Wrong display."
    before = dump(storage)
    assert revise(client, old, value).status_code == 400
    assert dump(storage) == before


def test_identical_retry_is_idempotent_after_success(api):
    client, storage = api
    old = seed(client)
    first = revise(client, old)
    assert first.status_code == 200
    before = dump(storage)
    second = revise(client, old)
    assert second.status_code == 200 and second.json["revised"] is False
    assert second.json["fingerprint"] == first.json["fingerprint"]
    assert dump(storage) == before


def test_already_plain_cube_cannot_use_formatting_route_to_rewrite_audio_again(api):
    client, storage = api
    first = revise(client, seed(client))
    assert first.status_code == 200
    changed = revised_packet()
    audio = b"different audio for already plain narration"
    changed["clips"][0]["mp3"] = base64.b64encode(audio).decode()
    changed["clips"][0]["mp3_sha256"] = hashlib.sha256(audio).hexdigest()
    before = dump(storage)
    response = revise(client, first.json["fingerprint"], changed)
    assert response.status_code == 409
    assert dump(storage) == before


def test_final_audio_update_failure_rolls_back_lesson_and_all_clips(api):
    client, storage = api
    old = seed(client)
    before = dump(storage)
    storage.fail_revision = True
    response = revise(client, old)
    assert response.status_code == 503 and "unconfirmed" in response.json["error"]
    assert "simulated" not in response.json["error"]
    assert dump(storage) == before


def test_revision_body_guards_precede_reads_and_decoder_remains_required(api, monkeypatch):
    client, storage = api
    old = seed(client)
    class Unreadable:
        def read(self, *args): pytest.fail("read before request guard")
        def readinto(self, *args): pytest.fail("read before request guard")
    for overrides, headers, status in [
        ({"CONTENT_LENGTH": str(12 * 1024 * 1024 + 1)}, {}, 413),
        ({"CONTENT_LENGTH": ""}, {}, 411),
        ({"CONTENT_LENGTH": "invalid"}, {}, 400),
        ({"CONTENT_LENGTH": "1"}, {"Content-Type": "text/plain"}, 415),
        ({"CONTENT_LENGTH": "1"}, {"Content-Encoding": "gzip"}, 415),
    ]:
        response = client.open(URL, method="POST", headers={**HEADERS, "If-Match": old, **headers}, environ_overrides={**overrides, "wsgi.input": Unreadable()})
        assert response.status_code == status and response.headers["Cache-Control"] == "no-store"
    before = dump(storage)
    def unavailable(*args):
        raise library_import.ImportUnavailable("private decoder detail")
    monkeypatch.setattr(library_import, "_validate_mp3", unavailable)
    response = revise(client, old)
    assert response.status_code == 503 and b"private decoder detail" not in response.data
    assert dump(storage) == before
