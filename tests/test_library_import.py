"""Prepared imports fail closed and do not alter existing publications."""
import base64
import copy
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import io
import json
import sqlite3
from pathlib import Path
import shutil
import subprocess

import pytest
from PIL import Image


@pytest.fixture
def importer():
    assert importlib.util.find_spec("lesson_platform.library_import") is not None, "private package importer missing"
    return importlib.import_module("lesson_platform.library_import")


def packet():
    image = io.BytesIO()
    Image.new("RGB", (640, 400), "#b9d9cb").save(image, "PNG")
    art = "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
    slides, clips = [], []
    for index in range(4):
        slides.append({"title": f"Step {index+1}", "explanation": "Original body.",
                       "narration": "Original body.", "fun_fact": "Separate fact.", "image_data_url": art})
        audio = b"test decoder boundary " + bytes([index])
        script = "Original body. Fun fact: Separate fact."
        clips.append({"slide_index": index, "script": script,
                      "source_text_hash": hashlib.sha256(script.encode()).hexdigest(),
                      "mp3": base64.b64encode(audio).decode(), "mp3_sha256": hashlib.sha256(audio).hexdigest(),
                      "model": "Kokoro-82M ONNX v1.0", "voice": "af_heart", "speed": 1.0})
    return {"entry": {"slug": "fixture-only", "question": "A fixture question?", "title": "Fixture only",
                      "summary": "Local fixture, never sent to production.", "category": "Science", "sort_order": 99,
                      "featured_rank": None, "lesson": {"title": "Fixture only", "slides": slides},
                      "cover_data_url": art}, "clips": clips, "player_rate": 0.9}


def cube_packet():
    value = packet()
    value['entry'].update(slug='solve-a-2x2-rubiks-cube', sort_order=28)
    value['entry']['lesson']['slides'] = [copy.deepcopy(value['entry']['lesson']['slides'][i % 4]) for i in range(16)]
    value['clips'] = [dict(copy.deepcopy(value['clips'][i % 4]), slide_index=i) for i in range(16)]
    return value


def prepare(importer, monkeypatch, value=None):
    # Only the external decoder is replaced; parsing, media hashes and storage stay real.
    monkeypatch.setattr(importer, "_validate_mp3", lambda audio, index: None)
    return importer.prepare_import(json.dumps(value if value is not None else packet()).encode())


def test_prepared_text_art_and_four_audio_hashes_are_preserved(importer, monkeypatch):
    result = prepare(importer, monkeypatch)
    assert result.entry["lesson"]["slides"][0]["narration"] == "Original body."
    assert result.prepared.lesson["slides"][0]["publication_script"] == "Original body. Fun fact: Separate fact."
    assert len(result.clips) == 4 and result.clips[3]["mp3"] == b"test decoder boundary \x03"


def test_cube_sixteen_slides_preserve_canonical_audio_and_atomic_private_idempotence(importer, monkeypatch, storage):
    ready = prepare(importer, monkeypatch, cube_packet())
    assert len(ready.prepared.lesson['slides']) == len(ready.clips) == 16
    assert ready.prepared.lesson['slides'][15]['publication_script'] == 'Original body. Fun fact: Separate fact.'
    assert importer.save_import(ready)['created'] is True
    assert storage.sql.execute('SELECT active,approved_for_publish FROM published_lessons').fetchall() == [(0, 0)]
    assert storage.sql.execute('SELECT slide_index FROM published_lesson_audio ORDER BY slide_index').fetchall() == [(i,) for i in range(16)]
    before = list(storage.sql.iterdump())
    assert importer.save_import(ready)['created'] is False
    assert list(storage.sql.iterdump()) == before


@pytest.mark.parametrize('defect', ['other_slug', 'wrong_sort', 'four_slides', 'missing_slide', 'missing_clip',
                                  'extra_clip', 'last_index', 'last_script', 'active', 'approved'])
def test_sixteen_slide_exception_is_cube_only_and_complete(importer, monkeypatch, storage, defect):
    value = cube_packet()
    if defect == 'other_slug': value['entry']['slug'] = 'arbitrary-sixteen-slides'
    elif defect == 'wrong_sort': value['entry']['sort_order'] = 29
    elif defect == 'four_slides':
        value['entry']['lesson']['slides'] = value['entry']['lesson']['slides'][:4]
        value['clips'] = value['clips'][:4]
    elif defect == 'missing_slide': value['entry']['lesson']['slides'].pop()
    elif defect == 'missing_clip': value['clips'].pop()
    elif defect == 'extra_clip': value['clips'].append(copy.deepcopy(value['clips'][-1]))
    elif defect == 'last_index': value['clips'][-1]['slide_index'] = 14
    elif defect == 'last_script': value['clips'][-1]['script'] = 'Wrong last-slide words'
    else: value['entry'][defect if defect == 'active' else 'approved_for_publish'] = True
    with pytest.raises(ValueError): prepare(importer, monkeypatch, value)
    assert storage.sql.execute('SELECT COUNT(*) FROM published_lessons').fetchone() == (0,)


def test_cube_last_clip_failure_rolls_back_all_sixteen(importer, monkeypatch, storage):
    ready = prepare(importer, monkeypatch, cube_packet())
    storage.fail_audio = True
    storage.fail_audio_index = 15
    with pytest.raises(RuntimeError, match='simulated'): importer.save_import(ready)
    assert storage.sql.execute('SELECT COUNT(*) FROM published_lessons').fetchone() == (0,)
    assert storage.sql.execute('SELECT COUNT(*) FROM published_lesson_audio').fetchone() == (0,)


@pytest.mark.parametrize("defect", ["count", "slides", "index", "source", "audiohash", "voice", "speed", "rate", "flags", "svg", "empty", "sort", "audio_size", "rank"])
def test_invalid_packages_are_rejected_before_storage(importer, monkeypatch, defect):
    value = packet()
    if defect == "count": value["clips"].pop()
    elif defect == "slides": value["entry"]["lesson"]["slides"].pop()
    elif defect == "index": value["clips"][1]["slide_index"] = 0
    elif defect == "source": value["clips"][0]["script"] = "Different words."
    elif defect == "audiohash": value["clips"][0]["mp3_sha256"] = "0"*64
    elif defect == "voice": value["clips"][0]["voice"] = "marin"
    elif defect == "speed": value["clips"][0]["speed"] = 0.9
    elif defect == "rate": value["player_rate"] = 0.81
    elif defect == "flags": value["entry"]["active"] = True
    elif defect == "svg": value["entry"]["lesson"]["slides"][0]["image_data_url"] = "data:image/svg+xml;base64,PHN2Zy8+"
    elif defect == "empty": value["clips"][0]["mp3"] = ""
    elif defect == "sort": value["entry"]["sort_order"] = True
    elif defect == "audio_size": value["clips"][0]["mp3"] = base64.b64encode(b"x"*(1024*1024+1)).decode()
    elif defect == "rank": value["entry"]["featured_rank"] = True
    with pytest.raises(ValueError): prepare(importer, monkeypatch, value)


def test_invalid_json_duplicate_keys_and_size_are_rejected(importer):
    for raw in (b"not json", b'{"entry":{},"entry":{}}', b'{"rate":NaN}', b"x"*(importer.MAX_PACKET_BYTES+1)):
        with pytest.raises(ValueError): importer.prepare_import(raw)


def test_real_mp3_decodes_and_unreadable_audio_is_rejected(importer, monkeypatch):
    executable = shutil.which("ffmpeg")
    bundled = Path(__file__).parents[2] / "japan-short/vendor/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe"
    if not executable and bundled.is_file():
        executable = str(bundled)
    if not executable:
        pytest.skip("FFmpeg is absent on this test host; production requires its existing FFmpeg")
    monkeypatch.setattr(importer.shutil, "which", lambda name: executable)
    fixture = subprocess.run([executable, "-nostdin", "-loglevel", "error", "-f", "lavfi",
                              "-i", "anullsrc=r=24000:cl=mono", "-t", "0.4", "-c:a", "libmp3lame",
                              "-f", "mp3", "pipe:1"], capture_output=True, check=True).stdout
    importer._validate_mp3(fixture, 0)
    with pytest.raises(ValueError): importer._validate_mp3(b"this is not an MP3", 0)


class SQLiteCursor:
    def __init__(self, db): self.db, self.cur = db, db.sql.cursor()
    def __enter__(self): return self
    def __exit__(self, *args): self.cur.close()
    def execute(self, sql, params=()):
        if "pg_advisory_xact_lock" in sql:
            return self.cur.execute("SELECT NULL")
        if self.db.fail_audio and "INSERT INTO published_lesson_audio" in sql and params[1] == getattr(self.db, 'fail_audio_index', 2):
            raise RuntimeError("simulated second-stage storage failure")
        args = [json.dumps(v.obj) if hasattr(v, "obj") else v for v in params]
        self.cur.execute(sql.replace("%s", "?").replace(" FOR UPDATE", ""), args)
    def _row(self, row):
        if row is None: return None
        return tuple(json.loads(v) if self.cur.description[i][0] == "lesson" else
                     bool(v) if self.cur.description[i][0] in ("active", "approved_for_publish") else v
                     for i, v in enumerate(row))
    def fetchone(self): return self._row(self.cur.fetchone())
    def fetchall(self): return [self._row(r) for r in self.cur.fetchall()]


class SQLiteDB:
    def __init__(self):
        self.sql = sqlite3.connect(":memory:")
        self.fail_audio = False
        self.sql.executescript("""CREATE TABLE published_lessons (
            slug TEXT PRIMARY KEY, source_question_hash TEXT, question TEXT, title TEXT, summary TEXT,
            category TEXT, sort_order INTEGER, featured_rank INTEGER, cover_slide_index INTEGER,
            lesson TEXT, thumbnail BLOB, thumbnail_mime TEXT, active INTEGER DEFAULT 0,
            approved_for_publish INTEGER DEFAULT 0);
            CREATE TABLE published_lesson_audio (slug TEXT, slide_index INTEGER, mp3 BLOB,
            model TEXT, voice TEXT, source_text_hash TEXT, PRIMARY KEY(slug,slide_index));""")
    def __enter__(self): return self
    def __exit__(self, kind, value, tb):
        self.sql.rollback() if kind else self.sql.commit()
    @contextmanager
    def transaction(self):
        with self.sql: yield
    def cursor(self): return SQLiteCursor(self)


@pytest.fixture
def storage(importer, monkeypatch):
    db = SQLiteDB()
    monkeypatch.setattr(importer.library, "_connect", lambda: db)
    yield db
    db.sql.close()


def test_atomic_new_import_is_private_and_identical_retry_does_not_duplicate(importer, monkeypatch, storage):
    ready = prepare(importer, monkeypatch)
    assert importer.save_import(ready)["created"] is True
    assert storage.sql.execute("SELECT active,approved_for_publish FROM published_lessons").fetchall() == [(0,0)]
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lesson_audio").fetchone() == (4,)
    before = storage.sql.iterdump()
    before = list(before)
    assert importer.save_import(ready)["created"] is False
    assert list(storage.sql.iterdump()) == before


def test_conflict_and_approved_existing_lesson_are_never_overwritten(importer, monkeypatch, storage):
    ready = prepare(importer, monkeypatch)
    importer.save_import(ready)
    changed = packet(); changed["entry"]["title"] = changed["entry"]["lesson"]["title"] = "Changed"
    other = prepare(importer, monkeypatch, changed)
    with pytest.raises(importer.ImportConflict): importer.save_import(other)
    storage.sql.execute("UPDATE published_lessons SET approved_for_publish=1")
    storage.sql.commit()
    before = list(storage.sql.iterdump())
    with pytest.raises(importer.ImportConflict): importer.save_import(ready)
    assert list(storage.sql.iterdump()) == before


def test_audio_failure_rolls_back_lesson_and_all_clips(importer, monkeypatch, storage):
    ready = prepare(importer, monkeypatch)
    storage.fail_audio = True
    with pytest.raises(RuntimeError, match="simulated"): importer.save_import(ready)
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (0,)
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lesson_audio").fetchone() == (0,)


def test_an_unrelated_existing_lesson_is_unchanged(importer, monkeypatch, storage):
    first = prepare(importer, monkeypatch)
    importer.save_import(first)
    storage.sql.execute("UPDATE published_lessons SET approved_for_publish=1, active=1")
    storage.sql.commit()
    before = storage.sql.execute("SELECT * FROM published_lessons").fetchall()
    old_clips = storage.sql.execute("SELECT * FROM published_lesson_audio").fetchall()
    value = packet(); value["entry"]["slug"] = "another-fixture"; value["entry"]["sort_order"] = 100
    importer.save_import(prepare(importer, monkeypatch, value))
    assert storage.sql.execute("SELECT * FROM published_lessons WHERE slug='fixture-only'").fetchall() == before
    assert storage.sql.execute("SELECT * FROM published_lesson_audio WHERE slug='fixture-only'").fetchall() == old_clips


@pytest.fixture
def site(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    module = importlib.import_module("app")
    module.app.config.update(TESTING=True, SECRET_KEY="test-private-import-secret")
    monkeypatch.setattr(module, "db_enabled", True)
    monkeypatch.setattr(module.library, "list_editorial_publications", lambda: [])
    return module, module.app.test_client()


def login(client):
    with client.session_transaction() as session:
        session["admin"] = True; session["library_csrf"] = "good-token"


def upload(client, raw=b"{}", token="good-token"):
    return client.post("/admin/published-lessons/import", data={"csrf_token": token,
                       "package": (io.BytesIO(raw), "lesson.json")})


def test_upload_route_requires_existing_admin_and_csrf(site):
    _, client = site
    assert upload(client).status_code == 403
    login(client)
    assert upload(client, token="wrong").status_code == 403


def test_import_form_and_malformed_upload_do_not_write(site):
    _, client = site
    login(client)
    response = client.get("/admin/published-lessons")
    assert b"Import private lesson" in response.data and b'multipart/form-data' in response.data
    assert upload(client, b"not json").status_code == 400
    assert client.post("/admin/published-lessons/import", data={"csrf_token": "good-token"}).status_code == 400


def test_upload_redirects_to_existing_private_preview_without_generation(site, importer, monkeypatch, storage):
    module, client = site
    login(client)
    monkeypatch.setattr(importer, "_validate_mp3", lambda *a: None)
    monkeypatch.setattr(module, "generate_lesson", lambda *a: pytest.fail("provider regeneration"))
    response = upload(client, json.dumps(packet()).encode())
    assert response.status_code == 303
    assert response.location.endswith("/admin/published-lessons/fixture-only/preview")
    assert upload(client, json.dumps(packet()).encode()).status_code == 303
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (1,)


def test_upload_size_is_bounded_before_form_parsing(site):
    _, client = site
    login(client)
    response = client.post("/admin/published-lessons/import", data=b"x",
                           environ_overrides={"CONTENT_LENGTH": str(20*1024*1024)})
    assert response.status_code == 413


def test_storage_outcome_error_is_not_reported_as_a_confirmed_rollback(site, importer, monkeypatch):
    _, client = site
    login(client)
    monkeypatch.setattr(importer, "prepare_import", lambda raw: object())
    def unconfirmed(ready): raise RuntimeError("simulated connection failure")
    monkeypatch.setattr(importer, "save_import", unconfirmed)
    response = upload(client)
    assert response.status_code == 503 and b"unconfirmed" in response.data


def test_validation_only_upload_cannot_write_a_lesson(site, importer, monkeypatch, storage):
    _, client = site
    login(client)
    monkeypatch.setattr(importer, "_validate_mp3", lambda *a: None)
    response = client.post("/admin/published-lessons/import", data={"csrf_token": "good-token",
        "validate_only": "1", "package": (io.BytesIO(json.dumps(packet()).encode()), "lesson.json")})
    assert response.status_code == 200 and response.json["saved"] is False
    assert storage.sql.execute("SELECT COUNT(*) FROM published_lessons").fetchone() == (0,)
