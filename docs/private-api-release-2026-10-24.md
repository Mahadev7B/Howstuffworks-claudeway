# Private upload API release receipt

Release approved only for private insert/receipt API; no token setup or lesson creation.

- Remote main: `c220d8f4acd94e468677586ff13471bb93cec357`, fast-forward from `987e19417487bdf97485a0d1e8b3bf3a660a68c2` with expected-SHA non-force update.
- Exact release: `app.py` two registration lines; new `lesson_platform/private_api.py` (146 lines); new `lesson_platform/private_import_protocol.py` (63 lines). No other local work included.
- Focused API/client/importer suite: 88 passed, 14 existing deprecation warnings, 3.63 seconds.
- Additional API/client run using exact previously deployed importer source: 61 passed, one pytest plugin warning, 1.31 seconds.
- GitHub main reports no required status checks; no check runs or workflow runs reported for this commit. No CI success inferred from empty results.
- Before push, public `/debug/status` reported `987e1941`, healthy DB. Network sandbox DNS failed; a reviewed outside-sandbox public curl read succeeded. No browser guard was bypassed; browser interaction remained stopped.
- At 2026-10-24 22:21:47 UTC (18:21:47 Indianapolis EDT), new capabilities GET returned 503 JSON `Private import is not configured`, with `Cache-Control: no-store` and Render/Gunicorn response headers.
- At 22:21:59 UTC approximately, public `/debug/status` reported `git_commit: c220d8f4`, `ok: true`, DB enabled, uptime35 seconds. Remote full SHA verified above; runtime exposes only eight characters. Render dashboard deployment ID/full-SHA verification was unavailable because native Computer Use URL safety guard stopped that interface.
- At 22:22:01 UTC, unauthenticated POST of empty JSON returned the same 503/no-store before storage. At 22:22:04 UTC, private receipt GET also returned the same 503/no-store. Public `/healthz` remained healthy, DB true, daily spend0.
- No real token generated, configured, retrieved or transmitted. No authenticated production save tested. No lesson insert/revision, approval, activation, migration, new media generation or paid API request performed.

## Manual setup still required

The user must generate and retain a cryptographically random private value (recommended 32 random bytes encoded as 64 hexadecimal characters) using their trusted local secret-generation mechanism. Do not send it in chat. Set the exact same value as `ASKLILOWL_IMPORT_TOKEN` in the existing Render service Environment and in the intended local uploader process's protected environment. Render Save/redeploy is required to apply its environment change. Do not reuse FLASK_SECRET_KEY, ADMIN_PASSWORD or an API-provider key. Keep this value out of repository files, JSON config, command arguments, URLs, screenshots and logs. Agent has not performed this security setup and must not change security settings through Computer Use.

Scope: complete inactive/unapproved inserts, exact idempotent repeats and private receipt/capability checks. No approval, publication, existing-row revision, admin session or general DB access. A leak could permit private spam/storage consumption. Token-authenticated behavior still needs an authorized real run after manual setup; no production fixture was created to test it.

## Outside services and costs

- Direct pipeline upload: existing AskLilOwl website, Render web hosting and its existing PostgreSQL storage. Not certified free; existing hosting/storage/usage terms apply, no new plan/service added.
- Pictures: existing ChatGPT built-in image tool, assistant-mediated and outside standalone Python. Not certified free under the user's current subscription/usage terms; no separate image API added or invoked during release.
- Release tooling: existing GitHub repository/connector. No new subscription or paid workflow started; account-wide billing was not inspected or certified free.
- Narration/validation: installed local Kokoro/ONNX, FFmpeg, Pillow/Python. No outside per-call service or API fee; local hardware/electricity still apply.
- No new Claude/Haiku, fal or OpenAI generation API is used by this pipeline/API. Existing unrelated dynamic website generation still reports its preexisting Claude/Flux/OpenAI configuration; this narrow release did not remove or invoke that unrelated functionality.

It would be inaccurate to confirm every outside service is universally free. No paid provider request or new paid service was introduced.
