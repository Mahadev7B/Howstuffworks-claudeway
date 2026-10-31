# Private lesson pipeline — no paid provider APIs

The user forbids paid APIs. New paid image/enrichment adapters were removed; no paid requests were executed. Existing unrelated website generation code is preserved, not used by this pipeline. Nothing here changes a hosting plan, installs a model, creates credentials or publishes a lesson.

## What is automated

`scripts/make_lesson.py --next` selects the next unused catalog entry, locks its original four narration/explanation lines, reads an editorially reviewed local recipe, imports four existing illustrations concurrently with local Kokoro narration, builds/checks one packet and sends one HTTPS upload. The server performs its required untrusted-input validation and atomic private insert. A stored-content receipt confirms metadata, images/thumbnail, all four MP3s and private/unapproved state. The command prints the login-protected preview URL and per-step durations.

There is one durable current review slot plus at most one private lookahead, not a deleted/recreated library. `--next` returns the current ready URL immediately. An exact configured next slug with supplied artwork/recipe can run as `--next --lookahead` (or `--resume --lookahead` for an existing claim) while the current lesson is reviewed. Version-bound local approval promotes the lookahead; duplicate approval never starts another. Production approval/public flags are never changed. The full30-entry catalog starts with16 human-approved lessons; Dosa17 is currently ready and Taekwondo18 is the sole ready lookahead. The original16-step cube stays held, never truncated to4.

## Important image limitation

The Python command has **no supported standalone bridge to the existing ChatGPT built-in image tool**. It does not replace that tool with a paid API, a script-spawned agent, fake artwork or a silent provider fallback. Codex must prepare/review the four illustrations using the same existing built-in tool, then supply their local hashed files. Missing recipes/artwork stop safely. The entire request-to-preview workflow is therefore not fully standalone yet. CLI timing explicitly excludes that off-script image-creation stage; it cannot establish an under3minute total.

Kokoro is verified locally. The repository's server narration path uses OpenAI TTS, and its deployment requirements do not install Kokoro. The pipeline never switches to that paid voice path. The capability endpoint reports package/model configuration separately from actual synthesis verification; no server Kokoro install or compute upgrade was attempted.

## One command after reviewed local artwork is available

Use the existing system app Python, not the audio-only venv:

```powershell
& C:/Python313/python.exe C:/Users/mymal/WorkingFolder/App4kids/Howstuffworks-admin-review-upload/scripts/make_lesson.py --plan
& C:/Python313/python.exe C:/Users/mymal/WorkingFolder/App4kids/Howstuffworks-admin-review-upload/scripts/make_lesson.py --next
```

The example config deliberately has `upload_authorized:false`, no secret, and no new lesson assets. It performs no upload until the separately authorized API is deployed and the user sets a dedicated `ASKLILOWL_IMPORT_TOKEN` through approved environment storage on both the server and intended client process. Do not put it in JSON, git, chat, logs, a URL or command arguments. Computer Use cannot change application security settings; user handoff is required. The token grants only complete private inserts, idempotent identical imports, private receipt checks and capability metadata—not approval, publication, existing-row revision, admin login or general database access. A leak could permit private spam/storage use; revoke/rotate via user-controlled environment settings if exposed. The HTTPS client sends it only to the exact official site origin and never follows redirects.

Config paths resolve relative to the app worktree, independent of caller working directory. `preparation_dir/<slug>/recipe.json` and `artwork.json` are deterministic; no browser/file chooser is used. Override `recipe_file`/`artwork_manifest` only for an explicit local source. Narration uses the already installed Kokoro environment and sets workspace-local TEMP/TMP plus UTF8 in its child process, preserving natural1.0 `af_heart` audio for player0.9.

## Local handoff files (not another provider)

Recipe JSON has `reviewed:true`, `original_lines_sha256` (SHA256 of canonical UTF8 JSON of the exact four lines, sorted keys/compact separators/ensure_ascii false), and `recipe:{slides:[{title,fun_fact,image_prompt} x4],source_issues:[]}`. Editorial review is not public approval. The original lines cannot be rewritten by the recipe.

Artwork JSON has `source_manifest` (local path relative to this file), `source_manifest_sha256`, and four `slides` objects with `path`, exact `sha256`, and `prompt_sha256`. That source manifest is a one-entry local JSON whose four slide `image_prompt` fields match the reviewed recipe. The media helper copies exact raster bytes, never calls an image-generation API. Structural checks detect unreadable/blank/duplicate/near-duplicate/wrong-size/wrong-aspect assets; they cannot guarantee anatomy, semantic relevance or factual accuracy. Human visual and listening approval remains required.

## Resume and lookahead

```powershell
python -m scripts.make_lesson --resume
python -m scripts.make_lesson --approve reviewed-slug --fingerprint EXACT_SAVED_REVIEW_FINGERPRINT
```

Approval updates only the local queue. The latter starts one hidden background child when required configuration is ready; it does not run continuously or mass-generate all drafts. Missing supplied artwork blocks it rather than buying anything. The review URL always shows one lesson; canonical approved records remain in the website database.

An uncertain POST is never automatically retried. The runner first checks for an existing slug before local media work; an existing or uncertain lookup blocks instead of overwriting. After POST the client performs exactly one read-only stored receipt check, including uncertain-POST recovery. Matching fingerprint confirms the saved version; otherwise the job stays uncertain. `--resume` rechecks it; only explicit `--resume --retry-upload` may submit identical unchanged bytes after reconciliation. Owned execution/queue locks prevent overlap and more than one lookahead; stale locks require inspection, not automatic deletion. State/jobs live under workspace `tmp/lesson-pipeline` on persistent local disk; preserve/back up this folder. An I/O failure preserves staging artifacts.

## Standalone local packaging/check command

`scripts/build_private_lesson_package.py` combines the former per-lesson builder, manual canonical/hash checker and local importer validation. Required flags are `--manifest`, `--draft`, `--topic`, `--audio-dir`, `--output-dir`, `--ffmpeg`; optional `--runtime-dir`. It publishes a complete new output directory atomically and reuses an identical one, refusing differing/incomplete existing output. It accepts exactly4 saved normalized clips and validates original words, separate facts, source/audio hashes, model/voice/speed, recorded QA bounds and actual MP3 decoding. Output is `stage-packet.json` plus `packet-receipt.json`; failure is a concise nonzero JSON receipt. Explicit decoder-path support is local-only additive API plumbing; existing server calls retain their old defaults.

## Outside-service inventory and cost truth

| Component | Outside service? | Can it be confirmed universally free? |
|---|---|---|
| Installed Kokoro/ONNX, FFmpeg, Pillow/Python | No; local tools | No per-call service fee; existing hardware/electricity still apply |
| Direct upload to AskLilOwl on existing Render | Existing hosting only | No; existing hosting/usage terms apply. No new service or plan added |
| Existing ChatGPT built-in image tool used by Codex | Yes, outside Python | Cannot certify from available plan/usage information. No separate paid API is used |
| Newly proposed Claude/Haiku/fal/OpenAI API generation | Not used; removed from pipeline | No requests, no charges incurred by this implementation |

Run `make_lesson.py --services` for the same machine-readable inventory. It would be false to promise that every existing outside service is free. No new external service is required or enrolled.

## Deployment boundary

Code/tests are local only until separately approved. Proposed release is limited to the dedicated private API, shared receipt/client support and required validated helpers; no homepage, existing lesson asset, database migration, production approval or public activation. Credential generation/configuration/transmission and a live new-lesson import need their applicable approvals. No lesson17 or production fixture was created while implementing this.

## Implementation evidence (2026-10-24)

- Final integrated suite: `python -m pytest -q -p no:cacheprovider --basetemp tmp/pipeline-tests/review-fixed-full --tb=short` →259 passed,14 preexisting deprecation warnings,65.91seconds. Independent read-only review found two upload-uncertainty/race defects; three reproducing tests failed first, fixes passed all11 orchestration cases and then the full suite.
- An earlier integrated run encountered `test_held_cube_preserves_sixteen_step_publication_gate` PermissionError/WinError5 during atomic state replacement in its temporary test folder. The unchanged test passed on isolated rerun, then the final full suite passed. No permission/security changes or destructive cleanup were used. Earlier missing-parent/defaultTMP fixture setup errors were resolved with a workspace-owned explicit test parent, not a product-code workaround.
- Saved #15 and #16 packages rebuilt byte-identically, without providers or fresh synthesis:3.181318seconds /3.102125seconds for packaging+essential checks (interpreter startup and full artwork generation/upload not included). Their prior SHA256 values still match; original source/media were not modified.
- `--services` returns no paid provider use and explicitly does not certify existing outside services as universally free. `--plan` reports next What Is a Dosa? with no current job or new runtime state. No17 manifest, artwork or narration was created. Image generation remains an assistant-tool handoff, not a script capability.
- No actual HTTP request, key generation/configuration/transmission, production insert, approval, publish, push or deployment occurred in this implementation. Existing16 and all unrelated dirty files remain preserved.
- Review minor retained deliberately: lookahead preflight validates local tools/configuration, not every next recipe/art file. A missing per-lesson handoff can therefore claim the one next job and block it safely. It cannot buy/generate replacements or start another job; documentation describes this limitation rather than pretending lookahead is fully autonomous.

The minimal server release, if approved later, is app.py's two new registration lines plus `lesson_platform/private_api.py` and `private_import_protocol.py`, based on a freshly verified live/remote commit. Queue, local media/CLI and packaging code stay local. Do not include previous worktree changes or the homepage in that release. Dedicated upload-token manual setup is a separate security action; leave authorization disabled until approved and confirmed.
