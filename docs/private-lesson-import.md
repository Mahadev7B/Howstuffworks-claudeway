# Import one private lesson

Open `/admin/published-lessons` while signed in with the existing admin account.
Under **Import private lesson**, choose the prepared `stage-packet.json` and click
**Import private lesson**. Successful imports redirect to the existing complete
private preview. **Validate without saving** runs the same media/text checks but
does not open a database connection or save a lesson, useful for a safe smoke test.

No provider, image generation, speech generation, new service, or credential is
used. Importing does not approve or activate a lesson. The public homepage and
existing publication/approval controls are unchanged.

## Package contract

The single UTF-8 JSON object is the existing offline packet format:
`entry`, `clips`, `player_rate` (0.9). `entry` contains editorial metadata, a
`lesson` with exactly four illustrated slides, and `cover_data_url`. Each slide
has title, explanation, narration, a separate fun_fact and embedded raster data.
The exact strings and images are preserved; existing derived display-script
fields, if supplied, must match the canonical script.

Each of four ordered clips (indices 0–3) contains its canonical `script`,
`source_text_hash`, base64 `mp3`, `mp3_sha256`, model `Kokoro-82M ONNX v1.0`, voice
`af_heart`, and source `speed` 1.0. Canonical speech is trimmed narration followed
by ` Fun fact: ` and the trimmed fact. Playback is 0.9; do not pre-slow the MP3.
Hashes verify declared source/integrity, not a speech-recognition transcription.
Human listening remains required for review.

Limits: 12 MiB JSON, existing raster checks (2 MiB/image, 16 million pixels,
PNG/JPEG/WebP only), 1 MiB/MP3 and 0.25–90 decoded seconds per clip. Existing
FFmpeg on the server validates bounded MP3 decoding, with a 15-second timeout
per clip. Missing decoder fails closed; no software is installed by this feature.
Multipart requests require a known bounded Content-Length. Filenames are never
used as filesystem paths; imports perform no extraction or URL fetching.

## Safety and retries

The existing admin session, private session-key guard and CSRF token are required.
All validation runs before storage. An insert-only transaction saves the lesson
and all four clips, verifies private/unapproved defaults and exact saved content,
then commits. Transaction-scoped advisory locking serializes importer requests.
An identical still-private/unapproved slug returns the existing preview without
writing. Different content, a reviewed/public existing row, or an occupied sort
order returns 409; no existing lesson is modified. A failed clip insert rolls
back the new lesson and all its clips. If commit acknowledgement is uncertain,
check the lesson list before resubmitting the identical packet.

Local tests use a relational SQLite adapter for storage/rollback contracts; they
do not exercise real PostgreSQL advisory-lock concurrency. Real MP3 decode is
tested when existing FFmpeg is available. No production test lessons are needed.
