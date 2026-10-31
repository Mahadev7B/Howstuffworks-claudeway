# Private 16-slide cube support

The insert-only private importer requires 16 slides and 16 canonical saved clips only for `solve-a-2x2-rubiks-cube` at integer `sort_order: 28`. All other topics still require four. A truncated four-slide cube is rejected. The existing 12 MiB packet limit, bearer scope, metadata/media validation, natural Kokoro Heart speed, 0.9 player rate, fingerprint checks, private flags, atomic transaction and idempotence remain in force.

Cube POST/GET receipts report `slides: 16`, `audio_clips: 16`, and `sort_order: 28`. Normal four-slide receipts retain their existing shape. The shared client validates either allowed shape and still performs one stored GET after POST.

Minimal server release files are `lesson_platform/private_import_protocol.py`, `lesson_platform/library_import.py`, and `lesson_platform/private_api.py`. Local packaging/audio additionally use `lesson_platform/private_package.py` and `lesson_platform/pipeline_media.py`. No schema migration or player changes are required. The existing local Kokoro generator already accepts slide numbers up to the supplied manifest length.

Use the manual manifest/audio/package path for this held lesson. The automatic four-slide recipe/image pipeline and queue/catalog hold are unchanged. These runtime checks do not verify cube algorithms, diagrams, first-time learner success, human listening, or physical solves, and do not grant editorial or publication approval.
