# Complete private cube animation implementation plan

> **For agentic workers:** use scoped parallel implementation with non-overlapping files; root independently verifies the integrated result. No production write or deployment until the specific pending approval is received.

**Goal:** Replace the confusing static #28 review with a full 16-step, private animated lesson and clean Kokoro narration.

**Architecture:** Preserve all original files and the saved v1 package. Derive v2 prose by removing Markdown emphasis markers only. A dependency-free, frame-driven Canvas cube player uses reproducible sticker states and move sequences exported from the existing verified simulator. The existing Flask/admin player and saved narration remain in use. Remotion skill timing/layout guidance is applied without adding React, a rendering service or runtime dependencies.

**Tech stack:** existing Python/Pillow/Kokoro/FFmpeg, Flask/Jinja, plain JavaScript Canvas, Node built-in tests. No paid APIs, purchases, credentials or installations.

**Spec / user direction:** complete #28 with animation, not a sample; instructions must be clearer and Kokoro must not read asterisks. All lessons remain unpublished. Existing wording/facts are retained apart from removal of formatting symbols. The old source, assets and MP3s are preserved.

## Global constraints

- Scope only #28; do not approve, activate, delete or modify another lesson.
- Preserve original Markdown and v1 MP3s; create v2 files/job paths.
- Sixteen matching visible/spoken scripts including separate Fun facts, Kokoro af_heart at1.0 with player0.9.
- Fixed camera during face turns; whole-cube regrips explicitly distinguished. One move at a time, highlighted active layer, named direction, pause/replay/manual stepping.
- Reuse the verified integer-coordinate simulator and prove scene endpoints; no invented sticker colors.
- Current bearer is insert-only. A formatting-revision route must not be deployed/used without the user's specific pending approval. Restrict it to inactive/unapproved cube28, exact old fingerprint, identical metadata/art/facts, and formatting-only prose changes. Preserve the exact v1 local package as rollback data. No schema migration or general revision permission.

## Review focus

1. R/D clockwise direction is judged looking at that face, not the camera.
2. Pause, backward seek and rapid slide navigation must not leave intermediate state or advance hidden scenes.
3. All sixteen steps have correct state/data; stage5–16 form one legal worked example, with practice scenes explicitly reset.
4. Wrong source hashes, stale fingerprints, approval/public flags, malformed/other-slug revisions fail closed.
5. Narration contains no emphasis markers; display and MP3 source hashes agree. Browser speech is never a fallback.

## Task1 — authored v2 data and narration preparation

- [ ] Add failing tests for formatting-only prose derivation and continuous scene move endpoints.
- [ ] Export sixteen scene definitions: `initial`24 `{position:[x,y,z],normal:[x,y,z],color}` stickers, `moves` legal notation tokens, bounded captions, focus/setup annotations, and canonical clean script hashes.
- [ ] Create new16-slide manifest/clean locked draft and provenance, retaining original source and all old files.
- [ ] Prove move endpoints against the existing simulator, corner/color invariants, white-layer progression and final solved state.
- [ ] Generate/normalize16 matching local Kokoro clips in a new v2 job; independent file/hash/decode/loudness checks.

## Task2 — private revision storage boundary

- [ ] Test missing/wrong bearer, If-Match absence/stale, wrong slug/count, artwork/fact/metadata changes, active/approved state, rollback on clip failure and identical retry.
- [ ] Implement a single cube-only formatting revision action using existing validators and a locked atomic transaction; no generic overwrite or token setup.
- [ ] Test through the real existing SQLite transaction fixture; prove unrelated row/audio preservation.

## Task3 — full animated player

- [ ] Node failing tests: literal R/U landmarks, inverse/four-turn identities, deterministic mid-frame/endpoints, non-mutating inputs, pause/reset/seek bounds.
- [ ] Implement smooth rigid-layer turns and projection/painter rendering, move label and face-on direction cue, full pause/replay/step controls.
- [ ] Bind private cube data only when all sixteen actual canonical script hashes match v2; leave all other lessons/current v1 unchanged.
- [ ] Integrate voice/demo/slide cancellation without provider calls or premature auto-advance.
- [ ] Verify real desktop/mobile render frames with an existing local browser if available; report limitations honestly.

## Task4 — integration and gated delivery

- [ ] Run Node tests, focused Python regressions and full Python suite, review exact release delta.
- [ ] If specifically approved, push/deploy only the animation/revision change, verify live commit and negative authorization checks.
- [ ] Reconcile stored #28 against the preserved v1 package before one revision POST; confirm new stored fingerprint/16 clips/private flags afterward. Never blindly retry uncertain writes.
- [ ] Return the same private preview link and actual timings. Human listening/physical learner testing still required; no public release.
