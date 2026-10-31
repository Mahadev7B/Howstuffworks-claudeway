# Batch editorial review and bounded lookahead implementation plan

> For agentic workers: use the authorized parallel worker workflow; TDD for runtime changes.

Goal: cache the factual/editorial review for the remaining13 drafts, avoid repeated manual checks, and keep one next private lesson ready while the current one is reviewed.

Architecture: retain existing local Kokoro and concurrent built-in image calls; reuse exact-text prepared recipes and atomic direct-upload importer. Add a local review manifest bound to exact source lines and recipe bytes. Keep current review and at most one lookahead job in the same durable queue, with safe promotion on explicit version-bound human approval. No production code deployment or credential changes.

Spec: user's two speed fixes and one-private-lesson-ahead request in this conversation, October24 Indianapolis / October25UTC. Original wording stays exact, no paid APIs, everything unpublished. No batch media generation; only18 as lookahead after existing17. The cube stays held for16-step diagrams and physical solves.

## Tasks

- [ ] Parallel factual/editorial source review for18–30 using authoritative sources; save exact original lines, individual claims, separate proposed facts, citations, source hashes and per-topic recipe. Do not silently fix originals. Save required corrections as gated proposals. No image/audio generation by reviewers.
- [ ] Red/green tests for automatic pre-generation duplicate lookup (stop without generation on existing/conflicting/unknown outcome), exactly one post-upload stored fingerprint check, safe uncertainty with no POST retry, and unchanged ready URL fast path.
- [ ] Red/green tests for one lookahead slot, concurrency/race and restart, current review unchanged, approval promotion without DB approval, no third active job, preserved held cube.
- [ ] Validate cached review against exact source and recipe bytes; changed text/fact/prompt/source issue invalidates it. Preserve existing17 and approved16.
- [ ] Run focused tests and full suite, inspect scoped diff and actual cache binding/results. No commit/push/deploy.
- [ ] Once18 facts/recipe pass, create its4pictures concurrently with local narration, use one command for package/upload, one post-upload receipt. Keep17 current and18 private lookahead. Stop at18, report source corrections/holds, exact URL and measured timing.

Review focus: missing/stale review files fail closed; time-sensitive claims remain flagged for publication-time revalidation; lost POST acknowledgement never reposts blindly; process and queue locks prevent multiple lookaheads; cached factual review does not claim image correctness or human listening.
