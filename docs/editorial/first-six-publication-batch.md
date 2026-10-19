# First six Explore lessons — private editorial batch

Status: six locally assembled, validated visual drafts, not staged or published. This file does not change live lessons.

2026-10-19 narrow review corrections: sky slide 3 now shows scattering through open air without a dome; volcano slides 1-2 now use underground schematics with mostly solid rock and no giant molten pool; internet slides 2 and 4 now use abstract data-packet envelopes rather than literal image tiles. Rocket slide 1 now mentions coasting with the engine off, and slide 3 says rockets like these carry fuel and an oxidizer. The original candidates remain preserved under their versioned filenames. `python -m scripts.stage_editorial_batch assets/editorial/first-six-lessons.json` validates all six/24 slides as a dry run, and the full local suite passes 54 tests when pytest uses a writable workspace temp directory.

Later Diwali editorial review expanded slide 1 to explain the festival's light-over-darkness meaning, family joy, and new beginnings, and replaced slide 2's contrasting “some families / others” construction with unified festival-centered wording about families gathering for prayers, diyas, rangoli, sweets, festive meals, and fireworks. This remains collective language rather than a claim that every household performs every activity. The cultural details are consistent with the Smithsonian National Museum of Asian Art's Diwali overview: https://asia.si.edu/whats-on/events/celebrations/diwali/

Voice remains blocked pending approval and access. The final narration batch currently contains 24 clips / 607 words (about 4.05-4.67 minutes at 150-130 wpm). Current OpenAI pricing for `gpt-4o-mini-tts` is $0.60 per million input text tokens and $12 per million output audio tokens; the working estimate is $0.061-$0.070, with a recommended one-time budget cap of $0.10. No key is present in this executor environment and no clip has been generated.

Read-only production check on 2026-10-19: `https://www.asklilowl.com/healthz` returned HTTP 200 with `ok: true`, `tts: true`, and `db: false`. This indicates existing production TTS configuration but no working database connection at that endpoint. Both `/explore` and `/sitemap.xml` returned HTTP 404, so the library is not deployed. Render authentication was unavailable to this executor, so no environment setting, migration, deployment, or activation was attempted.

Clarification: in the current code, `tts: true` proves only that `OPENAI_API_KEY` is non-empty; it is not an authenticated TTS test. `db: false` means `init_db()` did not complete at startup, which can mean either a missing `DATABASE_URL` or a connection/schema-init failure. The health response does not distinguish those cases.

## Exact remaining deployment sequence

1. Obtain explicit commit/push authorization required by workspace policy, then commit this branch and push it to the existing GitHub remote.
2. In the authenticated Render service, verify the deploy source/branch and deploy that exact commit without changing unrelated settings.
3. Inspect startup logs and `/healthz`; require `db: true`. If false, diagnose whether `DATABASE_URL` is absent or connection/schema initialization failed. Do not replace or erase the existing database.
4. Run `python -m scripts.stage_editorial_batch assets/editorial/first-six-lessons.json --write-private-drafts` once in the deployed environment. This inserts inactive snapshots only.
5. With the approved one-time OpenAI budget cap of $0.10, generate narration for all six private drafts through `lesson_platform.library_audio`, pinned to `gpt-4o-mini-tts` / `marin`. Stop before any additional generation if projected or observed batch cost would exceed the cap.
6. Verify 24 saved MP3 rows, HTTP playback for every private clip, durations/decodability, slide navigation, and desktop/mobile layout. Automated audio inspection does not replace the required human listening review for warmth, clarity, pronunciation and pacing.
7. After human listening approval, activate all six snapshots. Verify the six production lesson URLs, covers, all 24 audio URLs, Explore shelves, home-page feature cards, sitemap entries, and that visitor GETs make no generation calls.

The 24-slide batch now lives in `assets/editorial/first-six-lessons.json`, with one distinct illustration per slide and a separate topic-specific cover for each lesson. The local-only validator/importer converts the source PNGs to bounded WebP snapshots without changing the original files or the ordinary lesson cache. `python -m scripts.stage_editorial_batch assets/editorial/first-six-lessons.json` passed for all six on 2026-10-18; it was a dry run and changed no database row. `--write-private-drafts` is an explicit later operation after deployment and review. `python -m scripts.preview_editorial_library assets/editorial/first-six-lessons.json --port 59589` runs a local visual preview; audio controls are disabled there because no saved GPT narration exists yet.

Read-only production audit on 2026-10-18: `cached_lessons` had 174 rows. 69 had an `image_data_url` on every slide, but those are mostly unrelated, one-off questions; the six below are a stronger evergreen launch mix. The existing `image_bytes` field on several rows is an integer size, not stored artwork. Neither `published_lessons` nor `published_lesson_audio` exists on the live database yet. Do not run the staging operation until the library code is deployed with permission and every lesson passes its content/art review.

The two script-revision documents in the main checkout are editorial proposals, not the approved production `lesson` JSON. The new batch reconciles their narration with explanations and fun facts, but is still a draft requiring line-by-line factual, visual, cultural, and listening review. None of these six is approved to publish yet.

| Order | Proposed title / slug | Cache question | Shelf | Saved images | Cover direction |
| --- | --- | --- | --- | --- | --- |
| 1 | Why Is the Sky Blue? / `why-is-the-sky-blue` | `Why is the sky blue?` | Earth & Nature | 0/4 | Wide daytime sky with a child looking upward; reserve dark, quiet space for card text. |
| 2 | How Do Rockets Fly? / `how-do-rockets-fly` | `How do rockets fly?` | Space | 0/4 | Rocket ascending against a clear sky; accurate exhaust direction and no unsafe close-up spectators. |
| 3 | How Does a Volcano Erupt? / `how-does-a-volcano-erupt` | `How does a volcano erupt?` | Earth & Nature | 0/4 | Cutaway volcano showing underground magma and surface lava, dramatic but not frightening. |
| 4 | How Octopuses Change Color / `how-octopuses-change-color` | `How do octopuses change color?` | Animals | 0/4 | Octopus visibly shifting color against a reef; no misleading claim that one color means one emotion. |
| 5 | How the Internet Works / `how-the-internet-works` | `How internet works?` | Technology | 0/4 | Warm, legible visual journey from a child's device to a server, with packets shown as small traveling pieces. |
| 6 | How Diwali Is Celebrated / `how-diwali-is-celebrated` | `How is Diwali celebrated?` | World & Culture | 4/4, unreviewed | A family-safe scene centered on diyas and rangoli; avoid implying every family celebrates in the same way. |

Six private cover candidates exist in `assets/editorial/`, one `<slug>-cover-v1.png` for each row above. They were generated with the built-in image tool, without an API key. The visual preview now crops those separate images into card thumbnails; no live thumbnail has been created. The Diwali candidate still needs a cultural review before use.

Independent main-assistant visual checks found these specific blockers: sky slide 2 exaggerates the atmosphere as a glowing shell; sky slide 3 v1 was rejected as magic-like, and v2 is clearer but still a stylized model; volcano slides 1–2 show an oversimplified giant magma pool; internet slide 2 makes packets look like literal picture tiles, so the draft fun fact explicitly says that is only a visual model; Diwali slide 3 shows one family story but cannot establish the range of community histories by itself. The rocket orbit path is partly off frame. These are private candidates, not accepted final educational art. All per-slide generation logs and SHA-256 values are in each slide-art folder; `sky/generation-log-v1.md` includes the rejected v1 and the current v2.

Factual starting-point checks used primary explanations from [NOAA on blue skies and sunsets](https://www.nesdis.noaa.gov/about/k-12-education/atmosphere/why-the-sky-blue), [NASA on rocket thrust and oxidizer](https://science.nasa.gov/learn/basics-of-space-flight/chapter3-2/), [NASA on orbit as continuous free fall](https://science.nasa.gov/learn/basics-of-space-flight/chapter3-4/), [USGS on magma and lava](https://www.usgs.gov/programs/VHP/about-volcanoes), [USGS on eruption styles](https://volcano.wr.usgs.gov/volcanic_ash/eruption_styles.html), [Smithsonian Ocean on chromatophores](https://ocean.si.edu/ocean-life/invertebrates/how-octopuses-and-squids-change-color), [Cloudflare's packet overview](https://www.cloudflare.com/en-gb/learning/network-layer/what-is-a-packet/), and the [Smithsonian National Museum of Asian Art on Diwali](https://asia.si.edu/whats-on/events/celebrations/diwali/). Source checks do not replace expert/cultural review of the final child-facing page.

The slide count is a description of today's cache, not a limit. Add or split slides wherever comprehension needs it. The Rubik's-cube tutorial is intentionally outside this first batch: its existing four-slide cache does not meet the separately drafted move-by-move teaching requirement.

## Per-slide art direction

Every illustration should make the *specific* slide easier to understand, be high-resolution enough for both the lesson player and a 480×300 card crop, and avoid embedded text (the site supplies accessible captions). Do not reuse one generic cover across four different explanations.

- Sky: sunlight's colors; sunlight entering the atmosphere; blue light scattering across the sky; longer sunset path with warm colors near the Sun.
- Rockets: launch and upward thrust; exhaust directed backward/downward; self-contained propellant and a separated empty stage; sideways orbital motion around Earth, not a rocket simply floating above it.
- Volcano: underground molten rock; magma rising through cracks; contrasting gentle lava flow and explosive ash cloud; scientists monitoring a volcano from a safe location.
- Octopus: octopus in a natural habitat; close-up schematic of chromatophore sacs opening and closing; color/pattern/texture changes; camouflage or communication shown without invented emotional color codes.
- Internet: device request to a server; packetized information; routers directing packets across networks; device reassembling a page. Use a consistent visual path across slides.
- Diwali: varied family celebrations; accurate diyas/rangoli and food; different stories/meanings represented respectfully; child asking a friend about *their* family tradition. First inspect the four already saved images before commissioning replacements.

## Publication gate for each of the six

1. Compare the saved lesson with its editorial script and fact-check all on-screen copy; fix misleading or vague explanations and fun facts. Have the Diwali wording and images reviewed for cultural accuracy.
2. Create or review *all* slide illustrations, then preview every full slide and the actual 480×300 cover crop. Reject blurry, inconsistent, frightening, text-garbled, or scientifically misleading art.
3. Stage a private snapshot only after all slide text and images agree. Generate GPT narration once from the final script, then have a person listen to every saved clip and check pronunciation/pacing.
4. Activate only with explicit publication authorization, after verifying the page, mobile cards, saved-audio playback, sitemap, and absence of generation/API calls on visitor requests.

No API images, paid service, live cache mutation, database migration, upload, deployment, or public activation was performed for this batch. GPT voice clips have not been generated. The production database still has no public-library tables until a separately authorized deployment runs the migration.
