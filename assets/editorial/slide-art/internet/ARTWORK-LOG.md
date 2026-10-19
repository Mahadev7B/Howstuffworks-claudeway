# How the Internet Works slide art

Source: `Howstuffworks-claudeway/docs/editorial/first-15-lesson-script-revisions.md`, four narration beats under “How the Internet Works.” Generated 2026-10-18 with the built-in `image_gen` tool. Each call used `transparent_background: false`; no API, paid service, or plugin was used. These are slide illustrations, separate from the existing cover candidate.

| Slide | Selected file | Built-in output | Visual review |
| --- | --- | --- | --- |
| 1 | `01-device-request-and-server-v1.png` | `exec-c32a29d3-f4bc-4269-8680-c36bc4cc2ea1.png` | Child/tablet and remote server are distinct; glowing exchange is a metaphor, no readable text. |
| 2 | `02-picture-in-packets-v1.png` | `exec-0e4240a8-d73d-4481-bf82-0c8bd154d165.png` | Fox picture visibly divided among separate tiles in cable. |
| 3 | `03-cables-radio-and-routers-v1.png` | `exec-dd6f2c12-e3a8-4400-a4fa-97a656c5059c.png` | Cables, router boxes, and radio link all visible; small packets glow along paths. |
| 4 | `04-packets-reassembled-v1.png` | `exec-e2a0d07b-f2e2-405d-b172-c608805834c9.png` | Tablet visibly assembling the same fox picture from tiles. |

All selected files are PNG, 1586 × 992 px. Visual QA was manual inspection of the displayed images. The glowing paths/tiles are educational metaphors, not literal depictions of networking equipment behavior. Generated outputs remain under `C:\Users\mymal\.codex\generated_images\01a15113-f9ee-7631-9c22-66ef1d30b914`; selected files were copied into this folder. No existing assets were replaced.

One additional candidate was generated for slide 2: `exec-fe0e7ed4-dedc-4b59-996f-6daed03d9ec6.png`. It was rejected because its undersea setting could confuse the internet lesson. It was not copied into the project.

## Exact prompts

1. `Use case: scientific-educational. Asset type: landscape illustration for one children's narrated lesson slide, no slide text. Scene: a child seen from behind using a tablet at a cozy desk, with a subtle luminous path from the tablet across the room to a clearly separate server computer in a small server room vignette; the server is sending a finished web page image back toward the tablet as a gentle visual metaphor. Teach beat 1: device requests a page from a server, server responds. Rich cinematic 3D storybook illustration, warm amber home light contrasted with clean blue network light, plausible hardware, clear focal points, simple composition for ages 6-10. No words, labels, letters, numbers, arrows, logos, watermark, or impossible screen text. Wide 16:10 composition.`
2. Initial rejected candidate: `Use case: scientific-educational. Asset type: landscape illustration for one children's narrated lesson slide, no slide text. Scene: a single large picture of a friendly sea turtle represented as a neat grid of separate glowing square tiles traveling along a luminous cable, each tile showing a different fragment of that same picture, with tiny abstract destination marks but no readable text. The visual must clearly show one image divided into many small packets. Rich cinematic 3D storybook look, deep blue and turquoise setting, warm golden accents, crisp tiles, legible at a glance for ages 6-10. No words, labels, letters, numbers, arrows, logos, border, watermark, or extra objects. Wide 16:10 composition.`
2. Selected replacement: `Use case: scientific-educational. Asset type: landscape illustration for one children's narrated lesson slide, no slide text. Scene: in a clearly digital network setting, a simple picture of a red fox on a tablet screen is being represented by many separate small glowing square packets moving through a luminous blue cable. Each square carries a visibly different fragment of that same fox picture. Place the intact fox picture at one end and the separated tiles in transit across the scene, no duplicate intact foxes. Keep the lesson concept unmistakable: information often travels in small pieces. Rich cinematic 3D storybook illustration matching a cozy technology lesson, deep navy background, blue cable and golden packet rims, simple graphic clarity for ages 6-10. No ocean, water, text, labels, letters, numbers, arrows, logos, border or watermark. Wide 16:10 composition.`
3. `Use case: scientific-educational. Asset type: landscape illustration for one children's narrated lesson slide, no slide text. Scene: a clearly grounded network map in a miniature city at dusk: glowing square information packets travel along physical buried fiber cables between three small router devices and over a short radio link between antennas, heading toward a lit home. Routers look like real small electronic network boxes, paths visible as a gentle visual metaphor. Teach beat 3: packets cross cables, radio links, and networks; routers direct them. Rich cinematic 3D storybook realism, blue and amber glow, clear route and uncluttered composition for ages 6-10. No words, labels, letters, numbers, arrows, logos, watermark, ocean or fantasy portals. Wide 16:10 composition.`
4. `Use case: scientific-educational. Asset type: landscape illustration for one children's narrated lesson slide, no slide text. Scene: close view of a tablet on a cozy desk assembling incoming glowing square image fragments into one complete colorful picture of a red fox on its screen. A few separate tiles approach the screen while most are fitted together like a seamless digital image; nearby child hands rest on the desk without touching the tiles. Teach beat 4: a device puts received pieces together to display a page or picture. Rich cinematic 3D storybook realism, warm amber desk lighting and clean blue digital glow, approachable for ages 6-10, composition legible at a glance. No words, letters, numbers, labels, arrows, logos, border, watermark, or impossible extra fingers. Wide 16:10 composition.`

## SHA-256 of selected files

| File | SHA-256 |
| --- | --- |
| `01-device-request-and-server-v1.png` | `c0714f81084cdcbadcf592173a4e282624015dd6bc101390a47fbd7f6b183446` |
| `02-picture-in-packets-v1.png` | `69ab079f489b65dedb5c50725f8306bcba6aa4587c5b6787f70869085bfa9719` |
| `03-cables-radio-and-routers-v1.png` | `7b65323327de51497d84a19d91e22635dd540ab828fea736614e9cbf0710406b` |
| `04-packets-reassembled-v1.png` | `3a791d9c88c34fd186ed77ca9d5fb67444454209fa9af89c36bacb52f1acdc4f` |

## Corrected packet analogy - 2026-10-19

The literal picture-tile candidates above remain preserved but are no longer referenced by the editorial manifest.

- `02-data-packets-v2.png`: abstract sealed packets carry dots and bars from server to device; the fox image remains intact only on the destination screen. SHA-256: `3F738C03A0126EB7B2E42ECBD7282C73C7A218E0B80877C7C1295E9F40FD3A45`.
- `04-page-built-from-data-v2.png`: abstract packets arrive beside a device displaying one intact, completed web page and fox image. SHA-256: `FDD851EF90E2D05EF271F2AF8CF6E6D86F17A1AFCE753CADC15F00E30B887013`.

Both were generated with the built-in image tool using `transparent_background: false`. Prompts explicitly prohibited photo fragments, image tiles, mosaics and jigsaws. Independent visual review confirmed that packets contain abstract data marks rather than pieces of the final image.
