# Volcano slide art generation log — 2026-10-18

Source: `Howstuffworks-claudeway/docs/editorial/remaining-15-lesson-script-revisions.md`, “How Does a Volcano Erupt?” slides 1–4, plus `docs/editorial/first-six-publication-batch.md` per-slide art direction. Generated with the built-in image generation tool (`transparent_background: false`), one call per illustration. No API key, paid API, or AskLilOwl plugin was used. The previously saved cover candidate was inspected as visual context and left unchanged. Each output was visually inspected, then copied from its generated-image path into this folder. No generated candidate was discarded or regenerated.

All four saved PNGs were checked with `System.Drawing.Image` and are 1672×941 pixels. Images contain no apparent embedded text, logo, border, or watermark. They are candidates for the parent assistant's independent review and lesson-slide preview.

## Slide 1 — `01-underground-magma-v1.png`

Exact prompt:

```text
Use case: scientific-educational. Asset type: wide 16:9 text-free children's Earth-science lesson slide, ages 7+. Primary request: Teach that melted rock underground is magma. Create a clear geologic cutaway of a peaceful cone-shaped volcano and surrounding ground: the lower half reveals an underground pocket of glowing orange molten rock beneath solid layered rock; the surface above is intact and quiet, with no eruption or lava flow. Style: premium painterly science-book illustration, crisp legible forms, warm luminous colors, child-friendly museum quality. Composition: wide cross-section, magma chamber prominent and unmistakably below ground, mountain and sky visible above. Scientific constraints: molten material stays underground in this image. No text, labels, arrows, logos, watermark, border, or inset panels.
```

Source: `C:\Users\mymal\.codex\generated_images\01a15113-4171-76d3-b2c1-b1d6dc8c80c8\exec-85933da6-23be-4665-84e3-6102c8502480.png`
Result: glowing molten rock below intact terrain; no surface lava. The large, open chamber is a teaching simplification and merits editorial science review. SHA-256: `0C4B3D1FBD688E9668D278538D0A73D61DAB892407A3B80C6EAD75616FE7F4CD`.

## Slide 2 — `02-magma-to-surface-lava-v1.png`

Exact prompt:

```text
Use case: scientific-educational. Asset type: wide 16:9 text-free children's Earth-science lesson slide, ages 7+. Primary request: Show magma rising through a crack or central conduit inside a volcano and becoming lava at the surface. Clear side cutaway through one volcano: glowing molten rock visibly connects from a deep underground chamber up a narrow conduit to the summit opening, then a modest orange lava stream flows down the outside slope. The path from below ground to surface must be visually continuous and easy for a child to follow. Style: premium polished painterly science-book illustration, realistic geology simplified for clarity, luminous yet calm, crisp detail. Scientific constraints: rock below the surface and lava above it should have continuous flow; no huge ash explosion. No text, labels, arrows, logos, watermark, border, or extra panels.
```

Source: `C:\Users\mymal\.codex\generated_images\01a15113-4171-76d3-b2c1-b1d6dc8c80c8\exec-e078daf8-1bf2-4437-9a29-108c6ee83b65.png`
Result: continuous molten path from underground reservoir to summit and down the outer slope. SHA-256: `572FCA3C1E32E92E1F4457DE2862E56398CA8964B69EBE0C46C71BAC53718B14`.

## Slide 3 — `03-flow-and-explosive-eruptions-v1.png`

Exact prompt:

```text
Use case: scientific-educational. Asset type: wide 16:9 text-free children's Earth-science lesson slide, ages 7+. Primary request: Compare two kinds of volcanic eruption in one coherent landscape with two separate distant volcanoes: on the left, a quiet low volcano releasing a slow glowing lava flow down its slope; on the right, another volcano sending a tall gray ash plume and small broken-rock fragments upward in an explosive eruption. The contrast should be instantly understandable but not frightening; no people or buildings in danger. Style: premium polished painterly science-book illustration, bright daylight, dramatic yet reassuring colors, crisp distinct silhouettes. Scientific constraints: ash plume is gray rock particles, not black smoke or flame; lava follows gravity downhill. No text, labels, arrows, logos, watermark, border, or inset captions.
```

Source: `C:\Users\mymal\.codex\generated_images\01a15113-4171-76d3-b2c1-b1d6dc8c80c8\exec-a35f9ef9-bf6f-48fe-8cce-2b5af9fa4f1c.png`
Result: two distinct distant eruption styles; left lava flow descends and right ash cloud rises. SHA-256: `8FC8A990538FF14B45C339243DAF79351BE005AA963ECAD873D963EDE4DFF9A8`.

## Slide 4 — `04-scientists-monitoring-v1.png`

Exact prompt:

```text
Use case: scientific-educational. Asset type: wide 16:9 text-free children's Earth-science lesson slide, ages 7+. Primary request: Show scientists monitoring a volcano from a safe distant observatory. Foreground: two field volcanologists beside a small seismic monitoring instrument and a laptop/graph screen, looking across a broad valley toward a quiet volcano; one scientist studies ground measurements and the other observes the mountain. Background: distant volcano showing no active eruption, a gentle swelling slope suggested naturally by the terrain. Emphasize preparation and observation, not danger. Style: polished painterly science-book illustration, realistic equipment simplified for age 7+, warm daylight, vivid clear composition, inclusive people shown at a sensible safe distance. No legible screen text, labels, logos, watermark, border, or frightening eruption.
```

Source: `C:\Users\mymal\.codex\generated_images\01a15113-4171-76d3-b2c1-b1d6dc8c80c8\exec-d2e50c21-bb57-4a31-93b5-e40ff64b43c1.png`
Result: scientists and seismic display in foreground; quiet volcano far across valley. SHA-256: `EC66A6AA621BC7B48C7E3EFEE59040F1C715D13C5ABDEB127EA1363D60BE6317`.

## Corrected underground schematics - 2026-10-19

- `01-underground-magma-v2.png` prompt: "Scientific educational landscape cross-section for children age 7+, polished painterly 3D style. A small dormant volcano at the surface; below it, mostly solid layered rock fills the frame. Deep underground, show a modest irregular zone of partially molten rock embedded tightly within solid rock, with only a few small orange melt pockets between rock crystals. Molten material occupies under 12 percent of the image and does not connect to the surface. No cavern, pool, lake, ocean of lava, vertical lava river, or eruption." SHA-256: `55C8F88298373C6CCB7283C36132D01E31DC33DBEB6A36A2E9E442FFB595A3A4`.
- `02-magma-to-surface-lava-v2.png` prompt: "Scientific educational landscape cross-section for children age 7+, polished painterly 3D style. Mostly solid layered rock beneath a small volcano. A small deep irregular magma pocket feeds a narrow branching crack and thin central conduit upward. Only at the surface vent does a modest red-orange lava flow begin down one slope. The path from underground magma to surface lava is clear but narrow. No cavern, giant pool, open underground lava lake, broad vertical river, or explosive eruption." SHA-256: `360EACCE2C988CD158D3C63A023B5B40A04A1778EF91AB7EDC42589C14A7AFAB`.

Independent visual check: slide 1 depicts partial melt within mostly solid rock; slide 2 depicts a narrow underground path and a modest surface lava flow. Both replaced the giant-pool candidates in the editorial manifest.
