# Bounded loudness normalization fallback

`MediaService.make_audio` retains the existing loudness targets (`I=-16`, `TP=-1.5`, `LRA=11`) and all audio acceptance bounds. If an unreceipted normalized clip fails only integrated LUFS, it makes one measured two-pass FFmpeg normalization attempt from the same raw Kokoro MP3. It does not regenerate narration, change text, apply tempo filters, or alter completed matching receipts.

Before retrying, it exclusively saves sibling `.one-pass-failed.mp3` and `.one-pass-failed.normalization-receipt.json` files. Their hashes, failed QA, raw/source hashes, raw loudnorm measurements, and exact applied filter are saved in `.two-pass-attempt.json`. A successful clip also records this provenance and repaired QA in its normalization binding and lesson receipt. The established normalized filename is updated only after the candidate passes every existing gate.

Any other audio defect fails closed. A failed or interrupted fallback preserves its outputs and blocks another attempt on resume; inspect the retained artifacts. A completed repaired clip can resume without another encode. Existing completed lesson receipts retain their original cache behavior. Automated measurements do not replace human listening.
