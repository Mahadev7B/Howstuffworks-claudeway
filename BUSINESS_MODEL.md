# Ask Lil Owl — Business Model (Refined)
Last updated: June 2026

---

## What It Is

A kids' educational web app where children ask "How does X work?" and get
4-slide illustrated lessons with voiceover. Target age: 6–12.
Website: asklilowl.com

---

## Tech Stack

| Layer | Service | Cost model |
|---|---|---|
| Web hosting | Render (Flask/gunicorn) | $25/month |
| Database + cache | Neon PostgreSQL (free tier) | $0 |
| Lesson text | Claude Haiku (Anthropic) | ~$0.007/new lesson |
| Image generation | Flux Dev (fal.ai → AWS T4 spot) | per-image → fixed |
| Text-to-speech | OpenAI Nova TTS → Kokoro TTS (GPU) | $0.02/lesson → $0 |
| CDN / tunnel | Cloudflare | $0 |
| Domain | asklilowl.com | ~$2/month |

---

## Pricing

- **Free tier:** 10 lessons/day per user
- **Paid:** $5/month — unlimited lessons
- Stripe payment processing: 2.9% + $0.30 per transaction

---

## Infrastructure Phases

### Phase 1 — Launch (0 to ~50 users)
- Image generation: fal.ai pay-per-image ($0.059/image, 4 images/lesson)
- No fixed GPU cost — pay only when someone asks a new question
- Cache absorbs repeat questions at $0
- Audio: OpenAI Nova TTS

### Phase 2 — Growing (50 to ~500 users)
- Switch image generation to AWS T4 spot instance
  - NVIDIA T4, 16GB VRAM
  - ~$0.16/hr spot price = **$115/month flat**
  - Replaces fal.ai variable cost (~$400/month at this scale)
- Switch TTS to Kokoro TTS running on the same GPU box = **$0**
- OpenAI Nova TTS dependency fully eliminated

### Phase 3 — Profitable (500+ users)
- Consider buying Ryzen AI Max box (one-time ~$1,900)
- Eliminate AWS monthly bill — payback in ~5 months vs T4 24/7
- Add Cloudflare Tunnel for home-to-internet routing
- fal.ai as overflow for parallel request spikes

---

## Business Model at 100 Paying Users

### Revenue
| | Monthly |
|---|---|
| 100 users × $5/month | $500 |
| Stripe fees (2.9% + $0.30/user) | -$44 |
| **Net revenue** | **$456** |

### Costs (Phase 2 stack)
| Service | Monthly |
|---|---|
| AWS T4 spot (Flux Dev + Kokoro) | $115 |
| Claude Haiku (new lessons only ~300/day) | $12 |
| Render web hosting | $25 |
| Neon PostgreSQL | $0 |
| OpenAI Nova TTS | $0 (replaced by Kokoro) |
| Domain / misc | $2 |
| **Total** | **$154** |

### Profit
| | Monthly |
|---|---|
| Net revenue | $456 |
| Total costs | -$154 |
| **Monthly profit** | **$302** |

---

## Cache Hit Rate Sensitivity

Because the GPU is a fixed cost, cache hit rate barely affects profit:

| Cache hit rate | New lessons/day | Haiku cost | Monthly profit |
|---|---|---|---|
| 90% (best) | ~100 | ~$6 | ~$315 |
| 70% (realistic) | ~300 | ~$18 | ~$302 |
| 50% (worst) | ~500 | ~$30 | ~$290 |

**Key insight:** Profitable at every cache rate once on AWS flat billing.

---

## fal.ai vs AWS T4 at 100 Users

| | fal.ai (Phase 1) | AWS T4 spot (Phase 2) |
|---|---|---|
| Image cost at 70% cache | ~$400/month variable | $115/month flat |
| Monthly profit | ~$44 | ~$302 |
| Switching saves | — | **+$258/month** |

---

## Break-Even Analysis

| Users | Monthly revenue (net) | Monthly costs | Profit |
|---|---|---|---|
| 23 | $105 | $154 | -$49 |
| 31 | $141 | $154 | -$13 |
| **33** | **$152** | **$154** | **~$0 (break-even)** |
| 50 | $228 | $154 | +$74 |
| 100 | $456 | $154 | +$302 |
| 500 | $2,285 | $200 | +$2,085 |

---

## Revenue Projections

### 1,000 Users
- Revenue: $4,560/month
- Costs: ~$250/month (Render upgrade + more DB)
- Profit: **~$4,310/month**

### 10,000 Users
- Revenue: $45,600/month
- Costs: ~$2,000/month (multiple GPU boxes / instances)
- Profit: **~$43,600/month**

### 1,000,000 Users
- Revenue: $4,560,000/month
- Costs: ~$50,000/month (fleet, CDN, team)
- Profit: **~$4,510,000/month**

---

## Go-to-Market Strategy

### Near term (Summer 2026)
1. ParentSquare outreach — message teachers directly via school app
2. Pre-seed top 500 questions so cache is warm on launch
3. PWA "Add to Home Screen" prompt with bonus lesson incentive

### B2B (Fall 2026 — school year)
- License to schools: $500–$2,000/school/year
- Target: ClassDojo (11M teachers), Epic! (50M kids), Khan Academy
- Pitch: COPPA-compliant, curriculum-aligned, works on iPads

### Open Source / BYOK
- Release core as open source (MIT or Apache 2.0)
- Schools/developers bring their own API keys
- You maintain the hosted paid version
- Moat: cache of pre-generated lessons, UX, curation

---

## Pre-Launch Blocklist (Must-Do Before Public)

- [ ] COPPA-compliant Privacy Policy
- [ ] Terms of Service
- [ ] Daily lesson cap enforcement (per IP / per user)
- [ ] User auth (sign up / login)
- [ ] Stripe subscription integration
- [ ] Mobile testing (iOS Safari especially)
- [ ] og:image for social sharing

---

## Cost Per Lesson Summary

| Component | Per new lesson | Per cached lesson |
|---|---|---|
| Claude Haiku (text) | $0.007 | $0 |
| Flux Dev images (4x) | $0 (AWS fixed) | $0 |
| Kokoro TTS (audio) | $0 | $0 |
| **Total** | **~$0.007** | **$0** |

Once on AWS flat billing, the marginal cost of a new lesson is under 1 cent.
Cached lessons cost nothing. At 70% cache rate, average cost per lesson = **$0.002**.

---

## AWS GPU Reference

| Instance | GPU | VRAM | Spot price | Monthly (spot) |
|---|---|---|---|---|
| g4dn.xlarge | NVIDIA T4 | 16GB | ~$0.16/hr | ~$115 |
| g5.xlarge | NVIDIA A10G | 24GB | ~$0.35/hr | ~$250 |
| p3.2xlarge | NVIDIA V100 | 16GB | ~$0.90/hr | ~$650 |

**Recommended:** g4dn.xlarge (T4) — handles Flux Dev + Kokoro comfortably,
cheapest entry point, break-even at 33 users.
