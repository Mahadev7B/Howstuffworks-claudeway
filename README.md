# Ask Lil Owl

Illustrated, narrated lessons for curious kids, used with a parent nearby.

Live site: [https://www.asklilowl.com](https://www.asklilowl.com)

Published lessons are free to watch. Asking a new question needs a parent's email. The first 3 different questions on that email are free. A $4.99 monthly Stripe subscription, when the question box offers it, lets that email keep asking.

## Stack

| Layer | What it is |
|---|---|
| App | Flask, served by gunicorn |
| Hosting | Render, behind Cloudflare |
| Database | PostgreSQL (optional on a laptop; needed for saved lessons, emails, and the library) |
| Lesson text | Anthropic Claude |
| Pictures | fal.ai Flux, or Matplotlib when Flux is off |
| Voice | OpenAI text-to-speech, or the browser's speech synthesis |
| Payments | Stripe Checkout |
| Pages | Jinja templates, CSS, and plain JavaScript |

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put ANTHROPIC_API_KEY in .env. The other keys can stay empty.
python app.py
```

Open http://localhost:5000

Without `DATABASE_URL` the app still starts. New questions are kept in memory for that process. The library, saved lessons, parent emails, and the admin stats need Postgres.

Useful environment variables, all optional except the Anthropic key when you want a new lesson:

- `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- `FLASK_SECRET_KEY`
- `DATABASE_URL`
- `OPENAI_API_KEY` for server voice
- `IMAGE_PROVIDER` (`matplotlib` or `flux`) and `FAL_API_KEY`
- `ADMIN_PASSWORD` for `/admin`
- `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PORTAL_URL`

## Tests

```bash
pytest
```

## Deploying

`render.yaml` is the Render service. Its health check is `GET /healthz`, which returns `{"ok": true}` and nothing else. `GET /debug/status` is the detailed snapshot and requires an admin login.

## Project layout

```
app.py                  Flask routes
lesson_platform/        lesson generation, billing, library, database
templates/              pages
static/                 CSS, JavaScript, brand images
tests/                  pytest
```
