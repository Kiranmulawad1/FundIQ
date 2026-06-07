# Deployment guide

Target architecture: **Vercel (frontend) + Render (backend) + Neon (Postgres+pgvector) + Upstash (Redis)**. All free tier.

This document is a checklist, not a tutorial — it assumes you understand each platform's UI. Estimated time first time through: **90-120 minutes**, mostly account creation + env var copy/paste.

---

## 0. Pre-flight

Before touching any platform, confirm:
- [ ] `git status` is clean and `main` builds + tests pass (`uv run pytest` and `cd frontend && npm run build`)
- [ ] You have a GitHub account and this repo is pushed there
- [ ] You have Cohere + Gemini API keys ready (Cohere from https://dashboard.cohere.com, Gemini from https://aistudio.google.com)
- [ ] Clerk app is in production mode (Dashboard → Configure → Customization → toggle the "Production" instance on)

---

## 1. Provision data: Neon Postgres + pgvector

1. Sign up at https://neon.tech (GitHub auth works)
2. **Create project** → region `Europe (Frankfurt)` to match the backend region
3. Once provisioned, in **Connection Details**:
   - Connection string with **Pooled** enabled → copy this. Looks like `postgresql://user:pass@ep-xxx-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require`
4. **Enable pgvector**: SQL Editor → run `CREATE EXTENSION IF NOT EXISTS vector;` and `CREATE EXTENSION IF NOT EXISTS pg_trgm;`
5. Save two derived URLs (you'll paste both into Render):
   - **`DATABASE_URL`**: prefix the Neon URL with `postgresql+asyncpg://` (replace `postgresql://`)
   - **`ALEMBIC_DATABASE_URL`**: prefix with `postgresql+psycopg://`

---

## 2. Provision Redis: Upstash

1. Sign up at https://upstash.com (GitHub auth works)
2. **Create database** → region `eu-west-1` (or closest to Frankfurt)
3. **TLS/SSL enabled** (default)
4. Copy the **`UPSTASH_REDIS_URL`** under "Connect to your database". This is your **`REDIS_URL`** — looks like `rediss://default:xxx@yyy.upstash.io:6379`
   - Note the `rediss://` (double `s`) for TLS

---

## 3. Deploy backend: Render

1. Sign up / log in at https://render.com (GitHub auth)
2. **New → Blueprint** → select this repo → branch `main`
3. Render reads `render.yaml` automatically. Click **Apply**.
4. Render will ask you to fill in every `sync: false` env var BEFORE the first build. Use these values:

   | Env var | Source |
   | --- | --- |
   | `DATABASE_URL` | Step 1 — Neon, `postgresql+asyncpg://...` |
   | `ALEMBIC_DATABASE_URL` | Step 1 — Neon, `postgresql+psycopg://...` |
   | `REDIS_URL` | Step 2 — Upstash, `rediss://...` |
   | `GEMINI_API_KEY` | https://aistudio.google.com → Get API key |
   | `COHERE_API_KEY` | https://dashboard.cohere.com → API Keys |
   | `CLERK_PUBLISHABLE_KEY` | Clerk Dashboard → API Keys (production instance) |
   | `CLERK_SECRET_KEY` | Same |
   | `CLERK_JWKS_URL` | `https://<your-clerk-domain>.clerk.accounts.dev/.well-known/jwks.json` — find your domain on Clerk Dashboard → Frontend API |
   | `CORS_ORIGINS` | Your Vercel URL once it exists, e.g. `https://fundiq.vercel.app`. For now leave as `https://fundiq.vercel.app` and update after step 4. |
   | `LANGFUSE_PUBLIC_KEY` | *(optional)* From https://cloud.langfuse.com → Settings → API Keys. Enables per-LLM-call tracing + cost dashboards. |
   | `LANGFUSE_SECRET_KEY` | *(optional)* Same page. |
   | `LANGFUSE_HOST` | *(optional)* `https://cloud.langfuse.com` for the EU instance, or `https://us.cloud.langfuse.com` for US. |

5. Click **Create web service**. First build takes ~5-8 minutes (Docker layer download + `uv sync`). Watch the logs.
6. Backend URL: `https://fundiq-backend.onrender.com` (or whatever Render assigned). Hit `/health` to confirm.

### Note on free tier cold starts
After 15 minutes of inactivity Render's free tier spins the container down. The first request after that wakes it up — **expect ~30 seconds for the first response**. Subsequent requests are fast. For an always-on backend upgrade to Render Starter ($7/mo).

---

## 4. Deploy frontend: Vercel

1. Sign up / log in at https://vercel.com (GitHub auth)
2. **Add new → Project** → import this repo → set **Root Directory** to `frontend`
3. **Framework preset**: Next.js (auto-detected from `frontend/package.json`)
4. **Environment variables** — add these before deploy:

   | Env var | Value |
   | --- | --- |
   | `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Same as backend's, **production instance** |
   | `CLERK_SECRET_KEY` | Same as backend's |
   | `BACKEND_INTERNAL_URL` | Render URL from step 3, e.g. `https://fundiq-backend.onrender.com` |
   | `NEXT_PUBLIC_CLERK_SIGN_IN_URL` | `/sign-in` |
   | `NEXT_PUBLIC_CLERK_SIGN_UP_URL` | `/sign-up` |
   | `NEXT_PUBLIC_CLERK_AFTER_SIGN_IN_URL` | `/` |
   | `NEXT_PUBLIC_CLERK_AFTER_SIGN_UP_URL` | `/` |

5. Click **Deploy**. First build ~2-3 minutes.
6. Once deployed, copy the production URL (e.g. `https://fundiq.vercel.app`).
7. **Back to Render** → fundiq-backend → Environment → update `CORS_ORIGINS` to that exact URL. Save → service auto-redeploys.

---

## 5. Seed data: one-time migration + re-embed

Once the backend is healthy and the frontend redirects through it:

1. **Migrate the schema** — the Dockerfile already runs `alembic upgrade head` on every boot, so this is automatic.
2. **Scrape** — sign in to the frontend, then via Render shell or `curl`:
   ```bash
   # Replace TOKEN with a Bearer JWT from your signed-in Clerk session
   # (browser devtools → Application → Cookies → `__session`)
   curl -X POST https://fundiq-backend.onrender.com/admin/scrape/exist \
        -H "Authorization: Bearer TOKEN"
   ```
   Run for each portal: `exist`, `kfw`, `eic`, `horizon`, `bayern`, `nrw`, `bw`.
3. **Re-embed** (only needed if you migrated from a non-Gemini-Embedding version of the corpus):
   ```bash
   curl -X POST https://fundiq-backend.onrender.com/admin/grants/re-embed \
        -H "Authorization: Bearer TOKEN"
   ```
4. **Enrich** (LLM-extracts sector + eligibility + funding_form from each grant):
   ```bash
   curl -X POST https://fundiq-backend.onrender.com/admin/grants/enrich \
        -H "Authorization: Bearer TOKEN"
   ```
   Paced by default at 7s/grant to stay under Gemini's free-tier 10 RPM. Expect ~3 minutes for 26 grants.

---

## 6. Verify

- [ ] https://your-app.vercel.app loads — Sign in/Sign up visible in nav
- [ ] Sign up creates a real user (visible in Clerk Dashboard)
- [ ] `/grants` shows the scraped grants
- [ ] `/recommend` runs through Planner → Retriever → Scorer → Writer → Critic
- [ ] `/saved` lets you bookmark + view saved grants
- [ ] DB query `SELECT owner_user_id FROM agent_sessions ORDER BY created_at DESC LIMIT 5;` shows your real Clerk user id (not `anon-…`)

---

## Cost summary (free tier)

| Service | Free tier limit | Where you'd hit it |
| --- | --- | --- |
| Vercel | 100GB bandwidth/mo | ~50k page views |
| Render | 750 hours/mo single service | One service always-on for the month works |
| Neon | 0.5 GB storage, 100 compute-hours/mo | Light demo traffic fits comfortably |
| Upstash | 10k commands/day | Cache misses dominate; should be fine |
| Clerk | 10k MAU | Effectively unlimited for a portfolio |
| Gemini | 1500 RPD on `gemini-embedding-001`, 10 RPM on `gemini-2.5-flash` | Embedding free; agent calls hit RPM in bursts |
| Cohere | 1000 rerank calls/mo | ~30 user queries/day before throttling |

Total recurring cost: **$0/month**. Render Starter ($7/mo) removes the 15-minute cold-start nap if that's worth it to you.

---

## Troubleshooting

- **"Failed to fetch" in browser** — backend cold start; reload after 30s.
- **Vercel build OOMs on `npm ci`** — set `NODE_OPTIONS=--max-old-space-size=4096` in the Vercel project env. Shouldn't happen for our footprint but possible.
- **CORS errors** — Render `CORS_ORIGINS` must EXACTLY match the Vercel URL incl. `https://` and no trailing slash.
- **Clerk modal stuck on loading** — `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is wrong, or backend `CLERK_JWKS_URL` doesn't match the same Clerk instance.
- **`alembic.ini not found`** — Dockerfile copies `backend/alembic.ini` into `/app/alembic.ini` and runs from `WORKDIR /app`. If you changed file layout, update both ends.
