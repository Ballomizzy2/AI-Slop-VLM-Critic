# Deploy Critic Pipeline to the Web

Deploy so you can open a link and run everything from the server. The app uses **Docker** and works on Railway, Render, Fly.io, or any container host.

---

## Prerequisites

- **ANTHROPIC_API_KEY** — Get one at [console.anthropic.com](https://console.anthropic.com/)
- **GitHub** — Push your code to a repo for one-click deploys

---

## Option 1: Railway (recommended)

1. Go to [railway.app](https://railway.app) and sign in with GitHub.
2. **New Project** → **Deploy from GitHub repo** → select `AI-Slop-VLM-Critic`.
3. Railway will detect the `Dockerfile` and build automatically.
4. **Variables** → Add:
   - `ANTHROPIC_API_KEY` = your API key
   - Optionally: `PORT` (Railway sets this automatically), `MAX_FRAMES`, `WHISPER_MODEL`, etc.
5. **Settings** → **Networking** → **Generate Domain** to get a public URL like `https://your-app.up.railway.app`.
6. Open the URL, upload a video, and run the pipeline.

**Note:** First run may be slow (Whisper model download). Consider upgrading to a plan with more RAM if videos time out.

---

## Option 2: Render

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. **New** → **Web Service** → connect your repo.
3. Configure:
   - **Build Command:** `docker build -t critic .` (or leave blank; Render uses Dockerfile)
   - **Start Command:** `python server.py`
   - **Instance Type:** Free tier works for small videos; upgrade for longer ones.
4. **Environment** → Add `ANTHROPIC_API_KEY`.
5. Deploy. Render assigns a URL like `https://your-app.onrender.com`.

---

## Option 3: Fly.io

1. Install [flyctl](https://fly.io/docs/hands-on/install-flyctl/).
2. From the project root:
   ```bash
   fly launch
   ```
   Accept defaults; Fly will create `fly.toml`.
3. Add secrets:
   ```bash
   fly secrets set ANTHROPIC_API_KEY=sk-ant-...
   ```
4. Deploy:
   ```bash
   fly deploy
   ```
5. Open `https://your-app.fly.dev`.

---

## Option 4: Run Docker locally (test before deploy)

```bash
# Build
docker build -t critic-pipeline .

# Run (replace with your real API key)
docker run -p 7474:7474 -e ANTHROPIC_API_KEY=sk-ant-... critic-pipeline

# Open http://localhost:7474
```

---

## Environment variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `PORT` | 7474 | Server port (cloud platforms set this) |
| `MAX_FRAMES` | 8 | Max frames sent to Claude Vision |
| `WHISPER_MODEL` | base | `tiny` \| `base` \| `small` \| `medium` \| `large` |
| `REJECT_THRESHOLD` | 50 | Score below which verdict = reject |
| `REFINE_THRESHOLD` | 80 | Score below which verdict = refine |

---

## Resource notes

- **Whisper** uses CPU by default. The `base` model needs ~1–2 GB RAM; first run downloads ~140 MB.
- **Claude Vision** calls cost per image; `MAX_FRAMES=8` keeps costs low.
- Long videos (10+ min) may hit timeouts on free tiers; consider shorter clips or paid plans.
