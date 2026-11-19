# Deploying the assistant

This is a practical path for putting the app online for a small audience (a portfolio
demo, a team of a few people). Nothing here has been provisioned: the steps are written
to be followed by hand, and every paid resource is called out. The container itself is
built and smoke-tested by CI (`.github/workflows/ci.yml`, job `docker`).

## What the container needs

| Resource | Value | Why |
|---|---|---|
| CPU | 2 vCPU | the cross-encoder reranker is the hot path (~2-4 s/query on 2 cores, see `results/v2/bench/`) |
| Memory | 2 GiB | ~600 MB for the two models + index; headroom for request buffers |
| Disk | image ~2 GB | CPU-only torch, models, dataset file and the built index are baked in |
| Network out | only to the LLM API (if any) | runtime sets `HF_HUB_OFFLINE=1`; nothing is downloaded at start-up |
| Port | 7860 | `uvicorn server:app` |

One process per container (`--workers 1`): the models live in process memory. Scale by
adding containers, not workers.

## The LLM in the cloud

Ollama on a laptop is the default for local use, but most hosts have no GPU. Options:

1. **An OpenAI-compatible API** (Groq, Together, OpenAI): set `LLM_PROVIDER=openai`,
   `OPENAI_BASE_URL`, `LLM_MODEL` and the key as a secret (below). The server always uses
   its configured model for paid APIs; callers can't pick an expensive one.
2. **Anthropic**: `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`.
3. **No LLM**: `LLM_PROVIDER=extractive`. Answers become quoted sentences with citations.
   Free, and `/readyz` reports `degraded` instead of `ready` so it's visible.

The round-2 answer-quality numbers in the README were measured with qwen3:8b. A different
model needs its own run of `answer_eval.py` before you quote numbers for it.

## Option A: Google Cloud Run (recommended)

Cloud Run fits this app: one container, scales to zero, HTTPS and request logs included.
It has a monthly free tier, but a demo that stays warm (`--min-instances 1`) or gets real
traffic **will be billed**. Check current pricing first.

```bash
PROJECT=my-project REGION=us-central1 SERVICE=paper-rag
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# 1. Image -> Artifact Registry (build in Cloud Build, or locally and push)
gcloud artifacts repositories create rag --repository-format=docker --location=$REGION
gcloud builds submit --tag $REGION-docker.pkg.dev/$PROJECT/rag/$SERVICE:$(git rev-parse --short HEAD)

# 2. Secrets: never as plain env vars, never in the image
printf '%s' "$GROQ_KEY"  | gcloud secrets create llm-api-key --data-file=-
printf '%s' "$(openssl rand -hex 24)" | gcloud secrets create app-api-keys --data-file=-

# 3. Deploy
gcloud run deploy $SERVICE \
  --image $REGION-docker.pkg.dev/$PROJECT/rag/$SERVICE:$(git rev-parse --short HEAD) \
  --region $REGION --cpu 2 --memory 2Gi --concurrency 4 --timeout 300 \
  --min-instances 0 --max-instances 2 --cpu-boost \
  --set-env-vars LLM_PROVIDER=openai,OPENAI_BASE_URL=https://api.groq.com/openai/v1,LLM_MODEL=llama-3.3-70b-versatile \
  --set-secrets OPENAI_API_KEY=llm-api-key:latest,API_KEYS=app-api-keys:latest \
  --allow-unauthenticated          # app-level key auth below; see "Authentication" for IAM-only
```

`--max-instances` caps the bill and the LLM spend; `--concurrency 4` matches
`MAX_CONCURRENT_GENERATIONS=2` plus streaming headroom.

Point Cloud Run's **startup probe** at `/readyz` (it returns 503 until the models are
loaded) and the **liveness probe** at `/healthz`:

```bash
gcloud run services update $SERVICE --region $REGION \
  --startup-probe httpGet.path=/readyz,initialDelaySeconds=10,periodSeconds=5,failureThreshold=30 \
  --liveness-probe httpGet.path=/healthz,periodSeconds=30
```

## Option B: any VM or Docker host

```bash
docker build -t paper-rag .
docker run -d --restart unless-stopped -p 127.0.0.1:7860:7860 \
  --env-file /etc/paper-rag.env  paper-rag     # file is root-only (chmod 600), holds the keys
```

Put a reverse proxy with TLS in front (Caddy: `reverse_proxy 127.0.0.1:7860`). Turn off
response buffering for `/api/ask` so tokens stream (the app already sends
`X-Accel-Buffering: no` for nginx).

## Option C: Hugging Face Spaces

The README header is a Spaces config (Docker SDK, port 7860). Docker Spaces need a paid
(PRO) account as of Oct 2026. Add the keys as Space **secrets**, not variables.

## Authentication

Two layers, use at least one:

- **App-level keys (built in).** `API_KEYS=key1,key2` makes every `/api/*` call require
  `Authorization: Bearer <key>` (or `X-API-Key`). `/`, `/static`, `/healthz` and `/readyz`
  stay open, so the page loads and the probes work; the page asks for the key once and
  keeps it in the browser's localStorage. Comparison is constant-time. Rotate by adding the
  new key, redeploying, telling users, then removing the old one.
- **Platform auth (stronger).** Deploy Cloud Run *without* `--allow-unauthenticated` and
  put Identity-Aware Proxy (or Cloudflare Access on a VM) in front, so only signed-in
  Google accounts you list can reach the app at all.

App-level keys are shared secrets with no per-user identity or rate limit. For anything
beyond a demo, use platform auth and per-user quotas at the proxy.

## Secrets handling

- Secrets live in the platform's secret store (Secret Manager, Spaces secrets) or a
  root-only env file, and reach the app only as environment variables at runtime.
- `.env` is git-ignored and `.dockerignore`d, so it never ends up in the repo or the image.
- The app never sends keys to the browser: `/api/models` lists providers, not keys.
- Logs never contain keys or prompts; the request log has ids, timings and outcomes only.

## Monitoring

The server writes one JSON object per line to stdout. Cloud Logging parses these into
`jsonPayload` automatically. The useful events:

| `event` | Fields | Use |
|---|---|---|
| `startup` | `status` (`ready`/`not_ready`), `error`, `n_chunks` | alert on `not_ready` |
| `http` | `request_id`, `path`, `status`, `ms` | error rate, latency per endpoint |
| `ask` | `request_id`, `outcome` (`answered`, `refused`, `error`, `busy`, `client_disconnected`), `reason`, `error_code`, `retrieval_s`, `dense_s`, `bm25_s`, `rerank_s`, `ttft_s`, `generation_s`, `total_s` | answer quality signals and per-stage latency |

Suggested alerts (Cloud Monitoring log-based metrics, or any log tool):

1. **Uptime check** on `/healthz` every minute; page if it fails 3 times in a row.
2. `event="ask" AND outcome="error"`: more than 5% of asks in 10 minutes.
   `error_code` says whether it's the LLM (`llm_unavailable`, `llm_timeout`,
   `llm_bad_response`) or the app (`internal`).
3. `event="ask"`: p95 of `total_s` above 30 s for 15 minutes (LLM provider slow).
4. `event="ask" AND outcome="busy"`: any, sustained, means `MAX_CONCURRENT_GENERATIONS`
   or `--max-instances` is too low for the traffic.
5. A jump in `reason="low_relevance"` or `reason="needs_paper"` usually means users ask
   about something the index doesn't cover, not that the service is broken.

To trace one bad answer, take the `X-Request-ID` the browser received (it is also in every
streamed event) and filter the logs by `jsonPayload.request_id`.

## Rollback

Every deploy is an immutable image tagged with the git commit.

- **Cloud Run.** Each deploy creates a revision. To roll back instantly:
  ```bash
  gcloud run revisions list --service $SERVICE --region $REGION
  gcloud run services update-traffic $SERVICE --region $REGION --to-revisions <previous-revision>=100
  ```
  For risky changes, deploy with `--no-traffic --tag canary`, test the tagged URL, then
  shift traffic gradually (`--to-tags canary=10`, then 100).
- **Docker host.** Keep the previous image tag; `docker run` it again with the same env file.
- **Index changes** are part of the image, so rolling back the image rolls back the index
  too. Nothing outside the container holds state.
- **Model changes** (embedding or reranker) change the index manifest. The server refuses
  to start against an index built with a different embedding model, so a half-rolled-back
  state fails loudly at `/readyz` instead of serving bad results.

## Pre-launch checklist

- [ ] `API_KEYS` set or platform auth on; `--max-instances` set
- [ ] LLM key in the secret store; `/readyz` shows `ready` (not `degraded`)
- [ ] uptime check + error-rate alert created
- [ ] previous revision noted for rollback
- [ ] `python answer_eval.py` re-run if the LLM differs from the evaluated one
