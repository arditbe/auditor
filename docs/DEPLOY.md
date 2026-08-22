# Deploying Auditor to Google Cloud

Three pieces deploy independently: the API, the dashboard, and (optionally) the
standalone agent on Agent Engine.

Set these once:

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1
gcloud config set project $PROJECT_ID
```

Enable what you need:

```bash
gcloud services enable \
  run.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

## 1. Firestore

Native mode, one database. Runs are one document each in `auditor_runs`.

```bash
gcloud firestore databases create --location=$REGION
```

No index configuration is needed: the only query is
`order_by(created_at_ms desc).limit(n)`, which Firestore serves from the
automatic single-field index.

## 2. The API on Cloud Run

```bash
cd backend
gcloud run deploy auditor-api \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION,GOOGLE_GENAI_USE_VERTEXAI=1,STORE_BACKEND=firestore" \
  --timeout 3600 \
  --session-affinity \
  --cpu 2 --memory 2Gi \
  --min-instances 1
```

Four of those flags are load-bearing:

- **`--timeout 3600`** — a run streams for minutes. The default 5-minute
  request timeout would cut the SSE connection mid-audit.
- **`--session-affinity`** — the replay log lives in process memory, so a run
  and its event stream must land on the same instance. Without this, a
  reconnect can hit an instance that has never heard of the run.
- **`--min-instances 1`** — a cold start in the middle of a demo is
  indistinguishable from a hang.
- **`--cpu 2`** — the orchestrator holds an SSE connection open while awaiting
  model calls.

Give the service account access to Vertex AI and Firestore:

```bash
SA=$(gcloud run services describe auditor-api --region $REGION \
      --format='value(spec.template.spec.serviceAccountName)')

for ROLE in roles/aiplatform.user roles/datastore.user; do
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SA" --role="$ROLE"
done
```

### What the container cannot do

Two capabilities depend on the machine, not on configuration, and both are
detected at runtime rather than assumed:

**No Ollama daemon.** The Local Gemma validator and the Ollama target list are
local-only. In the cloud, use the Vertex AI validators (they light up as soon
as `GOOGLE_CLOUD_PROJECT` is set) and audit deployed endpoints by URL.

**No Apple Silicon.** MLX runs only on Apple hardware, so a Cloud Run instance
cannot fuse or serve MLX adapters. `mlx_available()` checks the platform, and
an upload of adapters returns a plain explanation instead of a stack trace.

**Uploading itself works fine either way.** The drop zone sends files from the
browser, so it does not matter that the container cannot see the user's disk.
What differs is only whether the container can *run* what was uploaded.

Set `AUDITOR_HOME` to a writable path. Cloud Run's filesystem is in-memory and
resets on every new instance, so uploads and prepared models do not persist —
which is correct for a public demo, and means the 400 MB upload cap is also a
memory cap. Raise the instance memory if you raise the cap.

For the demo video, run locally to show the $0 path with a real fine-tune, and
show the Cloud Run service and Firestore documents in the console to prove the
backend is real.

### Locking down the public instance

`POST /api/models/detect` takes a filesystem path and reports what is there. On
a laptop that is the point; on a public URL it lets anyone probe the container's
filesystem. It reveals only whether a path looks like a model, never file
contents — but if you deploy this publicly, either remove that endpoint or put
the service behind IAM. The upload path does not have this property and is the
one users are steered toward.

## 3. The dashboard on Cloud Run

The API base URL is compiled into the bundle, so pass it at build time:

```bash
API_URL=$(gcloud run services describe auditor-api \
            --region $REGION --format='value(status.url)')

cd frontend
gcloud run deploy auditor-web \
  --source . \
  --region $REGION \
  --allow-unauthenticated \
  --build-env-vars "VITE_API_BASE=$API_URL"
```

The API already sends permissive CORS headers, so no extra configuration is
needed for the two services to talk.

## 4. The standalone agent on Agent Engine

`backend/agents/auditor` is the audit packaged as one ADK agent that drives
itself through tools, rather than being orchestrated by Python. This is the
Agent Engine deployable.

Try it locally first:

```bash
cd backend/agents
cp auditor/.env.example auditor/.env   # fill in your project
../.venv/bin/adk web                   # then pick "auditor"
```

Deploy:

```bash
cd backend/agents
../.venv/bin/adk deploy agent_engine \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display_name="Auditor" \
  auditor
```

Update an existing instance instead of creating a second one by passing
`--agent_engine_id=<id>`.

The agent needs the model under test to be reachable over HTTPS — Agent Engine
cannot see anything on your laptop. Give it an endpoint in the prompt:

> Audit the model at `https://my-model-xyz.a.run.app/v1` with six probes. It
> was fine-tuned to summarise discharge notes.

## Costs

- **Local Gemma validator**: free. This is the default and the demo path.
- **Gemini Flash judge**: a 6-probe run is roughly 7 model calls with short
  prompts — cents, not dollars.
- **Cloud Run** with `--min-instances 1` bills continuously. Set it back to `0`
  after the demo.
- **Firestore**: one document per run. Negligible.

The API rate-limits run creation to 10 per 5 minutes per client IP, so a public
demo URL cannot be used to burn credits.

## Troubleshooting

**Stream dies after ~5 minutes.** `--timeout` was not raised, or a proxy in
front is buffering. The API already sends `X-Accel-Buffering: no`.

**Reconnect shows an empty transcript.** Session affinity is off, or the
instance restarted. The dashboard hydrates from Firestore before streaming, so
enabling `STORE_BACKEND=firestore` also makes this recoverable.

**Vertex validators greyed out.** `GOOGLE_CLOUD_PROJECT` is unset. Check
`/api/health` — it reports `vertex_configured`.

**`403` on a Vertex model.** The model is not enabled in your region, or the
service account is missing `roles/aiplatform.user`. Gemma and MedGemma need to
be enabled from Model Garden before first use.
