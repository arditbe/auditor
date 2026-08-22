# Google Cloud free-tier deploy

This is the low-cost demo path for Auditor. It deploys two Cloud Run services:

- `auditor-api` — FastAPI backend, locked to HTTPS model endpoints
- `auditor-web` — static React dashboard, built without model upload controls

It avoids Firestore, Vertex AI, and always-on instances by default. The app uses
the Google AI Studio API-key Gemini validators instead.

> Free tier is not the same thing as "no billing account." Cloud Run normally
> needs billing enabled, then applies its monthly free tier. Set a budget alert
> before deploying.

## 0. Pick a project and region

Use a free-tier Cloud Run region. `us-central1` is the safest default.

```bash
export PROJECT_ID=your-google-cloud-project-id
export REGION=us-central1

gcloud config set project "$PROJECT_ID"
gcloud config set run/region "$REGION"
```

Enable only the services this deploy uses:

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

## 1. Store your Gemini API key

Get a key from <https://aistudio.google.com/apikey>, then store it as a Cloud
Run secret:

```bash
printf "PASTE_YOUR_GOOGLE_AI_STUDIO_KEY_HERE" | \
  gcloud secrets create auditor-google-api-key --data-file=-
```

If the secret already exists:

```bash
printf "PASTE_YOUR_GOOGLE_AI_STUDIO_KEY_HERE" | \
  gcloud secrets versions add auditor-google-api-key --data-file=-
```

Allow Cloud Run to read the secret:

```bash
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding auditor-google-api-key \
  --member="serviceAccount:${RUNTIME_SA}" \
  --role="roles/secretmanager.secretAccessor"
```

## 2. Deploy the API

This keeps cost down:

- `--min-instances 0` lets the service scale to zero.
- `--cpu 1 --memory 512Mi` stays lightweight.
- `STORE_BACKEND=memory` avoids Firestore.
- `GOOGLE_GENAI_USE_VERTEXAI=0` keeps Gemini on the API-key path.
- `AUDITOR_CLOUD_DEMO=1` disables model upload, prepared-model, and local-path
  APIs on the public backend.

```bash
cd backend

gcloud run deploy auditor-api \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --cpu 1 \
  --memory 512Mi \
  --timeout 900 \
  --session-affinity \
  --set-env-vars "STORE_BACKEND=memory,GOOGLE_GENAI_USE_VERTEXAI=0,AUDITOR_CLOUD_DEMO=1" \
  --set-secrets "GOOGLE_API_KEY=auditor-google-api-key:latest"
```

Check it:

```bash
API_URL=$(gcloud run services describe auditor-api \
  --region "$REGION" \
  --format="value(status.url)")

curl "$API_URL/api/health"
curl "$API_URL/api/models/validator"
```

The `Gemini 3 Flash (API key)` and `Gemini 3 Pro (API key)` validators should
show `"available": true`. The Vertex AI cards can remain unavailable; that is
expected for this free-tier setup.

## 3. Deploy the dashboard

Build the frontend with the API URL baked in:

```bash
cd ../frontend

gcloud run deploy auditor-web \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --cpu 1 \
  --memory 256Mi \
  --build-env-vars "VITE_API_BASE=$API_URL,VITE_CLOUD_DEMO=1"
```

Open it:

```bash
WEB_URL=$(gcloud run services describe auditor-web \
  --region "$REGION" \
  --format="value(status.url)")

echo "$WEB_URL"
```

## What works in this deploy

Works:

- dashboard
- API
- validator context CSV/TXT/JSON upload
- Gemini API-key judges
- auditing an HTTPS model endpoint

Hidden or disabled in this web version:

- model upload
- local filesystem paths
- prepared model management
- local Ollama models are not available
- local Gemma validator is not available
- MLX adapter fusing does not work because Cloud Run is not Apple Silicon
- uploaded/prepared models do not persist because the free-tier setup uses
  memory storage

For a hackathon demo, use the cloud URL to show the product works online, and
use your Mac for the local model/adapters demo.

## Stop spending

Cloud Run with `min-instances 0` should scale to zero when unused, but you can
delete the demo services after judging:

```bash
gcloud run services delete auditor-web --region "$REGION"
gcloud run services delete auditor-api --region "$REGION"
```

You can also delete the secret:

```bash
gcloud secrets delete auditor-google-api-key
```
