# Auditor

Real-time validation for fine-tuned language models.

Point Auditor at a model. It writes its own test questions, puts them to the
model one at a time, and has a second model score every answer against criteria
it committed to *before* seeing the response. You watch the whole interrogation
happen live, and get a report at the end.

Nothing is pre-scripted. There is no fixed benchmark to overfit to — the probe
set is generated per run, for the model and the purpose you describe.

```
┌─ browser ────────────┐   SSE    ┌─ Cloud Run ──────────────┐
│ live transcript      │ ◀─────── │ FastAPI  ── orchestrator │
│ meter bridge         │          │              │           │
│ score / final report │ ──REST─▶ │        Google ADK        │
└──────────────────────┘          │      ┌───────┴────────┐  │
                                  │  probe generator   judge │
                                  └───────┬────────────┬─────┘
                                          │            │
                              ┌───────────▼──┐   ┌─────▼──────────┐
                              │ model under  │   │ validator:     │
                              │ test (Ollama │   │ local Gemma or │
                              │ or endpoint) │   │ Gemini/MedGemma│
                              └──────────────┘   └────────────────┘
                                          Firestore ◀── run history
```

## What it measures

Five dimensions, scored 0–5 per probe, weighted into a single 0–100 figure:

| Dimension | Weight | What a low score means |
|---|---|---|
| `accuracy` | 0.30 | It gets facts wrong. |
| `hallucination_resistance` | 0.25 | It confidently answers questions with no answer. |
| `instruction_following` | 0.20 | It ignores explicit format constraints. |
| `safety` | 0.15 | It complies with things it should refuse. |
| `coherence` | 0.10 | Its output is malformed or off-topic. |

Roughly one probe in four is a **trap** — unanswerable, false-premised, or
asking for a fact that does not exist. Declining scores high; answering anyway
scores near zero. This is where weak models fail hardest and where a fixed
benchmark tells you nothing.

A dimension no probe tested reads **`not tested`**, never `0`. The instrument
does not report a measurement it has not taken.

## Quick start

Requires Python 3.12+, Node 22+, and [Ollama](https://ollama.com) running.

```bash
# 1. Judge model (8 GB) and something small to audit
ollama pull gemma3:12b
ollama pull qwen2:0.5b

# 2. Backend
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --port 8000 --reload

# 3. Frontend, in a second terminal
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>, pick a model, and start an audit. Everything runs
locally and costs nothing — no Google Cloud project needed to try it.

Run the tests with `cd backend && .venv/bin/python -m pytest`.

## Validator context files

On the audit setup screen, **What was this model tuned for?** accepts either
typed text or an uploaded file. Upload a CSV, JSON, Markdown, TXT, or another
UTF-8 text file containing the tuning description, examples, or source data.
Auditor extracts the text and passes it to the validator when it designs the
probe set, so the validator tests the model against what the file says it was
tuned for.

CSV files are not sent to the validator in full. Small CSVs are flattened row
by row; large CSVs are broken into representative chunks from across the file
with the column names and total row count included. Other large text files are
sampled from the beginning, middle, and end so probe generation stays
responsive without spending the whole context window on one upload.

## Validators

The dropdown picks which model does the judging. Local costs nothing; the
Vertex AI options need `GOOGLE_CLOUD_PROJECT` set.

| Option | Runs on | Cost |
|---|---|---|
| Local Gemma 3 (Ollama) | your machine | free |
| Gemini 3 Flash | Vertex AI | paid |
| Gemini 3 Pro | Vertex AI | paid |
| Gemma 3 27B | Vertex AI | paid |
| MedGemma 27B | Vertex AI | paid |

You can **switch validator mid-run**. Probes already scored keep their original
judge, and the report records every validator that contributed. That is the
point: start free on local Gemma, and when a verdict looks wrong, escalate the
remaining probes to Gemini without restarting.

Adding another Google model is one entry in
[`backend/app/providers/registry.py`](backend/app/providers/registry.py).

## Models under test

Today the dropdown lists whatever Ollama has pulled locally — point a
fine-tuned model at Ollama and it appears. To audit a model deployed behind an
HTTP API instead, `HttpEndpointTarget` already speaks the OpenAI-compatible
`/chat/completions` shape; construct it with `build_target("http:...")`. The
agent is indifferent to which one it is talking to.

## What works locally vs. deployed

These are genuinely different environments, and it is worth being explicit
rather than discovering it during a demo.

| | Local (your Mac) | Cloud Run |
|---|---|---|
| Ollama models | yes | no — no daemon in the container |
| Audit a model server URL | yes | yes |
| Local Gemma validator | yes | no |
| Gemini / Gemma / MedGemma validators | with a GCP project | yes |
| Firestore history | with a GCP project | yes |

The deployed service is therefore an auditor **of deployed endpoints**, judged
by Vertex AI models. The laptop is where local Ollama models get audited,
because that is where the local daemon runs.

For a demo: run locally to show a real fine-tune being audited at zero cost,
and show the Cloud Run service and Firestore documents to prove the backend is
real. `docs/DEPLOY.md` covers both.

## Where Google Cloud fits

- **Google ADK** drives both agents. The probe generator and the judge are ADK
  `LlmAgent`s; the validator dropdown resolves to either a native Gemini model
  or a `LiteLlm`-wrapped one, which is what lets the same code run on local
  Gemma and on Vertex AI.
- **Vertex AI Agent Engine** — `backend/agents/auditor` is the audit packaged
  as a single deployable ADK agent. See [docs/DEPLOY.md](docs/DEPLOY.md).
- **Firestore** stores run history. Set `STORE_BACKEND=firestore`. If it is
  unreachable the service logs a warning and falls back to memory rather than
  failing the run.
- **Cloud Run** hosts the API and the dashboard. Dockerfiles for both are in
  the repo.

## Configuration

All env vars, with defaults:

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | — | Enables the Vertex AI validators |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex region |
| `STORE_BACKEND` | `memory` | `firestore` to persist runs |
| `FIRESTORE_COLLECTION` | `auditor_runs` | Collection name |
| `OLLAMA_HOST` | `http://localhost:11434` | Local daemon |
| `OLLAMA_KEEP_ALIVE` | `20m` | Keeps both models resident (see below) |
| `TARGET_TIMEOUT_S` | `90` | A slower answer counts as a failed probe |
| `PROBE_DELAY_S` | `0` | Pause between probes, to slow a live demo down |

## Notes from building it

**Model swap thrashing.** The judge and the model under test share one Ollama
daemon and alternate every probe. Without `keep_alive`, Ollama evicts whichever
ran last and every turn pays a full model load — an 8 GB reload looks exactly
like a hung model. If probes are timing out locally, that is almost always why.

**A run survives partial failure.** If the target errors, or the judge returns
unparseable output, that probe is recorded as a failure with the reason and the
run continues. An auditor that aborts on first error cannot audit a bad model,
which is the case it exists for.

**JSON by instruction, not schema.** Both agents are asked for bare JSON rather
than using ADK's `output_schema`, because the same prompt has to work on Gemini
*and* on a local Gemma through LiteLlm. `app/agent/parsing.py` handles fenced
blocks, preambles, and trailing commas — but returns `None` rather than
inventing data when nothing parses.

**Speed.** On an M-series Mac with `gemma3:12b` judging, budget ~30 s to design
the probe set and ~10 s per probe. Six probes lands around 90 seconds. Gemini
Flash is considerably faster if you have credits.

## Layout

```
backend/
  app/
    main.py            FastAPI: REST + SSE
    orchestrator.py    the run loop; emits every event
    events.py          in-process pub/sub with per-run replay log
    agent/             ADK agents, prompts, tolerant JSON parsing
    providers/         models under test + the validator registry
    store/             Firestore, with an in-memory fallback
  agents/auditor/      the same audit, as a deployable Agent Engine agent
  tests/
frontend/
  src/
    hooks/useRunStream.js   folds the SSE stream into render state
    components/             transcript, meter bridge, score, report
```

## Known limits

- The SSE replay log is per-process, so a run and its stream must be served by
  the same Cloud Run instance. `docs/DEPLOY.md` covers the setting that
  guarantees this, and the Firestore-listener alternative if you outgrow it.
- The judge is a language model. It is a good relative instrument — it reliably
  ranks a weak model below a strong one — but do not read a single probe's
  score as ground truth.
- One run at a time per browser tab; there is no run queue.
