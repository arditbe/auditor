# Auditor

An agent that audits your fine-tuned model — on its own, on a schedule.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arditbe/auditor/blob/main/notebooks/auditor_colab.ipynb)
[![Run in the terminal](https://img.shields.io/badge/Run_in_the_terminal-%E2%80%BA_bash_scripts%2Fcolab.sh-1D6A7A?style=flat&logo=gnubash&logoColor=white)](#terminal--colab)
[![Tests](https://img.shields.io/badge/tests-206_passing-2F6B4F?style=flat)](#)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?style=flat&logo=python&logoColor=white)](#quick-start)

**Colab** needs nothing but a free [Gemini API key](https://aistudio.google.com/apikey) — no install, no Google Cloud project.
**Terminal** runs the same audit locally, against Ollama or your own endpoint.

Point Auditor at a model. It writes its own test questions, puts them to the
model one at a time, and has Gemini score every answer against criteria it
committed to *before* seeing the response.

Nothing is pre-scripted. There is no fixed benchmark to overfit to — the probe
set is generated per run, for the model and the purpose you describe.

Then it keeps going without you:

- **It decides where to dig.** When a dimension scores badly, the agent writes
  harder probes aimed at that weakness and runs those too. A 3-probe audit
  becomes 9 because the agent judged it worth the effort.
- **It decides whether to raise the alarm.** Each run is compared against the
  last audit of the same model, and a real drop is told apart from judge noise.
- **It fixes what it finds.** Every failure becomes a training example with a
  corrected answer, written by the judge — a dataset you can fine-tune on.
- **It runs while you sleep.** Schedule an audit hourly, daily or weekly and
  nobody needs to open the dashboard again.

```
                    ┌──────────────────────────────────────┐
   Cloud Scheduler ─┤  POST /api/scheduled/tick   (nightly)│
                    └───────────────┬──────────────────────┘
                                    │ which watches are due?
   browser ──REST/SSE──┐            ▼
                       ▼   ┌────────────────────────────────┐
              ┌────────────┤   Cloud Run  ·  FastAPI        │
              │ dashboard  │   orchestrator + watch loop    │
              └────────────┤                                │
                           │   ┌────────────────────────┐   │
                           │   │      Google ADK        │   │
                           │   │  probe generator       │   │
                           │   │  judge                 │   │
                           │   └──────────┬─────────────┘   │
                           └──────────────┼─────────────────┘
                                          │
            ┌─────────────────────────────┼──────────────────────┐
            ▼                             ▼                      ▼
   ┌─────────────────┐        ┌────────────────────┐   ┌──────────────────┐
   │ model under test│        │ Gemini 3.5 Flash   │   │    Firestore     │
   │ Ollama · MLX    │        │ (Vertex AI, global)│   │ runs · watches   │
   │ · HTTPS endpoint│        └────────────────────┘   └──────────────────┘
   └─────────────────┘                                          │
                                                                ▼
                                                    training data (.jsonl/.csv)
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

Which model does the judging. The local option costs nothing; the Vertex AI
ones need `GOOGLE_CLOUD_PROJECT` set, and the API-key ones need a free
[AI Studio key](https://aistudio.google.com/apikey).

| Option | Runs on | Cost |
|---|---|---|
| Local Gemma 3 (Ollama) | your machine | free |
| Gemini 3.5 Flash | Vertex AI | paid |
| Gemini 3.1 Pro | Vertex AI | paid |
| Gemini 3.5 Flash-Lite | Vertex AI | paid, cheapest |
| Gemini 3.5 Flash / 3.1 Pro | AI Studio key | paid |

> Gemma 3 27B and MedGemma are **not** offered. On Vertex they are not
> serverless — they need a Model Garden endpoint you deploy and pay for by the
> hour — so listing them would mean offering a button that always fails. The
> local Gemma 3 judge is real Gemma and works out of the box.

You can **switch validator mid-run**. Probes already scored keep their original
judge, and the report records every validator that contributed. That is the
point: start free on local Gemma, and when a verdict looks wrong, escalate the
remaining probes to Gemini without restarting.

Adding another Google model is one entry in
[`backend/app/providers/registry.py`](backend/app/providers/registry.py).

## Models under test

You should not need to know what Ollama is to audit your own model. The
dashboard has a drop zone: pick the folder your training run saved, and Auditor
works out the rest.

Uploading is the primary path because it works the same whether Auditor runs on
your laptop or on Cloud Run — the files come from the browser, not from a
filesystem the server may not share.

| What you have | What Auditor does |
|---|---|
| A folder of **MLX LoRA adapters** | Fuses them into their base model and serves the result |
| A **`.gguf` file** | Registers it with Ollama (no copy) |
| A **complete model folder** | Serves it locally |
| A **model server URL** | Sends probes straight to it |
| A model **already in Ollama** | Picks it from the list |

Only the files that matter are sent. MLX writes periodic checkpoints
(`0000200_adapters.safetensors`) that the final adapter supersedes, so a 60 MB
folder uploads as 21 MB. The base model is named in `adapter_config.json`
rather than shipped, which is what keeps this small enough to be practical.

Detection never guesses silently. If a run saved no weights, or the adapters do
not name their base model, Auditor says exactly that instead of failing
halfway through.

Pasting a filesystem path is still available under **Use a path or a server URL
instead** — but note it only works when Auditor is running on the same machine
as the model.

### Keeping a model in Ollama (optional)

A prepared model is served by Auditor on demand. If you would rather Ollama own
it permanently, use **Keep in Ollama**: the model is de-quantized, converted to
GGUF, and registered as a normal Ollama tag.

This needs llama.cpp's `convert_hf_to_gguf.py`. Point `LLAMA_CPP_PATH` at a
checkout, or leave it — the button is hidden when the converter is missing, and
auditing works identically without it.

> `mlx_lm` has its own `--export-gguf`, and Auditor deliberately does not use
> it. In 0.31 it writes every tensor with shape `(0,)` — a silently empty
> model. An auditing tool must never hand you a corrupt export.

## Running on its own

The dashboard is for watching one audit. A **watch** is a standing instruction:
audit this model on a schedule, and tell me when it gets worse.

Open **Scheduled** in the top bar and set what to audit, how often, and which
Gemini model judges. From then on nobody has to open the app.

| | |
|---|---|
| Cadence | hourly, daily or weekly, at a chosen UTC hour |
| Regression | flagged when the score drops more than `REGRESSION_DROP` points |
| Training data | built from the failures, optionally only when it regresses |

Schedules live in Firestore rather than as individual Cloud Scheduler jobs.
One scheduler job pings `/api/scheduled/tick` and the backend decides which
watches are due. That way adding a watch needs no Google Cloud permissions and
creates no infrastructure — and the same code runs off a plain cron entry.

```bash
# what Cloud Scheduler calls; safe to run by hand
curl -X POST https://<your-service>/api/scheduled/tick
```

### What the agent decides for itself

Given a weak result, it does not stop and wait to be asked:

```
[opening]  3 probes across dimensions
>>> hallucination resistance scored 0.00/5 — going deeper
[round 1]  3 harder probes, all hallucination
>>> instruction following scored 1.00/5 — going deeper
[round 2]  3 harder probes, all instruction following
```

Capped at `ADAPTIVE_MAX_ROUNDS` so a bad model cannot loop forever, and it
never drills the same dimension twice.

### Training data from failures

Every failed probe becomes a training example whose completion is the answer
the model *should* have given, written by the judge that failed it:

```jsonl
{"prompt": "Explain how the treaty signed at the 2021 Denver Peace Conference…",
 "completion": "The premise of this question is incorrect. There was no 2021
                Denver Peace Conference…"}
```

Two files per run: `.jsonl` for a trainer, `.csv` with the rejected answer and
the reason it failed, so you can check the corrections before training on them.

## Terminal / Colab

No browser needed. Two ways in:

**Colab** — click the badge at the top. It clones the repo, takes a Gemini key
with hidden input so it never lands in notebook output, installs Ollama with a
small model to test against, and prints a scored audit.

**Your own machine** — `scripts/colab.sh` does the same thing locally:

```bash
bash scripts/colab.sh
```

It creates a venv, prompts for the key and the target, then runs. Everything is
skippable with environment variables:

```bash
GOOGLE_API_KEY=... TARGET=https://my-model.example.com/v1 PROBES=8 \
  bash scripts/colab.sh
```

`WITH_OLLAMA=1` installs Ollama and pulls a small model to audit.

### The CLI directly

```bash
cd backend
.venv/bin/python -m app.cli --target ollama:qwen2:0.5b --probes 6
.venv/bin/python -m app.cli --list-validators
```

| Flag | |
|---|---|
| `--target` | `ollama:<tag>`, an `https://` URL, or `prepared:<name>` |
| `--validator` | judge model; `gemini-flash-key` needs `GOOGLE_API_KEY` |
| `--probes` | how many questions (default 6) |
| `--suite` | `general`, `medical`, `code`, `safety` |
| `--purpose` | what the model was tuned for; steers the probes |
| `--min-score` | exit 2 below this, so it works as a CI gate |
| `--full` | print answers in full instead of truncating |

Exit codes are meaningful: `0` finished, `1` failed, `2` scored below
`--min-score`.

### Gemini without Google Cloud

The `*-key` validators take a plain Google AI Studio key — no `gcloud`, no
project, no application-default credentials. Get one free at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey) and set
`GOOGLE_API_KEY`, or paste it into Settings in the dashboard.

The key is held only by the running process and never written to disk.

## What works locally vs. deployed

These are genuinely different environments, and it is worth being explicit
rather than discovering it during a demo.

| | Local (your Mac) | Cloud Run |
|---|---|---|
| Upload a model folder | yes | yes |
| Detect what it is | yes | yes |
| **Fuse and run MLX adapters** | **yes** | **no — needs Apple Silicon** |
| Ollama models | yes | no — no daemon in the container |
| Audit a model server URL | yes | yes |
| Local Gemma validator | yes | no |
| Gemini / Gemma / MedGemma validators | with a GCP project | yes |
| Firestore history | with a GCP project | yes |

The deployed service is therefore an auditor **of deployed endpoints**, judged
by Gemini API-key or Vertex AI models. The laptop is where local weights get
audited, because that is where the hardware to run them is.

For a demo: run locally to show a real fine-tune being audited at zero cost,
and show the Cloud Run service and Firestore documents to prove the backend is
real. Use [`docs/GOOGLE_CLOUD_FREE_TIER.md`](docs/GOOGLE_CLOUD_FREE_TIER.md)
for the cheapest Cloud Run demo; it builds a URL-only web version with model
uploads hidden and disabled. Use [`docs/DEPLOY.md`](docs/DEPLOY.md) for the
full Vertex AI + Firestore version.

An upload of MLX adapters to a deployed instance is detected correctly and then
says so plainly — *"Your adapters uploaded fine, but this server cannot fuse MLX
models — that needs Apple Silicon"* — rather than failing somewhere confusing.

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
| `GOOGLE_CLOUD_LOCATION` | `global` | Gemini 3.x is served only from the global endpoint |
| `STORE_BACKEND` | `memory` | `firestore` to persist runs |
| `FIRESTORE_COLLECTION` | `auditor_runs` | Collection name |
| `OLLAMA_HOST` | `http://localhost:11434` | Local daemon |
| `OLLAMA_KEEP_ALIVE` | `20m` | Keeps both models resident (see below) |
| `TARGET_TIMEOUT_S` | `90` | A slower answer counts as a failed probe |
| `PROBE_DELAY_S` | `0` | Pause between probes, to slow a live demo down |
| `ADAPTIVE_PROBING` | `true` | Let the agent drill into weaknesses it finds |
| `ADAPTIVE_THRESHOLD` | `3.0` | Score out of 5 below which a dimension is weak |
| `ADAPTIVE_MAX_ROUNDS` | `2` | Follow-up rounds allowed per run |
| `REGRESSION_DROP` | `10.0` | Points a score must fall to count as a regression |
| `AUDITOR_HOME` | `~/.auditor` | Where prepared models and the registry live |
| `LLAMA_CPP_PATH` | — | llama.cpp checkout, enables GGUF export |

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
