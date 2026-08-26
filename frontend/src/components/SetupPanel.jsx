import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { ModelPicker } from './ModelPicker'
import { JudgePicker } from './JudgePicker'
import { PurposeField } from './PurposeField'

/**
 * The setup flow.
 *
 * Three numbered steps, each one decision. Everything that is not a decision
 * — probe count, suite, what the model was tuned for — is folded into step
 * three, which has a sensible default and can be ignored entirely.
 */
export function SetupPanel({
  onStarted,
  onOpenSettings,
  reloadKey = 0,
  // null while the server is still being asked what it can do.
  cloudMode = null,
}) {
  const [installed, setInstalled] = useState([])
  const [owned, setOwned] = useState([])
  const [validators, setValidators] = useState([])
  const [suites, setSuites] = useState([])
  const [installError, setInstallError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const [form, setForm] = useState({
    target_model: '',
    validator_model: 'local-gemma',
    suite: 'general',
    num_probes: 6,
    model_purpose: '',
  })

  const load = () =>
    Promise.all([
      cloudMode
        ? Promise.resolve({ models: [], error: null })
        : api.targetModels().catch(() => ({ models: [], error: null })),
      api.validators(),
      api.suites(),
      cloudMode
        ? Promise.resolve({ models: [] })
        : api.preparedModels().catch(() => ({ models: [] })),
    ])

  useEffect(() => {
    if (cloudMode === null) return undefined
    let cancelled = false
    load()
      .then(([t, v, s, p]) => {
        if (cancelled) return
        const local = t.models.filter((m) => m.is_local)
        setInstalled(local)
        setInstallError(t.error)
        // Every judge is offered in both modes; the server already marks
        // each one available or not. Cloud mode changes how you pick a
        // *model*, not who is allowed to judge -- and on Cloud Run the Vertex
        // judges are the ones that work, so filtering them out left nothing.
        const shownValidators = v.validators
        setValidators(shownValidators)
        setSuites(s.suites)
        setOwned(p.models)
        setForm((f) => {
          const current = shownValidators.find((x) => x.key === f.validator_model)
          const preferredJudge =
            current?.available
              ? f.validator_model
              : (
                  shownValidators.find((x) => x.available && x.requires === 'api_key')?.key ??
                  shownValidators.find((x) => x.available)?.key ??
                  shownValidators[0]?.key ??
                  f.validator_model
                )
          return {
            ...f,
            // Prefer a model the person prepared themselves: if they went to the
            // trouble of adding one, that is what they came to audit.
            target_model: f.target_model || p.models[0]?.spec || local[0]?.spec || '',
            validator_model: preferredJudge,
          }
        })
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [reloadKey, cloudMode])

  const refresh = () =>
    load().then(([t, v, s, p]) => {
      setInstalled(t.models.filter((m) => m.is_local))
      setValidators(v.validators)
      setOwned(p.models)
    })

  const onModelAdded = (model) => {
    setOwned((list) =>
      list.some((m) => m.spec === model.spec) ? list : [model, ...list],
    )
    setForm((f) => ({ ...f, target_model: model.spec }))
    refresh()
  }

  const removeOwned = async (name) => {
    try {
      await api.deletePrepared(name, true)
      setForm((f) =>
        f.target_model === `prepared:${name}` ? { ...f, target_model: '' } : f,
      )
      refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  const set = (key) => (e) => {
    const value = e.target.type === 'number' ? Number(e.target.value) : e.target.value
    setForm((f) => ({ ...f, [key]: value }))
  }

  const submit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const { run_id } = await api.startRun(form)
      onStarted(run_id)
    } catch (err) {
      setError(err.message)
      setSubmitting(false)
    }
  }

  const chosenModel =
    owned.find((m) => m.spec === form.target_model)?.name ??
    installed.find((m) => m.spec === form.target_model)?.name ??
    form.target_model.replace(/^(ollama|prepared):/, '')

  const suiteLabel = suites.find((s) => s.key === form.suite)?.label ?? form.suite
  const chosenValidator = validators.find((v) => v.key === form.validator_model)
  const ready = Boolean(form.target_model) && Boolean(chosenValidator?.available) && !loading

  return (
    <form onSubmit={submit}>
      <div className="hero">
        <h1>What should we audit?</h1>
        <p>
          Auditor writes its own test questions, puts them to your model, and has
          a second model score every answer against criteria it wrote first.
          Nothing is pre-scripted.
        </p>
      </div>

      <section className="step">
        <div className="step-head">
          <span className="step-num">1</span>
          <h2>Choose a model</h2>
        </div>
        <p className="step-sub">
          {cloudMode
            ? 'Paste the HTTPS URL for a running model server.'
            : 'Drop the folder your training saved, or pick one already on this machine.'}
        </p>
        <div className="step-body">
          <ModelPicker
            installed={installed}
            owned={owned}
            selected={form.target_model}
            onSelect={(spec) => setForm((f) => ({ ...f, target_model: spec }))}
            onAdded={onModelAdded}
            onRemoveOwned={removeOwned}
            installError={installError}
            cloudMode={Boolean(cloudMode)}
          />
        </div>
      </section>

      <section className="step">
        <div className="step-head">
          <span className="step-num">2</span>
          <h2>Choose a judge</h2>
        </div>
        <p className="step-sub">
          The model that scores the answers. You can switch mid-audit.
        </p>
        <div className="step-body">
          <JudgePicker
            validators={validators}
            selected={form.validator_model}
            onSelect={(key) => setForm((f) => ({ ...f, validator_model: key }))}
            onNeedKey={onOpenSettings}
          />
        </div>
      </section>

      <section className="step">
        <div className="step-head">
          <span className="step-num">3</span>
          <h2>Fine-tune the test</h2>
        </div>
        <p className="step-sub">Optional — the defaults work.</p>
        <div className="step-body">
          <details className="disclosure">
            <summary>
              Test settings
              <span className="summary-note">
                {form.num_probes} probes · {suiteLabel}
              </span>
            </summary>
            <div className="disclosure-body">
              <div className="field-row">
                <div className="field">
                  <label htmlFor="suite">What to focus on</label>
                  <select id="suite" value={form.suite} onChange={set('suite')}>
                    {suites.map((s) => (
                      <option key={s.key} value={s.key}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="probes">Number of questions</label>
                  <input
                    id="probes"
                    type="number"
                    min="1"
                    max="25"
                    value={form.num_probes}
                    onChange={set('num_probes')}
                  />
                  <div className="hint">6 keeps a run under two minutes.</div>
                </div>
              </div>

              <PurposeField
                value={form.model_purpose}
                onChange={(text) => setForm((f) => ({ ...f, model_purpose: text }))}
              />
            </div>
          </details>
        </div>
      </section>

      {error && (
        <div className="notice" data-kind="error" style={{ marginBottom: 18 }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <button className="btn btn-lg" type="submit" disabled={!ready || submitting}>
          {submitting ? 'Starting…' : 'Start audit'}
        </button>
        <span style={{ color: 'var(--muted)', fontSize: 14 }}>
          {ready
            ? `${chosenModel} · ${form.num_probes} questions`
            : 'Choose a model to begin'}
        </span>
      </div>
    </form>
  )
}
