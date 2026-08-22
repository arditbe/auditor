import { useEffect, useState } from 'react'
import { api } from '../lib/api'
import { AddModel } from './AddModel'
import { PreparedModels } from './PreparedModels'

function formatSize(bytes) {
  if (!bytes) return null
  return `${(bytes / 1e9).toFixed(1)} GB`
}

export function SetupPanel({ onStarted, disabled }) {
  const [targets, setTargets] = useState([])
  const [targetError, setTargetError] = useState(null)
  const [validators, setValidators] = useState([])
  const [suites, setSuites] = useState([])
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [uploadingContext, setUploadingContext] = useState(false)
  const [contextFile, setContextFile] = useState(null)
  const [error, setError] = useState(null)

  const [form, setForm] = useState({
    target_model: '',
    validator_model: 'local-gemma',
    suite: 'general',
    num_probes: 6,
    model_purpose: '',
  })

  const [prepared, setPrepared] = useState([])
  const [canExport, setCanExport] = useState(false)
  const [addingModel, setAddingModel] = useState(false)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      api.targetModels(),
      api.validators(),
      api.suites(),
      api.preparedModels().catch(() => ({ models: [] })),
    ])
      .then(([t, v, s, p]) => {
        if (cancelled) return
        const local = t.models.filter((m) => m.is_local)
        setTargets(local)
        setTargetError(t.error)
        setValidators(v.validators)
        setSuites(s.suites)
        setPrepared(p.models)
        setCanExport(Boolean(p.can_export_to_ollama))
        // Prefer a model the user prepared themselves -- if they went to the
        // trouble of adding one, that is what they came to audit.
        const preferred = p.models[0]?.spec ?? local[0]?.spec
        if (preferred) {
          setForm((f) => ({ ...f, target_model: preferred }))
        } else {
          // Nothing to choose from. An empty dropdown is a dead end, so open
          // the one control that can actually get them unstuck.
          setAddingModel(true)
        }
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
  }, [])

  const refreshModels = () => {
    Promise.all([
      api.preparedModels().catch(() => ({ models: [] })),
      api.targetModels().catch(() => ({ models: [] })),
    ]).then(([p, t]) => {
      setPrepared(p.models)
      setCanExport(Boolean(p.can_export_to_ollama))
      setTargets(t.models.filter((m) => m.is_local))
    })
  }

  /* A newly added model becomes the selection immediately -- the person just
   * told us this is the one they care about. */
  const onModelAdded = (model) => {
    setPrepared((list) =>
      list.some((m) => m.spec === model.spec) ? list : [model, ...list],
    )
    setForm((f) => ({ ...f, target_model: model.spec }))
    setAddingModel(false)
    api
      .targetModels()
      .then((t) => setTargets(t.models.filter((m) => m.is_local)))
      .catch(() => {})
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
    } finally {
      setSubmitting(false)
    }
  }

  const uploadContext = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploadingContext(true)
    setError(null)
    try {
      const result = await api.uploadValidatorContext(file)
      setContextFile(result.context)
      setForm((f) => ({ ...f, model_purpose: result.context.text }))
    } catch (err) {
      setError(err.message)
    } finally {
      setUploadingContext(false)
      e.target.value = ''
    }
  }

  const chosenValidator = validators.find((v) => v.key === form.validator_model)
  const ready = form.target_model && !loading && !disabled

  return (
    <form className="panel setup" onSubmit={submit}>
      <h2>Audit a model</h2>
      <p className="lede">
        Auditor writes its own test questions, puts them to your model, and has a
        second model score every answer against the criteria it wrote. Nothing is
        pre-scripted.
      </p>

      {targetError && (
        <div className="notice" data-kind="error" style={{ marginBottom: 18 }}>
          {targetError}
        </div>
      )}

      <div className="field">
        <label htmlFor="target">Model to audit</label>
        <select
          id="target"
          value={form.target_model}
          onChange={set('target_model')}
          disabled={loading || (!targets.length && !prepared.length)}
        >
          {!targets.length && !prepared.length && (
            <option value="">No models yet — add yours below</option>
          )}
          {prepared.length > 0 && (
            <optgroup label="Your models">
              {prepared.map((m) => (
                <option key={m.spec} value={m.spec}>
                  {m.name}
                  {m.base_model ? ` · fine-tune of ${m.base_model}` : ''}
                </option>
              ))}
            </optgroup>
          )}
          {targets.length > 0 && (
            <optgroup label="Installed with Ollama">
              {targets.map((m) => (
                <option key={m.spec} value={m.spec}>
                  {m.name}
                  {m.parameter_size ? ` · ${m.parameter_size}` : ''}
                  {formatSize(m.size_bytes) ? ` · ${formatSize(m.size_bytes)}` : ''}
                </option>
              ))}
            </optgroup>
          )}
        </select>

        <PreparedModels
          models={prepared}
          onChanged={refreshModels}
          canExport={canExport}
        />

        {addingModel ? (
          <AddModel onAdded={onModelAdded} />
        ) : (
          <button
            type="button"
            className="link-btn"
            style={{ marginLeft: 0 }}
            onClick={() => setAddingModel(true)}
          >
            Trained your own model? Add it →
          </button>
        )}

        <div className="hint" style={{ display: addingModel ? 'none' : undefined }}>
          Everything here runs on this machine.
        </div>
      </div>

      <div className="field">
        <label htmlFor="validator">Validator</label>
        <select
          id="validator"
          value={form.validator_model}
          onChange={set('validator_model')}
          disabled={loading}
        >
          {validators.map((v) => (
            <option key={v.key} value={v.key} disabled={!v.available}>
              {v.label}
              {v.cost === 'free' ? ' — free' : ''}
              {!v.available ? ' (needs GOOGLE_CLOUD_PROJECT)' : ''}
            </option>
          ))}
        </select>
        <div className="hint">{chosenValidator?.blurb}</div>
      </div>

      <div className="field-row">
        <div className="field">
          <label htmlFor="suite">Suite</label>
          <select id="suite" value={form.suite} onChange={set('suite')} disabled={loading}>
            {suites.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="probes">Probes</label>
          <input
            id="probes"
            type="number"
            min="1"
            max="25"
            value={form.num_probes}
            onChange={set('num_probes')}
          />
          <div className="hint">6 keeps a live demo under two minutes.</div>
        </div>
      </div>

      <div className="field">
        <label htmlFor="purpose">What was this model tuned for? (optional)</label>
        <textarea
          id="purpose"
          rows="5"
          placeholder="Upload a CSV/text file, or type a short description."
          value={form.model_purpose}
          onChange={set('model_purpose')}
        />
        <div className="button-row context-upload-row">
          <label className="btn btn-quiet btn-small file-button">
            {uploadingContext ? 'Reading file...' : 'Upload context file'}
            <input type="file" onChange={uploadContext} />
          </label>
          {contextFile && (
            <span className="context-file">
              {contextFile.filename} · {contextFile.chars} chars
              {contextFile.truncated ? ' · sampled' : ''}
            </span>
          )}
        </div>
        <div className="hint">
          The validator uses this file text to design probes instead of relying on a typed description.
        </div>
      </div>

      {error && (
        <div className="notice" data-kind="error" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}

      <button className="btn" type="submit" disabled={!ready || submitting}>
        {submitting ? 'Starting…' : 'Start audit'}
      </button>
    </form>
  )
}
