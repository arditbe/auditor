import { useEffect, useState } from 'react'
import { api } from '../lib/api'

const CADENCES = [
  { key: 'hourly', label: 'Every hour' },
  { key: 'daily', label: 'Every day' },
  { key: 'weekly', label: 'Every week' },
]

function when(watch) {
  if (watch.cadence === 'hourly') return 'every hour'
  const h = String(watch.hour_utc).padStart(2, '0')
  return `${watch.cadence === 'weekly' ? 'weekly' : 'daily'} at ${h}:00 UTC`
}

function due(ms) {
  if (!ms) return '—'
  const mins = Math.round((ms - Date.now()) / 60000)
  if (mins <= 0) return 'due now'
  if (mins < 60) return `in ${mins} min`
  const hours = Math.round(mins / 60)
  return hours < 48 ? `in ${hours}h` : `in ${Math.round(hours / 24)}d`
}

/**
 * Standing instructions: audit this model on a schedule, unattended.
 *
 * This is what makes Auditor autonomous rather than something you drive. Once
 * a watch exists, nobody has to open the dashboard for auditing to keep
 * happening — and if the score drops, the watch says so.
 */
export function Watches({ validators, onClose }) {
  const [watches, setWatches] = useState([])
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const [adding, setAdding] = useState(false)

  const [form, setForm] = useState({
    name: '',
    target_model: '',
    validator_model: 'gemini-flash',
    suite: 'general',
    num_probes: 6,
    cadence: 'daily',
    hour_utc: 3,
    build_dataset: true,
    dataset_on_regression_only: true,
  })

  const load = () =>
    Promise.all([
      api.watches().catch(() => ({ watches: [] })),
      api.datasets().catch(() => ({ datasets: [] })),
    ]).then(([w, d]) => {
      setWatches(w.watches)
      setDatasets(d.datasets)
    })

  useEffect(() => {
    load().catch((e) => setError(e.message)).finally(() => setLoading(false))
  }, [])

  const act = async (label, fn) => {
    setBusy(label)
    setError(null)
    try {
      await fn()
      await load()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const create = (e) => {
    e.preventDefault()
    if (!form.target_model.trim()) {
      setError('A watch needs something to audit.')
      return
    }
    act('create', async () => {
      await api.createWatch(form)
      setAdding(false)
      setForm((f) => ({ ...f, name: '', target_model: '' }))
    })
  }

  const usable = validators.filter((v) => v.available)

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal modal-wide"
        role="dialog"
        aria-modal="true"
        aria-label="Scheduled audits"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>Scheduled audits</h2>
          <button className="btn btn-quiet btn-sm" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="modal-body">
          <p style={{ color: 'var(--muted)', fontSize: 14, marginBottom: 16 }}>
            A watch audits a model on a schedule with nobody present, and tells
            you when the score drops.
          </p>

          {loading ? (
            <div className="empty">Loading…</div>
          ) : watches.length === 0 && !adding ? (
            <div className="empty">
              <h3>No scheduled audits</h3>
              <p>Add one and Auditor keeps checking your model on its own.</p>
            </div>
          ) : (
            <div className="owned">
              {watches.map((w) => (
                <div className="owned-row" key={w.watch_id}>
                  <div style={{ minWidth: 0 }}>
                    <div className="owned-name">
                      {w.name}
                      {!w.enabled && (
                        <span className="pill" style={{ marginLeft: 8 }}>
                          paused
                        </span>
                      )}
                      {w.last_summary?.startsWith('Regression') && (
                        <span className="pill pill-fail" style={{ marginLeft: 8 }}>
                          regressed
                        </span>
                      )}
                    </div>
                    <div className="owned-base">
                      {when(w)} · {w.num_probes} probes ·{' '}
                      {w.enabled ? `next ${due(w.next_due_ms)}` : 'paused'}
                    </div>
                    {w.last_summary && (
                      <div className="owned-base" style={{ marginTop: 2 }}>
                        Last: {w.last_score ?? '—'}/100 — {w.last_summary}
                      </div>
                    )}
                  </div>
                  <div className="owned-actions">
                    <button
                      className="link"
                      disabled={Boolean(busy)}
                      onClick={() =>
                        act('run', () => api.runWatchNow(w.watch_id))
                      }
                    >
                      {busy === 'run' ? 'Running…' : 'Run now'}
                    </button>
                    <button
                      className="link"
                      disabled={Boolean(busy)}
                      onClick={() =>
                        act('toggle', () =>
                          api.updateWatch(w.watch_id, { enabled: !w.enabled }),
                        )
                      }
                    >
                      {w.enabled ? 'Pause' : 'Resume'}
                    </button>
                    <button
                      className="link link-danger"
                      disabled={Boolean(busy)}
                      onClick={() =>
                        act('delete', () => api.deleteWatch(w.watch_id))
                      }
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {adding ? (
            <form onSubmit={create} className="card" style={{ marginTop: 16, padding: 16 }}>
              <div className="field">
                <label htmlFor="w-target">Model to audit</label>
                <input
                  id="w-target"
                  type="text"
                  placeholder="https://my-model.example.com/v1  or  ollama:mistral"
                  value={form.target_model}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, target_model: e.target.value }))
                  }
                />
                <div className="hint">
                  Scheduled runs happen on the server, so this has to be
                  reachable from there — a URL, not a model on your laptop.
                </div>
              </div>

              <div className="field-row">
                <div className="field">
                  <label htmlFor="w-cadence">How often</label>
                  <select
                    id="w-cadence"
                    value={form.cadence}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, cadence: e.target.value }))
                    }
                  >
                    {CADENCES.map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="w-hour">At (UTC)</label>
                  <input
                    id="w-hour"
                    type="number"
                    min="0"
                    max="23"
                    value={form.hour_utc}
                    disabled={form.cadence === 'hourly'}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, hour_utc: Number(e.target.value) }))
                    }
                  />
                </div>
              </div>

              <div className="field-row">
                <div className="field">
                  <label htmlFor="w-judge">Judge</label>
                  <select
                    id="w-judge"
                    value={form.validator_model}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, validator_model: e.target.value }))
                    }
                  >
                    {usable.map((v) => (
                      <option key={v.key} value={v.key}>
                        {v.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="field">
                  <label htmlFor="w-probes">Questions</label>
                  <input
                    id="w-probes"
                    type="number"
                    min="1"
                    max="25"
                    value={form.num_probes}
                    onChange={(e) =>
                      setForm((f) => ({ ...f, num_probes: Number(e.target.value) }))
                    }
                  />
                </div>
              </div>

              <label className="check">
                <input
                  type="checkbox"
                  checked={form.build_dataset}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, build_dataset: e.target.checked }))
                  }
                />
                <span>
                  Build training data from the failures
                  <span className="hint" style={{ margin: 0 }}>
                    The judge writes the answer the model should have given.
                  </span>
                </span>
              </label>

              {form.build_dataset && (
                <label className="check">
                  <input
                    type="checkbox"
                    checked={form.dataset_on_regression_only}
                    onChange={(e) =>
                      setForm((f) => ({
                        ...f,
                        dataset_on_regression_only: e.target.checked,
                      }))
                    }
                  />
                  <span>
                    Only when the score drops
                    <span className="hint" style={{ margin: 0 }}>
                      A stable model needs no repair data.
                    </span>
                  </span>
                </label>
              )}

              <div className="modal-actions">
                <button className="btn btn-sm" type="submit" disabled={Boolean(busy)}>
                  {busy === 'create' ? 'Creating…' : 'Create watch'}
                </button>
                <button
                  type="button"
                  className="link"
                  onClick={() => setAdding(false)}
                >
                  Cancel
                </button>
              </div>
            </form>
          ) : (
            <button
              className="btn btn-sm"
              style={{ marginTop: 16 }}
              onClick={() => setAdding(true)}
            >
              Add a scheduled audit
            </button>
          )}

          {error && (
            <div className="notice" data-kind="error" style={{ marginTop: 14 }}>
              {error}
            </div>
          )}

          {datasets.length > 0 && (
            <>
              <div className="eyebrow" style={{ marginTop: 26 }}>
                Training data
              </div>
              <div className="owned">
                {datasets.slice(0, 8).map((d) => (
                  <div className="owned-row" key={d.name}>
                    <div style={{ minWidth: 0 }}>
                      <div className="owned-name">{d.name}</div>
                      <div className="owned-base">
                        {d.rows} corrected example{d.rows === 1 ? '' : 's'} ·{' '}
                        {(d.bytes / 1024).toFixed(1)} KB
                      </div>
                    </div>
                    <div className="owned-actions">
                      <a className="link" href={api.datasetUrl(d.name)} download>
                        Download
                      </a>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
