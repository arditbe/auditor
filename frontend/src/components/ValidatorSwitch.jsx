import { useState } from 'react'
import { api } from '../lib/api'

/**
 * Repoints a live audit at a different judge.
 *
 * Available mid-run on purpose: start free on local Gemma, and when a verdict
 * looks wrong, escalate the remaining questions to Gemini without losing the
 * ones already scored.
 */
export function ValidatorSwitch({ runId, current, validators, active }) {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState(null)

  const spec = validators.find((v) => v.key === current)

  const change = async (e) => {
    const next = e.target.value
    if (next === current) return
    setPending(true)
    setError(null)
    try {
      await api.switchValidator(runId, next)
    } catch (err) {
      setError(err.message)
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="card">
      <div className="card-head">
        <h3>Judge</h3>
        {spec?.cost === 'free' && (
          <span className="pill pill-free" style={{ marginLeft: 'auto' }}>
            free
          </span>
        )}
      </div>
      <div className="card-body">
        {active ? (
          <div className="field">
            <select value={current ?? ''} onChange={change} disabled={pending}>
              {validators.map((v) => (
                <option key={v.key} value={v.key} disabled={!v.available}>
                  {v.label}
                  {!v.available ? ' — unavailable' : ''}
                </option>
              ))}
            </select>
            <div className="hint">
              Switching applies to the questions not yet scored.
            </div>
          </div>
        ) : (
          <div style={{ fontSize: 14, fontWeight: 600 }}>
            {spec?.label ?? current ?? '—'}
          </div>
        )}

        {error && (
          <div className="notice" data-kind="error" style={{ marginTop: 10 }}>
            {error}
          </div>
        )}
      </div>
    </div>
  )
}
