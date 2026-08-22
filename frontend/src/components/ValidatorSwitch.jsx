import { useState } from 'react'
import { api } from '../lib/api'

/**
 * Repoints a live run at a different validator.
 *
 * Available mid-run on purpose: the pitch is that you can start free on local
 * Gemma and escalate to Gemini when a verdict looks wrong, without losing the
 * probes already scored.
 */
export function ValidatorSwitch({ runId, current, validators, active }) {
  const [pending, setPending] = useState(false)
  const [error, setError] = useState(null)

  const currentSpec = validators.find((v) => v.key === current)

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
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Validator</span>
      </div>
      <div className="panel-body">
        <div className="switcher">
          <div className="current">
            {currentSpec?.label ?? current ?? '—'}
            {currentSpec && (
              <span className="cost" data-cost={currentSpec.cost}>
                {currentSpec.cost}
              </span>
            )}
          </div>

          {active && (
            <>
              <select value={current ?? ''} onChange={change} disabled={pending}>
                {validators.map((v) => (
                  <option key={v.key} value={v.key} disabled={!v.available}>
                    {v.label}
                    {!v.available ? ' (needs GOOGLE_CLOUD_PROJECT)' : ''}
                  </option>
                ))}
              </select>
              <div className="hint mono" style={{ fontSize: 11, color: 'var(--slate)' }}>
                Switching applies to the probes that have not been scored yet.
              </div>
            </>
          )}

          {error && (
            <div className="notice" data-kind="error">
              {error}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
