import { useState } from 'react'
import { api } from '../lib/api'

/**
 * Models Auditor prepared for this person, with the option to hand one to
 * Ollama permanently.
 *
 * Exporting is offered, never pushed: a prepared model audits exactly the
 * same. The reason to export is that Ollama keeps it after Auditor exits.
 */
export function PreparedModels({ models, onChanged, canExport }) {
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const [confirming, setConfirming] = useState(null)

  if (!models.length) return null

  const exportModel = async (name) => {
    setBusy(name)
    setError(null)
    setConfirming(null)
    try {
      await api.exportToOllama(name)
      onChanged()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const remove = async (name) => {
    setBusy(name)
    setError(null)
    try {
      await api.deletePrepared(name, true)
      onChanged()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="prepared">
      <span className="eyebrow">Your models</span>
      {models.map((m) => (
        <div className="prepared-row" key={m.spec}>
          <div className="prepared-name">
            <strong>{m.name}</strong>
            {m.base_model && (
              <span className="prepared-base mono">
                fine-tune of {m.base_model}
              </span>
            )}
          </div>

          {confirming === m.name ? (
            <div className="prepared-confirm">
              <p>
                This converts the model to GGUF so Ollama keeps it for good.
                It needs about 30 GB of free space while it runs and takes a
                few minutes. Auditing works without it.
              </p>
              <div className="prepared-actions">
                <button
                  className="btn btn-small"
                  onClick={() => exportModel(m.name)}
                >
                  Convert
                </button>
                <button
                  className="btn btn-quiet btn-small"
                  onClick={() => setConfirming(null)}
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="prepared-actions">
              {canExport && (
                <button
                  className="link-btn"
                  disabled={busy === m.name}
                  onClick={() => setConfirming(m.name)}
                >
                  {busy === m.name ? 'Converting…' : 'Keep in Ollama'}
                </button>
              )}
              <button
                className="link-btn danger"
                disabled={busy === m.name}
                onClick={() => remove(m.name)}
              >
                Remove
              </button>
            </div>
          )}
        </div>
      ))}

      {error && (
        <div className="notice" data-kind="error" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}
    </div>
  )
}
