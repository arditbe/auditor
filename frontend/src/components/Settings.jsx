import { useEffect, useState } from 'react'
import { api, isDesktop } from '../lib/api'

/**
 * Where a Google AI Studio key gets entered.
 *
 * In the desktop app the key is stored by the shell in the OS keychain and the
 * page never reads it back — it only ever learns whether one is set. In the
 * browser build there is nowhere safe to keep it, so the key lives only in the
 * running backend and is gone when it restarts.
 */
export function Settings({ open, onClose, onChanged }) {
  const [status, setStatus] = useState(null)
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (!open) return
    setError(null)
    setSaved(false)
    setValue('')
    api.settings().then(setStatus).catch((e) => setError(e.message))
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])

  if (!open) return null

  const save = async (key) => {
    setSaving(true)
    setError(null)
    try {
      if (isDesktop) {
        // The shell persists it and applies it to the backend in one step.
        const result = await window.auditorDesktop.setApiKey(key)
        if (result.error) throw new Error(result.error)
        if (key && !result.persisted && result.reason === 'no-encryption') {
          setError(
            'Saved for this session only — your system has no secure storage, ' +
              'so the key was not written to disk.',
          )
        }
      } else {
        await api.setGoogleApiKey(key)
      }
      setStatus(await api.settings())
      setValue('')
      setSaved(true)
      onChanged?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Settings"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="modal-head">
          <h2>Settings</h2>
          <button className="link-btn" onClick={onClose} aria-label="Close">
            Close
          </button>
        </div>

        <div className="modal-body">
          <div className="field">
            <label htmlFor="apikey">Google AI Studio API key</label>
            <input
              id="apikey"
              type="password"
              autoComplete="off"
              spellCheck="false"
              placeholder={
                status?.google_api_key_set
                  ? `Saved — ${status.google_api_key_hint}`
                  : 'Paste your key'
              }
              value={value}
              onChange={(e) => {
                setValue(e.target.value)
                setSaved(false)
              }}
              disabled={saving}
            />
            <div className="hint">
              Unlocks the Gemini validators. Free to create at{' '}
              <a
                href="https://aistudio.google.com/apikey"
                target="_blank"
                rel="noreferrer"
              >
                aistudio.google.com/apikey
              </a>
              . {isDesktop
                ? 'Stored encrypted by your operating system.'
                : 'Kept only in the running server, never written to disk.'}
            </div>
          </div>

          <div className="modal-actions">
            <button
              className="btn btn-small"
              onClick={() => save(value.trim())}
              disabled={!value.trim() || saving}
            >
              {saving ? 'Saving…' : 'Save key'}
            </button>
            {status?.google_api_key_set && (
              <button
                className="link-btn danger"
                onClick={() => save('')}
                disabled={saving}
              >
                Remove key
              </button>
            )}
          </div>

          {saved && !error && (
            <div className="notice" data-kind="info">
              Key applied. The Gemini validators are now selectable.
            </div>
          )}
          {error && (
            <div className="notice" data-kind="error">
              {error}
            </div>
          )}

          <hr className="rule" style={{ margin: '20px 0 16px' }} />

          <span className="eyebrow">This machine</span>
          <ul className="capability-list">
            <li data-ok={Boolean(status?.ollama_available)}>
              Ollama{' '}
              <span>
                {status?.ollama_available
                  ? 'installed — local models available'
                  : 'not found — install it from ollama.com for free local models'}
              </span>
            </li>
            <li data-ok={Boolean(status?.google_api_key_set || status?.vertex_configured)}>
              Google models{' '}
              <span>
                {status?.google_api_key_set
                  ? 'ready via API key'
                  : status?.vertex_configured
                    ? 'ready via Vertex AI'
                    : 'add a key above to enable'}
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
