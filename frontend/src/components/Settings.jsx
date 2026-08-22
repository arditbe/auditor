import { useEffect, useState } from 'react'
import { api, isDesktop } from '../lib/api'

function Cap({ ok, name, detail }) {
  return (
    <li>
      <span className={`caps-icon ${ok ? 'bg-pass' : 'bg-idle'}`}>{ok ? '✓' : '–'}</span>
      <span>
        <strong>{name}</strong> <span>{detail}</span>
      </span>
    </li>
  )
}

/**
 * Where a Google AI Studio key gets entered.
 *
 * In the desktop app the shell stores the key in the OS keychain and applies it
 * to the running backend. In the browser build the key lives only in the
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
    if (!open) return undefined
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
          <button className="btn btn-quiet btn-sm" onClick={onClose}>
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
              Unlocks the Gemini judges. Free at{' '}
              <a href="https://aistudio.google.com/apikey" target="_blank" rel="noreferrer">
                aistudio.google.com/apikey
              </a>
              . No Google Cloud project needed. {isDesktop
                ? 'Stored encrypted by your operating system.'
                : 'Kept only in the running server, never written to disk.'}
            </div>
          </div>

          <div className="modal-actions">
            <button
              className="btn btn-sm"
              onClick={() => save(value.trim())}
              disabled={!value.trim() || saving}
            >
              {saving ? 'Saving…' : 'Save key'}
            </button>
            {status?.google_api_key_set && (
              <button className="link link-danger" onClick={() => save('')} disabled={saving}>
                Remove key
              </button>
            )}
          </div>

          {saved && !error && (
            <div className="notice" data-kind="ok" style={{ marginTop: 14 }}>
              Key applied. Choose the Gemini cards labeled “API key”; Vertex AI
              still needs a Google Cloud project.
            </div>
          )}
          {error && (
            <div className="notice" data-kind="error" style={{ marginTop: 14 }}>
              {error}
            </div>
          )}

          <div className="eyebrow" style={{ marginTop: 24 }}>
            This machine
          </div>
          <ul className="caps">
            <Cap
              ok={Boolean(status?.ollama_available)}
              name="Ollama"
              detail={
                status?.ollama_available
                  ? 'installed — local models available'
                  : 'not found — install from ollama.com for free local models'
              }
            />
            <Cap
              ok={Boolean(status?.mlx_available)}
              name="Adapter fusing"
              detail={
                status?.mlx_available
                  ? 'available — you can audit LoRA fine-tunes'
                  : 'unavailable — needs an Apple Silicon Mac'
              }
            />
            <Cap
              ok={Boolean(status?.google_api_key_set || status?.vertex_configured)}
              name="Google judges"
              detail={
                status?.google_api_key_set
                  ? 'ready via API key'
                  : status?.vertex_configured
                    ? 'ready via Vertex AI'
                    : 'add a key above to enable'
              }
            />
          </ul>
        </div>
      </div>
    </div>
  )
}
