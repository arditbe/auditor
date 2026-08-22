import { useRef, useState } from 'react'
import { api } from '../lib/api'

const KIND_LABEL = {
  mlx_adapters: 'LoRA adapters',
  peft_adapters: 'LoRA adapters',
  gguf_file: 'GGUF file',
  model_dir: 'Model folder',
  server_url: 'Model server',
  ollama_tag: 'Installed model',
}

/* Checkpoints are superseded by the final adapter, and the base model is named
 * in the config rather than shipped. Sending only what matters turns a 60 MB
 * folder into a 21 MB upload. */
const CHECKPOINT = /^\d+_adapters\.safetensors$/
const USEFUL = /\.(safetensors|json|bin|gguf|model|txt|jinja)$/i

function worthUploading(file) {
  if (file.name.startsWith('.')) return false
  if (CHECKPOINT.test(file.name)) return false
  return USEFUL.test(file.name)
}

function gb(bytes) {
  if (!bytes) return null
  return bytes >= 1e9
    ? `${(bytes / 1e9).toFixed(1)} GB`
    : `${Math.round(bytes / 1e6)} MB`
}

function UploadIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 15v2.5A2.5 2.5 0 0 0 6.5 20h11a2.5 2.5 0 0 0 2.5-2.5V15"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

/**
 * Step one: what are we auditing?
 *
 * Uploading is the headline because it works the same whether Auditor runs on
 * this laptop or on a server — the files come from the browser, not from a
 * filesystem the backend may not share. Models already on the machine are
 * offered underneath as one-click cards rather than a dropdown.
 */
export function ModelPicker({
  installed,
  owned,
  selected,
  onSelect,
  onAdded,
  onRemoveOwned,
  installError,
}) {
  const [detection, setDetection] = useState(null)
  const [busy, setBusy] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [showUrl, setShowUrl] = useState(false)
  const [url, setUrl] = useState('')

  const folderInput = useRef(null)
  const fileInput = useRef(null)

  const upload = async (fileList) => {
    const files = Array.from(fileList).filter(worthUploading)
    if (!files.length) {
      setError(
        'No model files in there. Pick the folder containing adapter_config.json.',
      )
      return
    }
    setBusy('uploading')
    setError(null)
    setDetection(null)
    try {
      setDetection(await api.uploadModel(files))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const checkUrl = async (e) => {
    e?.preventDefault?.()
    if (!url.trim()) return
    setBusy('checking')
    setError(null)
    setDetection(null)
    try {
      setDetection(await api.detectModel(url))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const confirm = async () => {
    setBusy('preparing')
    setError(null)
    try {
      const source = detection.resolved_path || url
      if (detection.readiness === 'ready' && detection.spec) {
        onAdded({
          spec: detection.spec,
          name: detection.suggested_name,
          kind: detection.kind,
        })
      } else {
        onAdded(await api.prepareModel(source, detection.suggested_name))
      }
      setDetection(null)
      setUrl('')
      setShowUrl(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const tone =
    detection?.readiness === 'blocked'
      ? 'bad'
      : detection?.readiness === 'ready'
        ? 'ok'
        : 'info'

  return (
    <>
      <div
        className="dropzone"
        data-dragging={dragging}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          if (e.dataTransfer.files?.length) upload(e.dataTransfer.files)
        }}
      >
        <div className="dropzone-icon">
          <UploadIcon />
        </div>
        <h3>Drop your model here</h3>
        <p className="dropzone-sub">
          The folder your training run saved, or a <code>.gguf</code> file
        </p>

        <div className="dropzone-actions">
          <button
            type="button"
            className="btn"
            onClick={() => folderInput.current?.click()}
            disabled={Boolean(busy)}
          >
            {busy === 'uploading' ? 'Uploading…' : 'Choose folder'}
          </button>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => fileInput.current?.click()}
            disabled={Boolean(busy)}
          >
            Choose file
          </button>
        </div>

        <p className="dropzone-note">
          Only the adapter and its config are sent — usually about 20 MB.
        </p>

        {/* webkitdirectory is how a browser offers a native folder picker. */}
        <input
          ref={folderInput}
          type="file"
          webkitdirectory=""
          directory=""
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) upload(e.target.files)
            e.target.value = ''
          }}
        />
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files?.length) upload(e.target.files)
            e.target.value = ''
          }}
        />
      </div>

      {detection && (
        <div className="detected" data-tone={tone}>
          <div className="detected-head">
            {KIND_LABEL[detection.kind] && (
              <span className="pill">{KIND_LABEL[detection.kind]}</span>
            )}
            <span className="detected-title">{detection.title}</span>
          </div>
          <p className="detected-detail">{detection.detail}</p>

          {detection.base_model && (
            <div className="detected-meta">
              Base model: <span className="mono">{detection.base_model}</span>
            </div>
          )}
          {detection.uploaded_files > 0 && (
            <div className="detected-meta">
              {detection.uploaded_files} file
              {detection.uploaded_files === 1 ? '' : 's'} ·{' '}
              {gb(detection.uploaded_bytes)}
            </div>
          )}
          {detection.warnings?.map((w) => (
            <div className="detected-meta" key={w}>
              {w}
            </div>
          ))}

          {detection.readiness !== 'blocked' && (
            <button
              className="btn btn-sm"
              style={{ marginTop: 12 }}
              onClick={confirm}
              disabled={Boolean(busy)}
            >
              {busy === 'preparing'
                ? 'Preparing…'
                : (detection.action_label ?? 'Use this model')}
            </button>
          )}
          {busy === 'preparing' && (
            <div className="detected-meta">
              This can take a minute for a large model. Keep this tab open.
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="notice" data-kind="error" style={{ marginTop: 14 }}>
          {error}
        </div>
      )}

      {/* Models the person already has, as one-click cards. */}
      {(owned.length > 0 || installed.length > 0) && (
        <div style={{ marginTop: 22 }}>
          <div className="eyebrow" style={{ marginBottom: 9 }}>
            Or pick one you already have
          </div>
          <div className="choices">
            {owned.map((m) => (
              /* Wrapped, because Remove cannot be a button inside a button. */
              <div className="choice-wrap" key={m.spec}>
                <button
                  type="button"
                  className="choice"
                  aria-pressed={selected === m.spec}
                  onClick={() => onSelect(m.spec)}
                >
                  {selected === m.spec && <span className="choice-check">✓</span>}
                  <span className="choice-name">
                    {m.name}
                    <span className="pill pill-primary">yours</span>
                  </span>
                  <span className="choice-meta">
                    {m.base_model ? `fine-tune of ${m.base_model}` : 'ready to audit'}
                  </span>
                </button>
                <button
                  type="button"
                  className="choice-remove"
                  aria-label={`Remove ${m.name}`}
                  title={`Remove ${m.name}`}
                  onClick={() => onRemoveOwned(m.name)}
                >
                  Remove
                </button>
              </div>
            ))}
            {installed.map((m) => (
              <button
                key={m.spec}
                type="button"
                className="choice"
                aria-pressed={selected === m.spec}
                onClick={() => onSelect(m.spec)}
              >
                {selected === m.spec && <span className="choice-check">✓</span>}
                <span className="choice-name">{m.name}</span>
                <span className="choice-meta">
                  {[m.parameter_size, gb(m.size_bytes)].filter(Boolean).join(' · ') ||
                    'Ollama'}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {installError && (
        <div className="notice" data-kind="warn" style={{ marginTop: 14 }}>
          {installError}
        </div>
      )}

      <div style={{ marginTop: 16 }}>
        {showUrl ? (
          /* Deliberately not a <form>: this sits inside the setup form, and a
             nested form is invalid HTML — the browser lets the inner submit
             escape to the outer one and reloads the page. Enter is wired up by
             hand instead. */
          <div className="field">
            <label htmlFor="url">Model server URL</label>
            <div className="inline-form">
              <input
                id="url"
                type="text"
                placeholder="https://my-model.example.com/v1"
                value={url}
                onChange={(e) => {
                  setUrl(e.target.value)
                  setDetection(null)
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    checkUrl(e)
                  }
                }}
                disabled={Boolean(busy)}
              />
              <button
                className="btn btn-ghost"
                type="button"
                onClick={checkUrl}
                disabled={!url.trim() || Boolean(busy)}
              >
                {busy === 'checking' ? 'Checking…' : 'Check'}
              </button>
            </div>
            <div className="hint">
              Anything that accepts OpenAI-style <code>/chat/completions</code>{' '}
              requests. A path on this machine works too.
            </div>
          </div>
        ) : (
          <button type="button" className="link" onClick={() => setShowUrl(true)}>
            Model already running somewhere? Use a URL →
          </button>
        )}
      </div>
    </>
  )
}
