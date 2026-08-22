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
  const name = file.name
  if (name.startsWith('.')) return false
  if (CHECKPOINT.test(name)) return false
  return USEFUL.test(name)
}

function mb(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`
}

/**
 * Getting a model into Auditor.
 *
 * Uploading is the primary path because it works the same whether Auditor is
 * running on this laptop or on Cloud Run. Typing a path only ever works in the
 * former case, so it is offered as a secondary option.
 */
export function AddModel({ onAdded }) {
  const [detection, setDetection] = useState(null)
  const [busy, setBusy] = useState(null) // 'uploading' | 'checking' | 'preparing'
  const [error, setError] = useState(null)
  const [source, setSource] = useState('')
  const [usePath, setUsePath] = useState(false)
  const [dragging, setDragging] = useState(false)

  const folderInput = useRef(null)
  const fileInput = useRef(null)

  const reset = () => {
    setDetection(null)
    setError(null)
  }

  const upload = async (fileList) => {
    const files = Array.from(fileList).filter(worthUploading)
    if (!files.length) {
      setError(
        'No model files in there. Pick the folder containing adapter_config.json.',
      )
      return
    }
    setBusy('uploading')
    reset()
    try {
      const result = await api.uploadModel(files)
      setSource(result.resolved_path)
      setDetection(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const checkPath = async (e) => {
    e.preventDefault()
    if (!source.trim()) return
    setBusy('checking')
    reset()
    try {
      const result = await api.detectModel(source)
      setDetection(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const prepare = async () => {
    setBusy('preparing')
    setError(null)
    try {
      const result = await api.prepareModel(source, detection.suggested_name)
      onAdded(result)
      setSource('')
      setDetection(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(null)
    }
  }

  const useDirectly = () => {
    onAdded({
      spec: detection.spec,
      name: detection.suggested_name,
      kind: detection.kind,
    })
    setSource('')
    setDetection(null)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    if (e.dataTransfer.files?.length) upload(e.dataTransfer.files)
  }

  const tone =
    detection?.readiness === 'blocked'
      ? 'fail'
      : detection?.readiness === 'ready'
        ? 'pass'
        : 'live'

  return (
    <div className="add-model">
      <div
        className="dropzone"
        data-dragging={dragging}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <p className="dropzone-lede">
          Drop the folder your training saved, or
        </p>

        <div className="dropzone-actions">
          <button
            type="button"
            className="btn btn-small"
            onClick={() => folderInput.current?.click()}
            disabled={busy}
          >
            {busy === 'uploading' ? 'Uploading…' : 'Choose folder'}
          </button>
          <button
            type="button"
            className="btn btn-quiet btn-small"
            onClick={() => fileInput.current?.click()}
            disabled={busy}
          >
            Choose file
          </button>
        </div>

        <p className="dropzone-hint">
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

      {usePath ? (
        <form onSubmit={checkPath} className="path-form">
          <label htmlFor="source">Or a path on this machine, or a server URL</label>
          <div className="add-model-row">
            <input
              id="source"
              type="text"
              value={source}
              placeholder="/Users/you/my-finetune/adapters"
              onChange={(e) => {
                setSource(e.target.value)
                setDetection(null)
              }}
              disabled={busy}
            />
            <button
              className="btn btn-quiet btn-small"
              type="submit"
              disabled={!source.trim() || busy}
            >
              {busy === 'checking' ? 'Checking…' : 'Check'}
            </button>
          </div>
          <div className="hint">
            Paths only work when Auditor runs on the same machine as the model.
          </div>
        </form>
      ) : (
        <button
          type="button"
          className="link-btn"
          style={{ marginLeft: 0 }}
          onClick={() => setUsePath(true)}
        >
          Use a path or a server URL instead
        </button>
      )}

      {detection && (
        <div className="detection" data-tone={tone}>
          <div className="detection-head">
            {KIND_LABEL[detection.kind] && (
              <span className="chip">{KIND_LABEL[detection.kind]}</span>
            )}
            <strong className="detection-title">{detection.title}</strong>
          </div>

          <p className="detection-detail">{detection.detail}</p>

          {detection.base_model && (
            <div className="detection-meta">
              Base model: <span className="mono">{detection.base_model}</span>
            </div>
          )}

          {detection.uploaded_files > 0 && (
            <div className="detection-meta">
              Uploaded {detection.uploaded_files} file
              {detection.uploaded_files === 1 ? '' : 's'} ·{' '}
              {mb(detection.uploaded_bytes)}
            </div>
          )}

          {detection.warnings?.map((w) => (
            <div className="detection-meta warn" key={w}>
              {w}
            </div>
          ))}

          {detection.readiness === 'needs_prepare' && (
            <button
              className="btn btn-small"
              onClick={prepare}
              disabled={busy}
              style={{ marginTop: 12 }}
            >
              {busy === 'preparing' ? 'Preparing…' : detection.action_label}
            </button>
          )}

          {detection.readiness === 'ready' && (
            <button
              className="btn btn-small"
              onClick={useDirectly}
              style={{ marginTop: 12 }}
            >
              Use this model
            </button>
          )}

          {busy === 'preparing' && (
            <div className="detection-meta">
              This can take a minute for a large model. Keep this tab open.
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="notice" data-kind="error" style={{ marginTop: 12 }}>
          {error}
        </div>
      )}
    </div>
  )
}
