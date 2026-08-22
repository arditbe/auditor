import { useRef, useState } from 'react'
import { api } from '../lib/api'

const ACCEPT = '.csv,.json,.jsonl,.md,.txt,.tsv,.yaml,.yml'

/**
 * What the model was fine-tuned to do.
 *
 * Two ways in, because people have this in two forms: some can describe it in
 * a sentence, others have the training data or a spec sitting in a file and
 * would rather not paraphrase it. The file is flattened server-side — a CSV
 * becomes row summaries, a long document is sampled from its beginning,
 * middle and end — so probe generation does not spend the whole context
 * window on one upload.
 */
export function PurposeField({ value, onChange }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const input = useRef(null)

  const pick = async (e) => {
    const chosen = e.target.files?.[0]
    e.target.value = ''
    if (!chosen) return

    setUploading(true)
    setError(null)
    try {
      const { context } = await api.uploadValidatorContext(chosen)
      setFile(context)
      onChange(context.text)
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const clear = () => {
    setFile(null)
    onChange('')
  }

  return (
    <div className="field">
      <label htmlFor="purpose">What was this model tuned for?</label>

      {file ? (
        <div className="context-file">
          <div>
            <div className="context-name">{file.filename}</div>
            <div className="context-meta">
              {file.chars.toLocaleString()} characters used
              {file.truncated && ' · sampled from a longer file'}
            </div>
          </div>
          <button type="button" className="link link-danger" onClick={clear}>
            Remove
          </button>
        </div>
      ) : (
        <textarea
          id="purpose"
          rows="3"
          placeholder="e.g. summarising discharge notes for clinicians"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      )}

      <div className="context-actions">
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => input.current?.click()}
          disabled={uploading}
        >
          {uploading ? 'Reading…' : file ? 'Use a different file' : 'Or upload a file'}
        </button>
        <span className="hint" style={{ margin: 0 }}>
          CSV, JSON, Markdown or text — your training data or a spec.
        </span>
        <input
          ref={input}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={pick}
        />
      </div>

      {error && (
        <div className="notice" data-kind="error" style={{ marginTop: 10 }}>
          {error}
        </div>
      )}

      <div className="hint">
        Steers the questions toward what the model claims to do. Leave it empty
        and Auditor probes broadly.
      </div>
    </div>
  )
}
