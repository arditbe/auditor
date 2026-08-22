function resolveBase() {
  const injected = new URLSearchParams(window.location.search).get('apiBase')
  if (injected) return injected.replace(/\/$/, '')
  return import.meta.env.VITE_API_BASE ?? ''
}

const BASE = resolveBase()

export const isDesktop = Boolean(window.auditorDesktop?.isDesktop)

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    // FastAPI puts the useful message in `detail`; surface it verbatim so the
    // UI can show what actually went wrong rather than "request failed".
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* non-JSON error body; keep the status line */
    }
    throw new Error(detail)
  }
  return res.json()
}

async function upload(path, formData) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body: formData })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* keep status */
    }
    throw new Error(detail)
  }
  return res.json()
}

export const api = {
  health: () => request('/api/health'),
  settings: () => request('/api/settings'),
  setGoogleApiKey: (key) =>
    request('/api/settings/google-api-key', {
      method: 'PUT',
      body: JSON.stringify({ google_api_key: key }),
    }),
  targetModels: () => request('/api/models/target'),
  preparedModels: () => request('/api/models/prepared'),
  uploadModel: (files) => {
    const body = new FormData()
    for (const file of files) {
      // Preserve the folder structure the picker gave us; the server
      // sanitises every component before touching disk.
      body.append('files', file, file.webkitRelativePath || file.name)
    }
    // No Content-Type: the browser must set the multipart boundary itself.
    return request('/api/models/upload', { method: 'POST', body, headers: {} })
  },
  detectModel: (source) =>
    request('/api/models/detect', {
      method: 'POST',
      body: JSON.stringify({ source }),
    }),
  prepareModel: (source, name = '') =>
    request('/api/models/prepare', {
      method: 'POST',
      body: JSON.stringify({ source, name }),
    }),
  exportToOllama: (name) =>
    request(`/api/models/prepared/${encodeURIComponent(name)}/export-to-ollama`, {
      method: 'POST',
    }),
  deletePrepared: (name, deleteFiles = false) =>
    request(
      `/api/models/prepared/${encodeURIComponent(name)}?delete_files=${deleteFiles}`,
      { method: 'DELETE' },
    ),
  validators: () => request('/api/models/validator'),
  suites: () => request('/api/suites'),
  uploadValidatorContext: (file) => {
    const form = new FormData()
    form.append('file', file)
    return upload('/api/validator/context', form)
  },
  run: (runId) => request(`/api/runs/${runId}`),
  startRun: (config) =>
    request('/api/runs', { method: 'POST', body: JSON.stringify(config) }),
  switchValidator: (runId, validatorModel) =>
    request(`/api/runs/${runId}/validator`, {
      method: 'POST',
      body: JSON.stringify({ validator_model: validatorModel }),
    }),
  cancelRun: (runId) =>
    request(`/api/runs/${runId}/cancel`, { method: 'POST' }),
  streamUrl: (runId, since = 0) =>
    `${BASE}/api/runs/${runId}/stream?since=${since}`,
}
