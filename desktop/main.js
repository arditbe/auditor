'use strict'

/**
 * Auditor desktop shell.
 *
 * Electron owns the window and the secrets; the audit itself runs in the same
 * Python backend the web version uses, spawned here as a child process on a
 * private port. Keeping one backend means the desktop app cannot drift from
 * the deployed one.
 */

const { app, BrowserWindow, dialog, ipcMain, Menu, safeStorage, shell } = require('electron')
const { spawn } = require('node:child_process')
const fs = require('node:fs')
const http = require('node:http')
const net = require('node:net')
const path = require('node:path')

const isDev = !app.isPackaged
const KEY_FILE = () => path.join(app.getPath('userData'), 'google-api-key.bin')

let mainWindow = null
let backend = null
let backendPort = 0
let backendExit = null // set when the backend dies unexpectedly

// --------------------------------------------------------------------------
// secrets
// --------------------------------------------------------------------------

/* Stored with the OS keychain when it will have us (Keychain on macOS, DPAPI
 * on Windows, libsecret on Linux). If encryption is unavailable we do not
 * silently write the key in the clear -- we decline to persist it. */
function saveApiKey(key) {
  const file = KEY_FILE()
  if (!key) {
    fs.rmSync(file, { force: true })
    return { persisted: false, cleared: true }
  }
  if (!safeStorage.isEncryptionAvailable()) {
    return { persisted: false, reason: 'no-encryption' }
  }
  fs.writeFileSync(file, safeStorage.encryptString(key), { mode: 0o600 })
  return { persisted: true }
}

function loadApiKey() {
  try {
    const buf = fs.readFileSync(KEY_FILE())
    if (!safeStorage.isEncryptionAvailable()) return ''
    return safeStorage.decryptString(buf)
  } catch {
    return ''
  }
}

// --------------------------------------------------------------------------
// backend process
// --------------------------------------------------------------------------

function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.unref()
    server.on('error', reject)
    // Port 0 asks the OS for any free port, which avoids racing another copy
    // of the app or an unrelated service on a hardcoded one.
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address()
      server.close(() => resolve(port))
    })
  })
}

/** Where the bundled interpreter and backend source live. */
function resolvePaths() {
  if (isDev) {
    const root = path.join(__dirname, '..')
    return {
      python: path.join(root, 'backend', '.venv', 'bin', 'python'),
      cwd: path.join(root, 'backend'),
    }
  }
  const resources = process.resourcesPath
  const exe = process.platform === 'win32'
    ? path.join('python', 'python.exe')
    : path.join('python', 'bin', 'python3')
  return {
    python: path.join(resources, 'backend', exe),
    cwd: path.join(resources, 'backend'),
  }
}

function waitForHealth(port, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs
  return new Promise((resolve, reject) => {
    const attempt = () => {
      if (backendExit) {
        reject(new Error(`The backend stopped before it was ready.\n\n${backendExit}`))
        return
      }
      if (Date.now() > deadline) {
        reject(new Error('The backend did not start within two minutes.'))
        return
      }
      const req = http.get(
        { host: '127.0.0.1', port, path: '/api/health', timeout: 2000 },
        (res) => {
          res.resume()
          if (res.statusCode === 200) resolve()
          else setTimeout(attempt, 400)
        },
      )
      req.on('error', () => setTimeout(attempt, 400))
      req.on('timeout', () => {
        req.destroy()
        setTimeout(attempt, 400)
      })
    }
    attempt()
  })
}

async function startBackend() {
  const { python, cwd } = resolvePaths()
  if (!fs.existsSync(python)) {
    throw new Error(
      `Auditor's Python runtime is missing.\n\nExpected it at:\n${python}\n\n` +
        (isDev ? 'Run: cd backend && python3.12 -m venv .venv' : 'Reinstall the app.'),
    )
  }

  backendPort = await freePort()
  backendExit = null

  const apiKey = loadApiKey()
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONDONTWRITEBYTECODE: '1',
    STORE_BACKEND: 'memory',
  }
  if (apiKey) env.GOOGLE_API_KEY = apiKey

  backend = spawn(
    python,
    ['-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', String(backendPort)],
    { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] },
  )

  // Keep the last of stderr so a crash can be explained rather than guessed at.
  const tail = []
  const keep = (chunk) => {
    tail.push(chunk.toString())
    if (tail.length > 40) tail.shift()
    if (isDev) process.stdout.write(chunk)
  }
  backend.stdout.on('data', keep)
  backend.stderr.on('data', keep)

  backend.on('exit', (code, signal) => {
    if (code === 0 || signal === 'SIGTERM') return
    backendExit = tail.join('').slice(-2000)
    if (mainWindow && !mainWindow.isDestroyed()) {
      dialog.showErrorBox(
        'Auditor stopped unexpectedly',
        `The backend exited with code ${code}.\n\n${backendExit}`,
      )
    }
  })

  await waitForHealth(backendPort)
  return backendPort
}

function stopBackend() {
  if (!backend || backend.exitCode !== null) return
  backend.kill('SIGTERM')
  // The backend shuts down its own model servers on SIGTERM; give it a moment
  // before insisting, so a 7B model is not orphaned holding memory.
  setTimeout(() => {
    if (backend && backend.exitCode === null) backend.kill('SIGKILL')
  }, 5000)
}

// --------------------------------------------------------------------------
// window
// --------------------------------------------------------------------------

function createWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1360,
    height: 940,
    minWidth: 900,
    minHeight: 620,
    show: false,
    backgroundColor: '#0E1416',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  mainWindow.once('ready-to-show', () => mainWindow.show())

  // Anything that is not the app itself opens in the real browser.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  const uiRoot = isDev
    ? path.join(__dirname, '..', 'frontend', 'dist')
    : path.join(process.resourcesPath, 'ui')
  const indexFile = path.join(uiRoot, 'index.html')

  if (!fs.existsSync(indexFile)) {
    dialog.showErrorBox(
      'Auditor is not built',
      `The dashboard files are missing.\n\nExpected: ${indexFile}\n\n` +
        'Run: cd frontend && npm run build',
    )
    app.quit()
    return
  }

  // The port changes per launch, so the API base is injected rather than baked
  // into the bundle the way the web build does it.
  mainWindow.loadFile(indexFile, { query: { apiBase: `http://127.0.0.1:${port}` } })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

function buildMenu() {
  const template = [
    ...(process.platform === 'darwin'
      ? [{
          label: app.name,
          submenu: [
            { role: 'about' },
            { type: 'separator' },
            { label: 'Settings…', accelerator: 'Cmd+,', click: openSettings },
            { type: 'separator' },
            { role: 'hide' },
            { role: 'hideOthers' },
            { type: 'separator' },
            { role: 'quit' },
          ],
        }]
      : []),
    {
      label: 'File',
      submenu: [
        ...(process.platform === 'darwin'
          ? []
          : [{ label: 'Settings…', accelerator: 'Ctrl+,', click: openSettings },
             { type: 'separator' }]),
        process.platform === 'darwin' ? { role: 'close' } : { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      role: 'help',
      submenu: [
        {
          label: 'Get a Google AI Studio key',
          click: () => shell.openExternal('https://aistudio.google.com/apikey'),
        },
        {
          label: 'Install Ollama',
          click: () => shell.openExternal('https://ollama.com/download'),
        },
      ],
    },
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

function openSettings() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('auditor:open-settings')
  }
}

// --------------------------------------------------------------------------
// IPC
// --------------------------------------------------------------------------

ipcMain.handle('auditor:get-api-key', () => {
  const key = loadApiKey()
  return { set: Boolean(key), canPersist: safeStorage.isEncryptionAvailable() }
})

ipcMain.handle('auditor:set-api-key', async (_event, key) => {
  const trimmed = (key || '').trim()
  const result = saveApiKey(trimmed)

  // Apply to the running backend so the change takes effect without a restart.
  try {
    await putJson(backendPort, '/api/settings/google-api-key', {
      google_api_key: trimmed,
    })
  } catch (err) {
    return { ...result, applied: false, error: String(err.message || err) }
  }
  return { ...result, applied: true }
})

ipcMain.handle('auditor:api-base', () => `http://127.0.0.1:${backendPort}`)

function putJson(port, pathname, body) {
  return new Promise((resolve, reject) => {
    const payload = JSON.stringify(body)
    const req = http.request(
      {
        host: '127.0.0.1',
        port,
        path: pathname,
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
        },
      },
      (res) => {
        const chunks = []
        res.on('data', (c) => chunks.push(c))
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString()
          if (res.statusCode >= 200 && res.statusCode < 300) resolve(text)
          else reject(new Error(text || `HTTP ${res.statusCode}`))
        })
      },
    )
    req.on('error', reject)
    req.write(payload)
    req.end()
  })
}

// --------------------------------------------------------------------------
// lifecycle
// --------------------------------------------------------------------------

// Two copies would spawn two backends and two model servers, which on a 7B
// model means double the memory for no benefit.
if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })

  app.whenReady().then(async () => {
    buildMenu()
    try {
      const port = await startBackend()
      createWindow(port)
    } catch (err) {
      dialog.showErrorBox('Auditor could not start', String(err.message || err))
      app.quit()
      return
    }

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0 && backendPort) {
        createWindow(backendPort)
      }
    })
  })
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', stopBackend)
app.on('will-quit', stopBackend)
