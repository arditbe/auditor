import { useEffect, useRef, useState } from 'react'
import { api, isDesktop } from './lib/api'
import { useRunStream } from './hooks/useRunStream'
import { SetupPanel } from './components/SetupPanel'
import { Transcript } from './components/Transcript'
import { MeterBridge } from './components/MeterBridge'
import { ScorePanel } from './components/ScorePanel'
import { ValidatorSwitch } from './components/ValidatorSwitch'
import { Report } from './components/Report'
import { Settings } from './components/Settings'

const STATUS_TEXT = {
  idle: 'Ready',
  running: 'Auditing',
  complete: 'Complete',
  failed: 'Failed',
  cancelled: 'Stopped',
}

/* The run id lives in the URL hash, so a run is linkable and survives a
 * refresh mid-audit — the stream replays from seq 0 and rebuilds the whole
 * transcript. */
function runIdFromHash() {
  const value = window.location.hash.replace(/^#\/?/, '').trim()
  return value.startsWith('run_') ? value : null
}

export default function App() {
  const [runId, setRunIdState] = useState(runIdFromHash)
  const [validators, setValidators] = useState([])
  const [health, setHealth] = useState(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [validatorEpoch, setValidatorEpoch] = useState(0)
  const run = useRunStream(runId)
  const transcriptRef = useRef(null)

  const setRunId = (id) => {
    setRunIdState(id)
    const target = id ? `#${id}` : ' '
    if (window.location.hash !== `#${id ?? ''}`) {
      window.history.pushState(null, '', id ? target : window.location.pathname)
    }
  }

  useEffect(() => {
    const onNav = () => setRunIdState(runIdFromHash())
    window.addEventListener('popstate', onNav)
    window.addEventListener('hashchange', onNav)
    return () => {
      window.removeEventListener('popstate', onNav)
      window.removeEventListener('hashchange', onNav)
    }
  }, [])

  useEffect(() => {
    api.validators().then((v) => setValidators(v.validators)).catch(() => {})
    api.health().then(setHealth).catch(() => {})
  }, [validatorEpoch])

  // The desktop shell's Settings menu item opens this window.
  useEffect(() => {
    if (!isDesktop) return undefined
    return window.auditorDesktop.onOpenSettings(() => setSettingsOpen(true))
  }, [])

  // Follow the newest probe as it lands, but never fight a user who has
  // scrolled up to read something.
  useEffect(() => {
    const el = transcriptRef.current
    if (!el || run.status !== 'running') return
    const nearBottom =
      window.innerHeight + window.scrollY >= document.body.offsetHeight - 260
    if (nearBottom) {
      el.scrollIntoView({ block: 'end', behavior: 'smooth' })
    }
  }, [run.activeProbeId, Object.keys(run.evaluations).length, run.status])

  const isActive = run.status === 'running'
  const validatorLabel =
    validators.find((v) => v.key === run.validator)?.label ?? run.validator

  const stop = async () => {
    try {
      await api.cancelRun(runId)
    } catch {
      /* the run finished on its own between render and click */
    }
  }

  return (
    <div className="shell">
      <header className="masthead">
        <h1>Auditor</h1>
        <span className="tagline">real-time model validation</span>
        <span className="spacer" />
        {runId && (
          <span className="status-pill" data-state={run.status}>
            <span className="dot" />
            {STATUS_TEXT[run.status] ?? run.status}
            {run.score ? ` ${run.score.completed}/${run.score.total}` : ''}
          </span>
        )}
        {health && (
          <span className="status-pill" title="Where run history is persisted">
            store: {health.store}
          </span>
        )}
        <button
          className="status-pill as-button"
          onClick={() => setSettingsOpen(true)}
        >
          Settings
        </button>
      </header>

      <Settings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onChanged={() => setValidatorEpoch((n) => n + 1)}
      />

      {!runId ? (
        <div className="workspace">
          <SetupPanel onStarted={setRunId} />
          <aside className="instrument">
            <ScorePanel score={null} status="idle" />
            <MeterBridge dimensions={null} />
          </aside>
        </div>
      ) : (
        <>
          <div className="workspace">
            <main ref={transcriptRef}>
              {run.status === 'failed' && (
                <div className="notice" data-kind="error" style={{ marginBottom: 18 }}>
                  {run.error}
                </div>
              )}
              {!run.connected && isActive && (
                <div className="notice" data-kind="info" style={{ marginBottom: 18 }}>
                  Reconnecting to the live stream…
                </div>
              )}
              <Transcript run={run} validatorLabel={validatorLabel} />
            </main>

            <aside className="instrument">
              <ScorePanel score={run.score} status={run.status} />
              <MeterBridge dimensions={run.score?.dimensions} />
              <ValidatorSwitch
                runId={runId}
                current={run.validator}
                validators={validators}
                active={isActive}
              />
              <div style={{ display: 'flex', gap: 8 }}>
                {isActive ? (
                  <button className="btn btn-quiet btn-small" onClick={stop}>
                    Stop audit
                  </button>
                ) : (
                  <button
                    className="btn btn-small"
                    onClick={() => setRunId(null)}
                  >
                    New audit
                  </button>
                )}
              </div>
            </aside>
          </div>

          <Report report={run.report} cancelled={run.status === 'cancelled'} />
        </>
      )}
    </div>
  )
}
