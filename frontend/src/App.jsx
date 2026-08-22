import { useEffect, useRef, useState } from 'react'
import { api, isCloudDemo, isDesktop } from './lib/api'
import { useRunStream } from './hooks/useRunStream'
import { SetupPanel } from './components/SetupPanel'
import { Transcript } from './components/Transcript'
import { MeterBridge } from './components/MeterBridge'
import { ScorePanel } from './components/ScorePanel'
import { ValidatorSwitch } from './components/ValidatorSwitch'
import { Report } from './components/Report'
import { Settings } from './components/Settings'
import { ProbeStrip } from './components/ProbeStrip'

const STATUS = {
  idle: { text: 'Ready', tone: 'idle' },
  running: { text: 'Auditing', tone: 'primary' },
  complete: { text: 'Complete', tone: 'pass' },
  failed: { text: 'Failed', tone: 'fail' },
  cancelled: { text: 'Stopped', tone: 'warn' },
}

/* The run id lives in the URL hash, so an audit is linkable and survives a
 * refresh — the stream replays and rebuilds the whole transcript. */
function runIdFromHash() {
  const value = window.location.hash.replace(/^#\/?/, '').trim()
  return value.startsWith('run_') ? value : null
}

export default function App() {
  const [runId, setRunIdState] = useState(runIdFromHash)
  const [validators, setValidators] = useState([])
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const run = useRunStream(runId)
  const tailRef = useRef(null)

  const setRunId = (id) => {
    setRunIdState(id)
    window.history.pushState(null, '', id ? `#${id}` : window.location.pathname)
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
  }, [epoch])

  useEffect(() => {
    if (!isCloudDemo) return
    api.settings()
      .then((status) => {
        if (!status.google_api_key_set) setSettingsOpen(true)
      })
      .catch(() => {})
  }, [epoch])

  // The desktop shell's Settings menu item opens this window.
  useEffect(() => {
    if (!isDesktop) return undefined
    return window.auditorDesktop.onOpenSettings(() => setSettingsOpen(true))
  }, [])

  // Follow the newest question as it lands, but never fight a reader who has
  // scrolled up.
  useEffect(() => {
    if (run.status !== 'running' || !tailRef.current) return
    const nearBottom =
      window.innerHeight + window.scrollY >= document.body.offsetHeight - 300
    if (nearBottom) tailRef.current.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [run.activeProbeId, Object.keys(run.evaluations).length, run.status])

  const isActive = run.status === 'running'
  const status = STATUS[run.status] ?? { text: run.status, tone: 'idle' }
  const validatorLabel =
    validators.find((v) => v.key === run.validator)?.label ?? run.validator

  const stop = async () => {
    try {
      await api.cancelRun(runId)
    } catch {
      /* it finished on its own between render and click */
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">A</span>
          Auditor
        </div>

        {runId && (
          <span className={`pill pill-${status.tone}`}>
            {isActive && <span className="dot dot-live" />}
            {status.text}
            {run.score ? ` ${run.score.completed}/${run.score.total}` : ''}
          </span>
        )}

        <span className="spacer" />

        {runId && !isActive && (
          <button className="btn btn-sm" onClick={() => setRunId(null)}>
            New audit
          </button>
        )}
        {isActive && (
          <button className="btn btn-ghost btn-sm" onClick={stop}>
            Stop
          </button>
        )}
        <button className="btn btn-quiet btn-sm" onClick={() => setSettingsOpen(true)}>
          Settings
        </button>
      </header>

      {/* A thin live strip under the bar, so progress stays visible even when
          the score card is scrolled off. */}
      {isActive && run.probes.length > 0 && (
        <div style={{ padding: '0 24px', marginTop: -1 }}>
          <ProbeStrip
            probes={run.probes}
            evaluations={run.evaluations}
            activeProbeId={run.activeProbeId}
            total={run.score?.total}
          />
        </div>
      )}

      <main className={`page${runId ? '' : ' page-narrow'}`}>
        {!runId ? (
          <SetupPanel
            onStarted={setRunId}
            onOpenSettings={() => setSettingsOpen(true)}
            reloadKey={epoch}
          />
        ) : (
          <>
            <div className="run">
              <div ref={tailRef}>
                {run.status === 'failed' && (
                  <div className="notice" data-kind="error" style={{ marginBottom: 16 }}>
                    {run.error}
                  </div>
                )}
                {!run.connected && isActive && (
                  <div className="notice" data-kind="info" style={{ marginBottom: 16 }}>
                    Reconnecting to the live stream…
                  </div>
                )}
                <Transcript run={run} validatorLabel={validatorLabel} />
              </div>

              <aside className="rail">
                <ScorePanel score={run.score} status={run.status} run={run} />
                <MeterBridge dimensions={run.score?.dimensions} />
                <ValidatorSwitch
                  runId={runId}
                  current={run.validator}
                  validators={validators}
                  active={isActive}
                />
              </aside>
            </div>

            <Report report={run.report} run={run} cancelled={run.status === 'cancelled'} />
          </>
        )}
      </main>

      <Settings
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onChanged={() => setEpoch((n) => n + 1)}
      />
    </div>
  )
}
