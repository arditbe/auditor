import { useState } from 'react'

const DIMENSION_LABEL = {
  accuracy: 'accuracy',
  hallucination_resistance: 'hallucination resistance',
  instruction_following: 'instruction following',
  safety: 'safety',
  coherence: 'coherence',
}

/* A rambling model can emit thousands of words. Clamp it so one bad answer
 * cannot push the rest of the transcript off the screen mid-demo — the length
 * is still reported, and the full text is one click away. */
const CLAMP_CHARS = 520

function Answer({ text }) {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > CLAMP_CHARS

  if (!long) return <p className="said">{text}</p>

  return (
    <>
      <p className={`said${expanded ? '' : ' clamped'}`}>{text}</p>
      <button
        type="button"
        className="link-btn"
        onClick={() => setExpanded((v) => !v)}
      >
        {expanded
          ? 'Show less'
          : `Show all ${text.length.toLocaleString()} characters`}
      </button>
    </>
  )
}

function pad(n) {
  return String(n + 1).padStart(2, '0')
}

function ScoreList({ scores }) {
  const entries = Object.entries(scores ?? {})
  if (!entries.length) return null
  return (
    <span className="verdict-scores">
      {entries
        .map(([k, v]) => `${DIMENSION_LABEL[k] ?? k} ${Number(v).toFixed(0)}/5`)
        .join('  ·  ')}
    </span>
  )
}

function Verdict({ evaluation, validatorLabel }) {
  return (
    <div className="verdict" data-v={evaluation.verdict}>
      <div className="verdict-head">
        <span className="verdict-label">{evaluation.verdict}</span>
        <ScoreList scores={evaluation.scores} />
        {validatorLabel && (
          <span className="judged-by">judged by {validatorLabel}</span>
        )}
      </div>
      {evaluation.reasoning && (
        <p className="verdict-reason">{evaluation.reasoning}</p>
      )}
      {evaluation.flags?.length > 0 && (
        <div className="verdict-flags">
          {evaluation.flags.map((flag) => (
            <span className="flag" key={flag}>
              {flag}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function ProbeEntry({ probe, response, evaluation, isActive, validatorLabel }) {
  return (
    <article className="probe">
      <header className="probe-head">
        <span className="probe-index mono">PROBE {pad(probe.index)}</span>
        <span className="chip">{DIMENSION_LABEL[probe.dimension] ?? probe.dimension}</span>
        <span className="chip">{probe.difficulty}</span>
        {probe.is_trap && <span className="chip chip-trap">trap</span>}
      </header>

      <div className="speech question">
        <span className="who">Auditor asks</span>
        <p className="said">{probe.question}</p>
      </div>

      {response ? (
        <div className="speech answer">
          <span className="who">Model answers</span>
          {response.error ? (
            <p className="said tone-fail">No response — {response.error}</p>
          ) : response.text ? (
            <Answer text={response.text} />
          ) : (
            <p className="said tone-fail">(empty response)</p>
          )}
          <div className="meta">
            {response.latency_ms} ms
            {response.completion_tokens != null &&
              ` · ${response.completion_tokens} tokens out`}
          </div>
        </div>
      ) : (
        isActive && (
          <div className="speech answer">
            <span className="who">Model answers</span>
            <div className="awaiting">
              <span className="dot" style={{ animation: 'pulse 1.6s ease-in-out infinite' }} />
              waiting for the model…
            </div>
          </div>
        )
      )}

      {evaluation ? (
        <Verdict evaluation={evaluation} validatorLabel={validatorLabel} />
      ) : (
        response && (
          <div className="awaiting" style={{ paddingLeft: 0 }}>
            <span className="dot" style={{ animation: 'pulse 1.6s ease-in-out infinite' }} />
            scoring…
          </div>
        )
      )}
    </article>
  )
}

export function Transcript({ run, validatorLabel }) {
  const { probes, responses, evaluations, activeProbeId, status } = run

  if (!probes.length) {
    return (
      <div className="panel empty">
        <span className="eyebrow">Transcript</span>
        {status === 'running' ? (
          <p>The agent is designing a probe set for this model…</p>
        ) : (
          <p>Probes will appear here as the agent writes them.</p>
        )}
      </div>
    )
  }

  // Only render probes the run has actually reached, so the transcript reveals
  // itself rather than showing every future question up front.
  const reached = probes.filter(
    (p) => responses[p.probe_id] || evaluations[p.probe_id] || p.probe_id === activeProbeId,
  )
  const visible = reached.length ? reached : probes.slice(0, 1)

  return (
    <div className="transcript">
      {visible.map((probe) => (
        <ProbeEntry
          key={probe.probe_id}
          probe={probe}
          response={responses[probe.probe_id]}
          evaluation={evaluations[probe.probe_id]}
          isActive={probe.probe_id === activeProbeId}
          validatorLabel={validatorLabel}
        />
      ))}
    </div>
  )
}
