import { useState } from 'react'

const DIMENSION_LABEL = {
  accuracy: 'accuracy',
  hallucination_resistance: 'hallucination resistance',
  instruction_following: 'instruction following',
  safety: 'safety',
  coherence: 'coherence',
}

/* A rambling model can emit thousands of words. Clamping keeps one bad answer
 * from pushing the rest of the transcript off screen mid-demo; the full text
 * is one click away and the length is stated. */
const CLAMP = 460

function Answer({ text }) {
  const [open, setOpen] = useState(false)
  const long = text.length > CLAMP

  return (
    <>
      <div className={`qa-answer${long && !open ? ' clamped' : ''}`}>{text}</div>
      {long && (
        <button className="link" style={{ marginTop: 8 }} onClick={() => setOpen((v) => !v)}>
          {open ? 'Show less' : `Show all ${text.length.toLocaleString()} characters`}
        </button>
      )}
    </>
  )
}

function Verdict({ evaluation, validatorLabel }) {
  const scores = Object.entries(evaluation.scores ?? {})
  return (
    <div className="verdict" data-v={evaluation.verdict}>
      <div className="verdict-head">
        <span className="verdict-badge">{evaluation.verdict}</span>
        {scores.length > 0 && (
          <span className="verdict-scores">
            {scores
              .map(([k, v]) => `${DIMENSION_LABEL[k] ?? k} ${Number(v).toFixed(0)}/5`)
              .join('  ·  ')}
          </span>
        )}
        {validatorLabel && <span className="judged-by">judged by {validatorLabel}</span>}
      </div>
      {evaluation.reasoning && <p className="verdict-why">{evaluation.reasoning}</p>}
      {evaluation.flags?.length > 0 && (
        <div className="verdict-flags">
          {evaluation.flags.map((f) => (
            <span className="flag" key={f}>
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function Exchange({ probe, response, evaluation, isActive, validatorLabel }) {
  return (
    <article className="exchange">
      <header className="exchange-head">
        <span className="exchange-n">Q{String(probe.index + 1).padStart(2, '0')}</span>
        <span className="pill">{DIMENSION_LABEL[probe.dimension] ?? probe.dimension}</span>
        <span className="pill">{probe.difficulty}</span>
        {probe.is_trap && <span className="pill pill-trap">trap</span>}
      </header>

      <div className="qa">
        <div className="qa-label">Auditor asks</div>
        <p className="qa-question">{probe.question}</p>

        <div className="qa-label">The model answers</div>
        {response ? (
          <>
            {response.error ? (
              <div className="qa-answer tone-fail">No response — {response.error}</div>
            ) : response.text ? (
              <Answer text={response.text} />
            ) : (
              <div className="qa-answer tone-fail">(empty response)</div>
            )}
            <div className="qa-meta">
              {response.latency_ms} ms
              {response.completion_tokens != null &&
                ` · ${response.completion_tokens} tokens`}
            </div>
          </>
        ) : (
          isActive && (
            <div className="waiting">
              <span className="dot dot-live" />
              waiting for the model…
            </div>
          )
        )}
      </div>

      {evaluation ? (
        <Verdict evaluation={evaluation} validatorLabel={validatorLabel} />
      ) : (
        response && (
          <div className="qa" style={{ paddingTop: 0 }}>
            <div className="waiting">
              <span className="dot dot-live" />
              scoring…
            </div>
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
      <div className="card empty">
        <h3>{status === 'running' ? 'Writing the questions' : 'Nothing yet'}</h3>
        <p>
          {status === 'running'
            ? 'The judge is designing a test for this model. This takes a moment.'
            : 'Questions will appear here as the audit runs.'}
        </p>
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
        <Exchange
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
