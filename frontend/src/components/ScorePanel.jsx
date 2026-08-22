import { ProbeStrip } from './ProbeStrip'

export function gradeFor(score) {
  if (score >= 90) return 'A'
  if (score >= 80) return 'B'
  if (score >= 70) return 'C'
  if (score >= 60) return 'D'
  return 'F'
}

export function toneFor(score) {
  if (score >= 70) return 'pass'
  if (score >= 50) return 'warn'
  return 'fail'
}

/** The headline reading: one number, one letter, and the strip underneath. */
export function ScorePanel({ score, status, run }) {
  const scored = score && score.completed > 0
  const value = scored ? score.overall : null
  const tone = scored ? toneFor(value) : 'idle'

  const caption =
    status === 'idle'
      ? 'No audit running'
      : score
        ? `${score.completed} of ${score.total} questions scored`
        : 'Writing the questions…'

  return (
    <div className="card">
      <div className="scorecard">
        <div className="eyebrow">Score</div>
        <div style={{ margin: '10px 0 2px' }}>
          <span className={`score-value tone-${tone}`}>
            {scored ? value.toFixed(1) : '—'}
          </span>
          {scored && <span className="score-of">/100</span>}
          {scored && (
            <span className={`score-grade bg-${tone}`}>{gradeFor(value)}</span>
          )}
        </div>
        <div className="score-caption">{caption}</div>

        {run && (
          <div style={{ marginTop: 14 }}>
            <ProbeStrip
              probes={run.probes}
              evaluations={run.evaluations}
              activeProbeId={run.activeProbeId}
              total={score?.total}
            />
          </div>
        )}
      </div>

      {scored && (
        <div className="tally">
          <div className="tally-item">
            <span className="tally-n tone-pass">{score.passes}</span>
            <span className="tally-k">passed</span>
          </div>
          <div className="tally-item">
            <span className="tally-n tone-warn">{score.warns}</span>
            <span className="tally-k">warned</span>
          </div>
          <div className="tally-item">
            <span className="tally-n tone-fail">{score.fails}</span>
            <span className="tally-k">failed</span>
          </div>
        </div>
      )}
    </div>
  )
}
