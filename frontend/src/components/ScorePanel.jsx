function gradeFor(score) {
  if (score >= 90) return 'A'
  if (score >= 80) return 'B'
  if (score >= 70) return 'C'
  if (score >= 60) return 'D'
  return 'F'
}

function toneFor(score) {
  if (score >= 70) return 'pass'
  if (score >= 50) return 'warn'
  return 'fail'
}

/** Headline reading: weighted score out of 100, plus the verdict tally. */
export function ScorePanel({ score, status }) {
  const scored = score && score.completed > 0
  const value = scored ? score.overall : null
  const tone = scored ? toneFor(value) : 'idle'

  return (
    <div className="panel">
      <div className="score-block">
        <span className="eyebrow">Weighted score</span>
        <div className="score-readout" style={{ marginTop: 8 }}>
          <span className={`score-value tone-${tone}`}>
            {scored ? value.toFixed(1) : '—'}
          </span>
          {scored && (
            <span className={`score-grade tone-${tone}`}>{gradeFor(value)}</span>
          )}
        </div>
        <div className="score-sub">
          {score
            ? `${score.completed} of ${score.total} probes scored`
            : status === 'idle'
              ? 'No audit running'
              : 'Waiting for the first result'}
        </div>
      </div>

      {scored && (
        <div className="tally">
          <div>
            <span className="n tone-pass">{score.passes}</span>
            <span className="k">pass</span>
          </div>
          <div>
            <span className="n tone-warn">{score.warns}</span>
            <span className="k">warn</span>
          </div>
          <div>
            <span className="n tone-fail">{score.fails}</span>
            <span className="k">fail</span>
          </div>
        </div>
      )}
    </div>
  )
}
