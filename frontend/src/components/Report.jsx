import { ProbeStrip } from './ProbeStrip'
import { gradeFor, toneFor } from './ScorePanel'

const DIMENSION_LABEL = {
  accuracy: 'Accuracy',
  hallucination_resistance: 'Makes things up',
  instruction_following: 'Follows instructions',
  safety: 'Safety',
  coherence: 'Coherence',
}

/** Plain-language summary, so the number is not the only thing on offer. */
function verdictLine(score, weakest) {
  const area = DIMENSION_LABEL[weakest]?.toLowerCase()
  if (score >= 85) return 'Strong across the board.'
  if (score >= 70) return area ? `Solid overall. Weakest on ${area}.` : 'Solid overall.'
  if (score >= 50) return area ? `Mixed. It struggles most with ${area}.` : 'Mixed results.'
  return area ? `Not reliable yet. It failed hardest on ${area}.` : 'Not reliable yet.'
}

export function Report({ report, run, cancelled }) {
  if (!report) return null

  const score = report.overall_score
  const tone = toneFor(score)

  return (
    <section style={{ marginTop: 26 }}>
      <div className="report-hero">
        <div className="eyebrow">
          {cancelled ? 'Partial report — stopped early' : 'Final report'}
        </div>
        <div style={{ margin: '12px 0 4px' }}>
          <span className={`score-value tone-${tone}`}>{score}</span>
          <span className="score-of">/100</span>
          <span className={`score-grade bg-${tone}`}>{gradeFor(score)}</span>
        </div>
        <p style={{ color: 'var(--muted)', fontSize: 15, marginTop: 6 }}>
          {verdictLine(score, report.weakest_dimension)}
        </p>
        {run && (
          <div style={{ maxWidth: 420, margin: '18px auto 0' }}>
            <ProbeStrip
              probes={run.probes}
              evaluations={run.evaluations}
              total={report.probes_run}
              large
            />
          </div>
        )}
      </div>

      <div className="stats">
        <div className="stat">
          <div className="stat-k">Questions</div>
          <div className="stat-v">{report.probes_run}</div>
        </div>
        <div className="stat">
          <div className="stat-k">Passed</div>
          <div className="stat-v tone-pass">{report.verdict_counts.pass}</div>
        </div>
        <div className="stat">
          <div className="stat-k">Failed</div>
          <div className="stat-v tone-fail">{report.verdict_counts.fail}</div>
        </div>
        <div className="stat">
          <div className="stat-k">Time</div>
          <div className="stat-v">{(report.duration_ms / 1000).toFixed(0)}s</div>
        </div>
        <div className="stat">
          <div className="stat-k">Judged by</div>
          <div className="stat-v small">{report.validators_used?.join(', ') || '—'}</div>
        </div>
      </div>

      {report.dimensions?.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-head">
            <h3>Where it stands</h3>
            <span className="eyebrow" style={{ marginLeft: 'auto' }}>
              weakest first
            </span>
          </div>
          <div className="card-body">
            <div className="dims">
              {report.dimensions.map((d) => {
                const t = toneFor(d.mean_score * 20)
                return (
                  <div className="dim" key={d.dimension}>
                    <div className="dim-top">
                      <span className="dim-name">
                        {DIMENSION_LABEL[d.dimension] ?? d.dimension}
                      </span>
                      <span className={`dim-score tone-${t}`}>
                        {d.mean_score.toFixed(2)} / 5
                        <span style={{ color: 'var(--faint)', fontWeight: 500 }}>
                          {' '}
                          · {d.probe_count}Q
                        </span>
                      </span>
                    </div>
                    <div className="dim-track">
                      <div
                        className="dim-fill"
                        style={{
                          transform: `scaleX(${d.mean_score / 5})`,
                          backgroundColor: `var(--${t})`,
                        }}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}

      {report.failures?.length > 0 && (
        <div className="card">
          <div className="card-head">
            <h3>What went wrong</h3>
            <span className="pill pill-fail" style={{ marginLeft: 'auto' }}>
              {report.failures.length} failure
              {report.failures.length === 1 ? '' : 's'}
            </span>
          </div>
          {report.failures.map((f) => (
            <div className="failure" key={f.probe_id}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <span className="pill">{DIMENSION_LABEL[f.dimension] ?? f.dimension}</span>
                {f.is_trap && <span className="pill pill-trap">trap</span>}
                {f.flags?.map((flag) => (
                  <span className="flag" key={flag}>
                    {flag}
                  </span>
                ))}
              </div>
              <p className="failure-q">{f.question}</p>
              <p className="failure-why">{f.reasoning}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
