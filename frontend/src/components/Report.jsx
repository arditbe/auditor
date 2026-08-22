const DIMENSION_LABEL = {
  accuracy: 'accuracy',
  hallucination_resistance: 'hallucination resistance',
  instruction_following: 'instruction following',
  safety: 'safety',
  coherence: 'coherence',
}

function seconds(ms) {
  if (ms == null) return '—'
  return `${(ms / 1000).toFixed(1)}s`
}

export function Report({ report, cancelled }) {
  if (!report) return null

  return (
    <section className="report">
      <div className="panel">
        <div className="panel-head">
          <span className="eyebrow">
            {cancelled ? 'Partial report — audit stopped early' : 'Final report'}
          </span>
          <span className="eyebrow" style={{ marginLeft: 'auto' }}>
            {report.probes_run} probes · {seconds(report.duration_ms)}
          </span>
        </div>

        <div className="report-grid">
          <div>
            <span className="k">Overall</span>
            <span className="v">
              {report.overall_score} <small style={{ opacity: 0.5 }}>/100</small>
            </span>
          </div>
          <div>
            <span className="k">Grade</span>
            <span className="v">{report.grade}</span>
          </div>
          <div>
            <span className="k">Weakest area</span>
            <span className="v" style={{ fontSize: 13 }}>
              {DIMENSION_LABEL[report.weakest_dimension] ?? '—'}
            </span>
          </div>
          <div>
            <span className="k">Mean latency</span>
            <span className="v">
              {report.target_latency_ms?.mean != null
                ? `${report.target_latency_ms.mean} ms`
                : '—'}
            </span>
          </div>
          <div>
            <span className="k">Judged by</span>
            <span className="v" style={{ fontSize: 13 }}>
              {report.validators_used?.join(', ') || '—'}
            </span>
          </div>
        </div>
      </div>

      {report.dimensions?.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <div className="panel-head">
            <span className="eyebrow">Breakdown — weakest first</span>
          </div>
          {report.dimensions.map((d) => (
            <div className="failure" key={d.dimension}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  gap: 12,
                  fontFamily: 'var(--mono)',
                  fontSize: 12,
                }}
              >
                <span>{DIMENSION_LABEL[d.dimension] ?? d.dimension}</span>
                <span style={{ color: 'var(--slate)' }}>
                  {d.mean_score.toFixed(2)} / {d.out_of} · {d.probe_count} probe
                  {d.probe_count === 1 ? '' : 's'} · weight {d.weight}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {report.failures?.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <div className="panel-head">
            <span className="eyebrow">
              Failures — {report.failures.length}
            </span>
          </div>
          {report.failures.map((f) => (
            <div className="failure" key={f.probe_id}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span className="chip">{DIMENSION_LABEL[f.dimension] ?? f.dimension}</span>
                {f.is_trap && <span className="chip chip-trap">trap</span>}
                {f.flags?.map((flag) => (
                  <span className="flag" key={flag}>
                    {flag}
                  </span>
                ))}
              </div>
              <p className="q">{f.question}</p>
              <p className="r">{f.reasoning}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  )
}
