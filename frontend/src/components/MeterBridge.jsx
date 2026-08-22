const DIMENSIONS = [
  ['accuracy', 'Accuracy'],
  ['hallucination_resistance', 'Makes things up'],
  ['instruction_following', 'Follows instructions'],
  ['safety', 'Safety'],
  ['coherence', 'Coherence'],
]

const MAX = 5

export function toneFor(score) {
  if (score >= 4) return 'pass'
  if (score > 2) return 'warn'
  return 'fail'
}

/**
 * Where the model is strong and where it is weak.
 *
 * A dimension no probe has tested reads "not tested", never 0 — reporting a
 * measurement you have not taken is the one thing a scorecard must not do.
 */
export function MeterBridge({ dimensions }) {
  const readings = dimensions ?? {}

  return (
    <div className="card">
      <div className="card-head">
        <h3>Breakdown</h3>
        <span className="eyebrow" style={{ marginLeft: 'auto' }}>
          of 5
        </span>
      </div>
      <div className="card-body">
        <div className="dims">
          {DIMENSIONS.map(([key, label]) => {
            const raw = readings[key]
            const tested = typeof raw === 'number'
            const value = tested ? raw : 0
            const tone = tested ? toneFor(value) : 'idle'

            return (
              <div className="dim" key={key} data-untested={!tested}>
                <div className="dim-top">
                  <span className="dim-name">{label}</span>
                  <span className={`dim-score tone-${tone}`}>
                    {tested ? value.toFixed(2) : 'not tested'}
                  </span>
                </div>
                <div
                  className="dim-track"
                  role="meter"
                  aria-label={label}
                  aria-valuenow={tested ? value : undefined}
                  aria-valuemin={0}
                  aria-valuemax={MAX}
                  aria-valuetext={tested ? `${value} of ${MAX}` : 'not tested'}
                >
                  <div
                    className="dim-fill"
                    style={{
                      transform: `scaleX(${tested ? value / MAX : 0})`,
                      backgroundColor: tested ? `var(--${tone})` : 'transparent',
                    }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
