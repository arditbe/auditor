const DIMENSIONS = [
  ['accuracy', 'accuracy'],
  ['hallucination_resistance', 'hallucination'],
  ['instruction_following', 'instruction'],
  ['safety', 'safety'],
  ['coherence', 'coherence'],
]

const MAX = 5

export function toneFor(score /* 0-5 */) {
  if (score >= 4) return 'pass'
  if (score > 2) return 'warn'
  return 'fail'
}

/**
 * The instrument. Five channels, one per scoring dimension.
 *
 * A dimension no probe has tested reads as a dash, not a zero — the meter
 * refuses to show a measurement it has not taken.
 */
export function MeterBridge({ dimensions }) {
  const readings = dimensions ?? {}

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Dimension readings</span>
        <span className="eyebrow" style={{ marginLeft: 'auto' }}>
          of {MAX}
        </span>
      </div>
      <div className="panel-body">
        <div className="meters">
          {DIMENSIONS.map(([key, label]) => {
            const raw = readings[key]
            const tested = typeof raw === 'number'
            const value = tested ? raw : 0
            const tone = tested ? toneFor(value) : 'idle'

            return (
              <div className="meter" key={key} data-untested={!tested}>
                <div className="meter-top">
                  <span className="meter-name">{label}</span>
                  <span className={`meter-value tone-${tone}`}>
                    {tested ? value.toFixed(2) : 'not tested'}
                  </span>
                </div>
                <div
                  className="meter-track"
                  role="meter"
                  aria-label={`${label} score`}
                  aria-valuenow={tested ? value : undefined}
                  aria-valuemin={0}
                  aria-valuemax={MAX}
                  aria-valuetext={tested ? `${value} of ${MAX}` : 'not tested'}
                >
                  <div
                    className="meter-fill"
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
