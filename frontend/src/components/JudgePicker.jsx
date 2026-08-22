/**
 * Step two: who does the scoring?
 *
 * Cards rather than a dropdown, because the choice has a real cost attached —
 * free and offline versus burning credits — and that should be visible before
 * you pick, not buried in a hint below a select.
 */
/* Card text has to stay short or the grid goes ragged. The full explanation
 * lives in the tooltip and in Settings. */
const SHORT_REASON = {
  gcp: 'Cloud-project version — API keys use the Gemini cards above',
  api_key: 'Needs an API key — add one in Settings',
}

const GROUPS = [
  {
    key: 'ready',
    title: 'Ready to use',
    match: (v) => v.available,
  },
  {
    key: 'key',
    title: 'Gemini with API key',
    match: (v) => !v.available && v.requires === 'api_key',
  },
  {
    key: 'cloud',
    title: 'Vertex AI (Google Cloud project)',
    match: (v) => !v.available && v.requires === 'gcp',
  },
]

export function JudgePicker({ validators, selected, onSelect, onNeedKey }) {
  if (!validators.length) {
    return <div className="empty">Loading judges…</div>
  }

  const grouped = GROUPS.map((group) => ({
    ...group,
    validators: validators.filter(group.match),
  })).filter((group) => group.validators.length > 0)

  return (
    <>
      {grouped.map((group) => (
        <div className="judge-group" key={group.key}>
          <div className="judge-group-title">{group.title}</div>
          <div className="choices">
            {group.validators.map((v) => (
              <button
                key={v.key}
                type="button"
                className="choice"
                aria-pressed={selected === v.key}
                disabled={!v.available}
                onClick={() => onSelect(v.key)}
                title={v.unavailable_reason ?? v.blurb}
              >
                {selected === v.key && <span className="choice-check">✓</span>}
                <span className="choice-name">
                  {v.label}
                  {v.cost === 'free' && <span className="pill pill-free">free</span>}
                </span>
                <span className="choice-meta">
                  {v.available ? v.blurb : (SHORT_REASON[v.requires] ?? v.unavailable_reason)}
                </span>
              </button>
            ))}
          </div>
        </div>
      ))}

      {validators.some((v) => !v.available && v.requires === 'api_key') && (
        <div className="notice" data-kind="info" style={{ marginTop: 12 }}>
          <span>
            Paste a Google AI Studio key to unlock the API-key Gemini judges.{' '}
            <button type="button" className="link" onClick={onNeedKey}>
              Add one now
            </button>
          </span>
        </div>
      )}
    </>
  )
}
