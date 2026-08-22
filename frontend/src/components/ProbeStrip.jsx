/**
 * One segment per probe, filling with its verdict colour as the audit runs.
 *
 * The progress bar and the result are the same object. "Four passed, one
 * failed, three still to go" is readable at a glance, with no legend and no
 * numbers to parse — which is what you want on a screen someone is watching
 * rather than reading.
 */
export function ProbeStrip({ probes, evaluations, activeProbeId, total, large }) {
  // Before the plan arrives we still know how many probes were asked for, so
  // the strip shows the right number of empty slots from the very start.
  const count = Math.max(probes.length, total || 0, 1)

  const segments = Array.from({ length: count }, (_, i) => {
    const probe = probes[i]
    if (!probe) return { key: `pending-${i}`, state: null }
    const verdict = evaluations[probe.probe_id]?.verdict
    if (verdict) return { key: probe.probe_id, state: verdict }
    if (probe.probe_id === activeProbeId) return { key: probe.probe_id, state: 'active' }
    return { key: probe.probe_id, state: null }
  })

  const done = segments.filter((s) => s.state && s.state !== 'active').length

  return (
    <div
      className={`strip${large ? ' strip-lg' : ''}`}
      role="progressbar"
      aria-valuenow={done}
      aria-valuemin={0}
      aria-valuemax={count}
      aria-label={`${done} of ${count} probes scored`}
    >
      {segments.map((s) => (
        <div key={s.key} className="strip-seg" data-v={s.state ?? undefined} />
      ))}
    </div>
  )
}
