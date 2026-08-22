import { useEffect, useReducer, useRef } from 'react'
import { api } from '../lib/api'

const EMPTY = {
  status: 'idle',
  probes: [],
  responses: {},
  evaluations: {},
  score: null,
  report: null,
  error: null,
  validator: null,
  activeProbeId: null,
  lastSeq: 0,
  connected: false,
}

/* One reducer over the event stream. Every event the backend publishes maps to
 * exactly one state transition, which keeps the UI a pure function of the
 * events rather than of fetch timing. */
function reduce(state, event) {
  const { type, data, seq } = event
  const next = { ...state, lastSeq: Math.max(state.lastSeq, seq ?? 0) }

  switch (type) {
    case 'run.started':
      return { ...next, status: 'running', validator: data.validator }

    case 'plan.ready':
      return { ...next, probes: data.probes }

    case 'probe.started':
      return {
        ...next,
        activeProbeId: data.probe.probe_id,
        // The plan usually arrives first, but tolerate a probe we have not seen.
        probes: next.probes.some((p) => p.probe_id === data.probe.probe_id)
          ? next.probes
          : [...next.probes, data.probe],
      }

    case 'probe.answered':
      return {
        ...next,
        responses: { ...next.responses, [data.probe_id]: data.response },
      }

    case 'probe.evaluated':
      return {
        ...next,
        evaluations: { ...next.evaluations, [data.probe_id]: data.evaluation },
        activeProbeId:
          next.activeProbeId === data.probe_id ? null : next.activeProbeId,
      }

    case 'score.updated':
      return { ...next, score: data }

    case 'validator.switched':
      return { ...next, validator: data.to }

    case 'run.completed':
      return { ...next, status: 'complete', report: data.report, activeProbeId: null }

    case 'run.cancelled':
      return { ...next, status: 'cancelled', report: data.report, activeProbeId: null }

    case 'run.failed':
      return { ...next, status: 'failed', error: data.error, activeProbeId: null }

    case 'connection':
      return { ...next, connected: data.connected }

    /* Seeded from the stored run document before the stream connects, so a
     * refresh after the server restarted still shows the full transcript.
     * Replayed events afterwards are idempotent — they key off probe_id and
     * re-set the same values. */
    case 'hydrate': {
      const run = data.run
      return {
        ...next,
        status: run.status === 'pending' ? 'running' : run.status,
        probes: run.probes ?? [],
        responses: run.responses ?? {},
        evaluations: run.evaluations ?? {},
        score: run.score ?? null,
        report: run.report ?? null,
        error: run.error ?? null,
        validator: run.config?.validator_model ?? null,
      }
    }

    case 'reset':
      return { ...EMPTY }

    default:
      return next
  }
}

const EVENT_TYPES = [
  'run.started',
  'plan.ready',
  'probe.started',
  'probe.answered',
  'probe.evaluated',
  'score.updated',
  'validator.switched',
  'run.completed',
  'run.cancelled',
  'run.failed',
]

/** Subscribes to a run's SSE stream and folds it into render-ready state. */
export function useRunStream(runId) {
  const [state, dispatch] = useReducer(reduce, EMPTY)
  const seqRef = useRef(0)

  // Keep the resume cursor outside React state so reconnects read the latest
  // value without re-running the effect on every event.
  seqRef.current = state.lastSeq

  useEffect(() => {
    if (!runId) {
      dispatch({ type: 'reset', data: {} })
      seqRef.current = 0
      return undefined
    }

    let source = null
    let retryTimer = null
    let closed = false

    const connect = () => {
      if (closed) return
      source = new EventSource(api.streamUrl(runId, seqRef.current))

      source.onopen = () =>
        dispatch({ type: 'connection', data: { connected: true } })

      for (const name of EVENT_TYPES) {
        source.addEventListener(name, (e) => {
          try {
            dispatch(JSON.parse(e.data))
          } catch {
            /* a malformed frame must not kill the stream */
          }
        })
      }

      source.onerror = () => {
        dispatch({ type: 'connection', data: { connected: false } })
        source.close()
        // The server closes the stream when a run ends; that surfaces here as
        // an error too. Retry once after a beat — if the run is over, the
        // replay log returns immediately and the stream closes again quietly.
        if (!closed) retryTimer = setTimeout(connect, 2000)
      }
    }

    dispatch({ type: 'reset', data: {} })
    seqRef.current = 0

    // Seed from the store, then stream. Connect regardless of whether the
    // fetch succeeds — a run that only exists in the live event log (not yet
    // flushed) must still stream.
    api
      .run(runId)
      .then((run) => {
        if (!closed) dispatch({ type: 'hydrate', data: { run } })
      })
      .catch(() => {})
      .finally(connect)

    return () => {
      closed = true
      if (retryTimer) clearTimeout(retryTimer)
      if (source) source.close()
    }
  }, [runId])

  return state
}
