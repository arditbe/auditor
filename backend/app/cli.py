"""Run an audit from the terminal.

The dashboard is the product, but a terminal runner earns its place in two
situations the browser cannot cover: a headless box such as a Colab runtime,
and CI, where the exit code needs to mean something.

    python -m app.cli --target ollama:qwen2:0.5b --validator gemini-flash-key

Exit codes: 0 the run finished, 1 the run failed, 2 the score was below
`--min-score`. That last one is what makes it usable as a quality gate.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import textwrap

from .agent.prompts import SUITES
from .config import settings
from .events import STREAM_END, EventType, bus
from .models.schemas import RunConfig
from .orchestrator import start_run
from .providers import VALIDATORS, inspect_source, list_validators


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------

class Style:
    """ANSI codes, disabled when the output is piped or NO_COLOR is set."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def dim(self, t): return self._wrap("2", t)
    def bold(self, t): return self._wrap("1", t)
    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def red(self, t): return self._wrap("31", t)
    def cyan(self, t): return self._wrap("36", t)

    def verdict(self, v: str) -> str:
        colour = {"pass": self.green, "warn": self.yellow, "fail": self.red}
        return colour.get(v, self.dim)(v.upper())


def _wrap_text(text: str, indent: str = "  ", width: int = 88) -> str:
    """Wrap a paragraph, marking only the first line.

    Continuation lines are indented to match but carry no marker -- repeating
    "Q" down the left of a wrapped question reads as several questions.
    """
    body = " ".join(text.split())
    return textwrap.fill(
        body,
        width=width,
        initial_indent=indent,
        subsequent_indent=" " * len(indent),
    )


def _bar(value: float, out_of: float = 5.0, width: int = 20) -> str:
    filled = int(round(value / out_of * width))
    return "█" * filled + "·" * (width - filled)


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

async def run_audit(args: argparse.Namespace) -> int:
    s = Style(
        enabled=sys.stdout.isatty()
        and not os.environ.get("NO_COLOR")
        and not args.no_color
    )

    config = RunConfig(
        target_model=args.target,
        validator_model=args.validator,
        suite=args.suite,
        num_probes=args.probes,
        model_purpose=args.purpose,
    )

    print()
    print(s.bold("Auditor"))
    print(s.dim(f"  target    {config.target_model}"))
    print(s.dim(f"  validator {VALIDATORS[config.validator_model].label}"))
    print(s.dim(f"  suite     {config.suite} · {config.num_probes} probes"))
    print()

    run = await start_run(config)
    queue, backlog, _ = await bus.subscribe(run.run_id)

    probes: dict[str, dict] = {}
    final_report: dict | None = None
    failed_error: str | None = None

    async def handle(event) -> bool:
        """Render one event. Returns False when the stream is done."""
        nonlocal final_report, failed_error
        kind, data = event.type, event.data

        if kind is EventType.LOG:
            print(s.dim(f"  {data['message']}"))

        elif kind is EventType.PLAN_READY:
            print(s.dim(f"  {data['count']} probes written\n"))

        elif kind is EventType.PROBE_STARTED:
            probe = data["probe"]
            probes[probe["probe_id"]] = probe
            tags = [probe["dimension"], probe["difficulty"]]
            if probe["is_trap"]:
                tags.append("TRAP")
            header = f"PROBE {probe['index'] + 1:02d}  {' · '.join(tags)}"
            print(s.cyan(header))
            print(_wrap_text(probe["question"], indent="  Q  "))

        elif kind is EventType.PROBE_ANSWERED:
            response = data["response"]
            if response.get("error"):
                print(_wrap_text(f"(no response) {response['error']}", "  A  "))
            else:
                text = response["text"] or "(empty)"
                if not args.full and len(text) > 300:
                    text = text[:300] + " …"
                print(_wrap_text(text, indent="  A  "))
            print(s.dim(f"     {response['latency_ms']} ms"))

        elif kind is EventType.PROBE_EVALUATED:
            ev = data["evaluation"]
            scores = "  ".join(
                f"{k} {v:.0f}/5" for k, v in ev["scores"].items()
            )
            print(f"  {s.verdict(ev['verdict'])}  {s.dim(scores)}")
            if ev.get("reasoning"):
                print(s.dim(_wrap_text(ev["reasoning"], indent="     ")))
            if ev.get("flags"):
                print(s.dim(f"     flags: {', '.join(ev['flags'])}"))
            print()

        elif kind is EventType.SCORE_UPDATED:
            print(
                s.dim(
                    f"  running score {data['overall']:.1f}/100 "
                    f"({data['completed']}/{data['total']})\n"
                )
            )

        elif kind is EventType.RUN_COMPLETED:
            final_report = data["report"]
            return False

        elif kind is EventType.RUN_CANCELLED:
            final_report = data["report"]
            return False

        elif kind is EventType.RUN_FAILED:
            failed_error = data["error"]
            return False

        return True

    for event in backlog:
        if not await handle(event):
            break
    else:
        while True:
            item = await queue.get()
            if item is STREAM_END:
                break
            if not await handle(item):
                break

    await bus.unsubscribe(run.run_id, queue)

    if failed_error:
        print(s.red(f"\nRun failed: {failed_error}\n"))
        return 1

    if not final_report:
        print(s.red("\nRun ended without a report.\n"))
        return 1

    _print_report(final_report, s)

    if args.min_score is not None and final_report["overall_score"] < args.min_score:
        print(
            s.red(
                f"Below the {args.min_score} threshold "
                f"({final_report['overall_score']}).\n"
            )
        )
        return 2
    return 0


def _print_report(report: dict, s: Style) -> None:
    score = report["overall_score"]
    tone = s.green if score >= 70 else s.yellow if score >= 50 else s.red

    print(s.bold("─" * 60))
    print(f"{s.bold('SCORE')}  {tone(f'{score}/100')}   grade {tone(report['grade'])}")
    counts = report["verdict_counts"]
    print(
        s.dim(
            f"  {counts['pass']} pass · {counts['warn']} warn · "
            f"{counts['fail']} fail   ·   {report['probes_run']} probes in "
            f"{report['duration_ms'] / 1000:.0f}s"
        )
    )
    print()

    if report.get("dimensions"):
        print(s.bold("BREAKDOWN") + s.dim("  (weakest first)"))
        for d in report["dimensions"]:
            mean = d["mean_score"]
            colour = s.green if mean >= 4 else s.yellow if mean > 2 else s.red
            name = d["dimension"]
            count = d["probe_count"]
            print(
                f"  {name:<26} {colour(_bar(mean))} "
                f"{colour(f'{mean:.2f}')}/5  {s.dim(f'n={count}')}"
            )
        print()

    if report.get("failures"):
        print(s.bold(f"FAILURES ({len(report['failures'])})"))
        for f in report["failures"]:
            print(_wrap_text(f["question"], indent="  · "))
            print(s.dim(_wrap_text(f["reasoning"], indent="    ")))
        print()

    if report.get("validators_used"):
        print(s.dim(f"judged by {', '.join(report['validators_used'])}"))
    print()


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auditor",
        description="Audit a language model and print the result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              auditor --target ollama:qwen2:0.5b
              auditor --target https://my-model.run.app/v1 --validator gemini-flash-key
              auditor --target ollama:mistral --probes 10 --min-score 70
            """
        ),
    )
    parser.add_argument(
        "--target", required=True,
        help="Model to audit: 'ollama:<tag>', a server URL, or 'prepared:<name>'.",
    )
    parser.add_argument(
        "--validator", default="local-gemma",
        help=f"Judge model. One of: {', '.join(VALIDATORS)}",
    )
    parser.add_argument("--probes", type=int, default=6, help="How many questions.")
    parser.add_argument(
        "--suite", default="general", choices=sorted(SUITES),
        help="What to focus the probes on.",
    )
    parser.add_argument(
        "--purpose", default="",
        help="What the model was tuned for. Steers the probe set.",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Exit 2 if the final score is below this. For use as a gate.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print model answers in full instead of truncating them.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colour.")
    parser.add_argument(
        "--list-validators", action="store_true",
        help="Show which judges are available and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]
    if "--list-validators" in argv:
        for v in list_validators():
            mark = "✓" if v["available"] else "✗"
            print(f"{mark} {v['key']:<20} {v['label']}")
            if not v["available"]:
                print(f"    {v['unavailable_reason']}")
        return 0

    args = parser.parse_args(argv)

    if args.validator not in VALIDATORS:
        parser.error(
            f"unknown validator {args.validator!r}. "
            f"Choose from: {', '.join(VALIDATORS)}"
        )

    spec = VALIDATORS[args.validator]
    if spec.requires == "api_key" and not settings.google_api_key:
        parser.error(
            f"{spec.label} needs a Google AI Studio key. "
            "Set GOOGLE_API_KEY, or get one at https://aistudio.google.com/apikey"
        )

    # A bare path or URL is a common mistake for --target; say so usefully.
    if not args.target.startswith(("ollama:", "prepared:", "http://", "https://")):
        detection = inspect_source(args.target)
        parser.error(
            f"--target must be 'ollama:<tag>', a URL, or 'prepared:<name>'. "
            f"{detection.detail}"
        )

    try:
        return asyncio.run(run_audit(args))
    except KeyboardInterrupt:
        print("\nStopped.\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
