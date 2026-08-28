from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from random import Random
from typing import Sequence

from backend.env.environment import RacingEnv

from .agents import RandomAgent
from .runner import EpisodeRecord, run_episodes, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the seeded random-policy racing baseline")
    parser.add_argument("--episodes", type=_positive_int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=_positive_int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def report(record: EpisodeRecord, total_steps: int, elapsed: float) -> None:
        if record.episode % args.report_every == 0 or record.episode == args.episodes:
            throughput = total_steps / elapsed if elapsed else 0.0
            print(
                f"episode {record.episode}/{args.episodes} | "
                f"steps {total_steps} | {throughput:.1f} steps/s"
            )

    records, wall_time = run_episodes(
        RacingEnv(),
        RandomAgent(Random(args.seed)),
        args.episodes,
        on_episode=report,
    )
    summary = summarize_run(records, seed=args.seed, wall_time=wall_time)
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
