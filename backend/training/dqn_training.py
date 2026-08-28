from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from random import Random
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Sequence

import torch

from backend.env.environment import RacingEnv
from backend.rl.dqn import DQNAgent, DQNConfig, DQNPolicy

from .curriculum import AdaptiveCurriculum
from .evaluator import EvaluationRecord, evaluate_policy_with_replay
from .replay import EvaluationReplay
from .runner import EpisodeRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the continuous-observation DQN racer")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--max-transitions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/dqn-latest.json"))
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume:
        state = torch.load(args.resume.with_suffix(".pt"), map_location="cpu", weights_only=False)
        agent = DQNAgent.from_snapshot(state["agent"])
        curriculum = AdaptiveCurriculum.from_snapshot(state["curriculum"])
        records = list(state["records"])
        evaluations = list(state["evaluations"])
        replays = list(state["replays"])
        start_episode = int(state["episode"]) + 1
        wall_time = float(state["wall_time"])
    else:
        agent = DQNAgent(Random(args.seed), DQNConfig(), seed=args.seed)
        curriculum = AdaptiveCurriculum(Random(args.seed + 2_000_000))
        records: list[EpisodeRecord] = []
        evaluations: list[EvaluationRecord] = []
        replays: list[EvaluationReplay] = []
        start_episode = 1
        wall_time = 0.0

    environment = RacingEnv()
    started = perf_counter()
    for episode in range(start_episode, args.episodes + 1):
        observation = curriculum.reset(environment)
        while True:
            action = agent.choose_action(observation)
            result = environment.step(action)
            agent.observe(observation, action, result.reward, result.observation, result.done)
            observation = result.observation
            if result.done:
                record = _record(result.info, episode)
                records.append(record)
                curriculum.observe(record.lap_completed)
                break
        if episode % args.evaluate_every == 0 or agent.transitions >= args.max_transitions:
            evaluation, replay = evaluate_policy_with_replay(
                environment, DQNPolicy(agent), 1, training_episode=episode
            )
            evaluations.append(evaluation)
            replays.append(replay)
            elapsed = wall_time + perf_counter() - started
            _save(args.checkpoint, agent, curriculum, records, evaluations, replays, episode, elapsed, args.seed)
            print(
                f"evaluation @{episode} | progress {evaluation.mean_progress:.1%} | "
                f"laps {evaluation.lap_completions} | epsilon {agent.epsilon:.4f}"
            )
            if evaluation.lap_completions:
                print(f"canonical lap completed in {replay.simulated_duration:.2f}s")
        if agent.transitions >= args.max_transitions:
            break
    return 0


def _record(info: dict[str, object], episode: int) -> EpisodeRecord:
    reason = str(info["termination_reason"])
    elapsed = float(info["elapsed_time"])
    return EpisodeRecord(
        episode, int(info["steps"]), elapsed, float(info["episode_return"]),
        float(info["furthest_progress"]), reason, reason == "lap", elapsed if reason == "lap" else None,
    )


def _save(
    path: Path,
    agent: DQNAgent,
    curriculum: AdaptiveCurriculum,
    records: list[EpisodeRecord],
    evaluations: list[EvaluationRecord],
    replays: list[EvaluationReplay],
    episode: int,
    wall_time: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = path.with_suffix(".pt")
    state = {
        "agent": agent.snapshot(), "curriculum": curriculum.snapshot(), "records": records,
        "evaluations": evaluations, "replays": replays, "episode": episode, "wall_time": wall_time,
    }
    with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary_weights = Path(handle.name)
    torch.save(state, temporary_weights)
    os.replace(temporary_weights, weights)
    best_replay = max(
        replays,
        key=lambda value: (
            value.lap_completed,
            value.furthest_progress,
            value.total_return,
            -value.simulated_duration,
        ),
    )
    best_weights = path.with_name(f"{path.stem}-best.pt")
    if best_replay is replays[-1]:
        with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary_best = Path(handle.name)
        torch.save(state, temporary_best)
        os.replace(temporary_best, best_weights)
    payload = {
        "schema_version": 2,
        "algorithm": "dqn",
        "run": {"seed": seed, "completed_episode": episode, "training_wall_time": wall_time},
        "agent": {"config": asdict(agent.config), "epsilon": agent.epsilon, "transitions": agent.transitions},
        "curriculum": {"floor": curriculum.floor},
        "model_file": weights.name,
        "best_model_file": best_weights.name,
        "best_training_episode": best_replay.training_episode,
        "history": {
            "episodes": [asdict(value) for value in records],
            "evaluations": [asdict(value) for value in evaluations],
            "replays": [asdict(value) for value in replays],
        },
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_manifest = Path(handle.name)
    os.replace(temporary_manifest, path)


if __name__ == "__main__":
    raise SystemExit(main())
