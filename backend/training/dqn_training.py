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
from .run_storage import RunMetadata, create_run, new_run_metadata, resume_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the continuous-observation DQN racer")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--max-transitions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is not None and args.checkpoint is not None:
        raise SystemExit("--checkpoint cannot be combined with --resume")
    if args.resume:
        manifest = _load_manifest(args.resume)
        run = manifest.get("run")
        run_mapping = run if isinstance(run, dict) else {}
        saved_seed = run_mapping.get("seed")
        if not isinstance(saved_seed, int) or isinstance(saved_seed, bool):
            raise SystemExit(f"invalid checkpoint {args.resume}: run.seed must be an integer")
        if args.seed is not None and args.seed != saved_seed:
            raise SystemExit(f"--seed={args.seed} conflicts with saved value {saved_seed}")
        seed = saved_seed
        model_path = _resume_model_path(args.resume, manifest)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        agent = DQNAgent.from_snapshot(state["agent"])
        curriculum = AdaptiveCurriculum.from_snapshot(state["curriculum"])
        records = list(state["records"])
        evaluations = list(state["evaluations"])
        replays = list(state["replays"])
        start_episode = int(state["episode"]) + 1
        wall_time = float(state["wall_time"])
        run_metadata = resume_run_metadata(
            args.resume,
            run_id=run_mapping.get("run_id") if isinstance(run_mapping.get("run_id"), str) else None,
            created_at=(
                run_mapping.get("created_at")
                if isinstance(run_mapping.get("created_at"), str)
                else None
            ),
        )
        checkpoint_path = args.resume
        if args.episodes < start_episode:
            raise SystemExit(
                f"target episodes ({args.episodes}) must exceed checkpoint episode {start_episode - 1}"
            )
    else:
        seed = args.seed if args.seed is not None else 0
        agent = DQNAgent(Random(seed), DQNConfig(), seed=seed)
        curriculum = AdaptiveCurriculum(Random(seed + 2_000_000))
        records: list[EpisodeRecord] = []
        evaluations: list[EvaluationRecord] = []
        replays: list[EvaluationReplay] = []
        start_episode = 1
        wall_time = 0.0
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
            run_metadata = new_run_metadata("dqn", seed)
        else:
            run_metadata, checkpoint_path = create_run("dqn", seed)

    print(f"run {run_metadata.run_id} | checkpoint {checkpoint_path}")

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
        if (
            episode % args.evaluate_every == 0
            or agent.transitions >= args.max_transitions
            or episode == args.episodes
        ):
            evaluation, replay = evaluate_policy_with_replay(
                environment, DQNPolicy(agent), 1, training_episode=episode
            )
            evaluations.append(evaluation)
            replays.append(replay)
            elapsed = wall_time + perf_counter() - started
            _save(
                checkpoint_path,
                agent,
                curriculum,
                records,
                evaluations,
                replays,
                episode,
                elapsed,
                seed,
                run_metadata.touch(),
            )
            print(
                f"evaluation @{episode} | progress {evaluation.mean_progress:.1%} | "
                f"laps {evaluation.lap_completions} | epsilon {agent.epsilon:.4f}"
            )
            if evaluation.lap_completions:
                print(f"canonical lap completed in {replay.simulated_duration:.2f}s")
        if agent.transitions >= args.max_transitions:
            break
    completed_episode = records[-1].episode if records else start_episode - 1
    print(
        json.dumps(
            {
                "run_id": run_metadata.run_id,
                "checkpoint": str(checkpoint_path),
                "completed_episode": completed_episode,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to read checkpoint {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid checkpoint {path}: expected a JSON object")
    return payload


def _resume_model_path(checkpoint_path: Path, manifest: dict[str, object]) -> Path:
    model_file = manifest.get("model_file")
    return (
        checkpoint_path.parent / model_file
        if isinstance(model_file, str)
        else checkpoint_path.with_suffix(".pt")
    )


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
    run_metadata: RunMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights, best_weights = _model_paths(path)
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
    if best_replay is replays[-1]:
        with NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary_best = Path(handle.name)
        torch.save(state, temporary_best)
        os.replace(temporary_best, best_weights)
    payload = {
        "schema_version": 2,
        "algorithm": "dqn",
        "run": {
            "seed": seed,
            "completed_episode": episode,
            "training_wall_time": wall_time,
            "run_id": run_metadata.run_id,
            "created_at": run_metadata.created_at,
            "updated_at": run_metadata.updated_at,
        },
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


def _model_paths(path: Path) -> tuple[Path, Path]:
    if path.name == "checkpoint.json":
        return path.with_name("model.pt"), path.with_name("best-model.pt")
    return path.with_suffix(".pt"), path.with_name(f"{path.stem}-best.pt")


if __name__ == "__main__":
    raise SystemExit(main())
