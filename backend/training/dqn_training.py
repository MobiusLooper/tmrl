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
from torch import nn

from backend.env.environment import LEARNING_ENVIRONMENT_VERSION, RacingEnv
from backend.rl.dqn import (
    OBSERVATION_SIZE,
    DQNAgent,
    DQNConfig,
    DQNPolicy,
    ReplayItem,
)

from .curriculum import AdaptiveCurriculum, CurriculumConfig
from .demonstrations import DemonstrationDataset, load_demonstration_dataset
from .evaluator import EvaluationRecord, evaluate_policy_with_replay
from .replay import EvaluationReplay
from .runner import EpisodeRecord
from .run_storage import RunMetadata, create_run, new_run_metadata, resume_run_metadata
from .trajectory import save_trajectory_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the continuous-observation DQN racer")
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--max-transitions", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--demonstrations", type=Path)
    parser.add_argument("--demonstration-epochs", type=_positive_int, default=200)
    parser.add_argument("--demonstration-mix", type=_fraction, default=0.25)
    parser.add_argument("--demonstration-bc-weight", type=_non_negative_float, default=0.1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is not None and args.checkpoint is not None:
        raise SystemExit("--checkpoint cannot be combined with --resume")
    if args.resume is not None and args.demonstrations is not None:
        raise SystemExit("--demonstrations cannot be combined with --resume")
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
        try:
            agent = DQNAgent.from_snapshot(state["agent"])
        except ValueError as error:
            raise SystemExit(f"incompatible checkpoint {args.resume}: {error}") from error
        curriculum = AdaptiveCurriculum.from_snapshot(state["curriculum"])
        records = list(state["records"])
        evaluations = list(state["evaluations"])
        replays = list(state["replays"])
        start_episode = int(state["episode"]) + 1
        wall_time = float(state["wall_time"])
        pretraining = state.get("pretraining")
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
        agent = DQNAgent(
            Random(seed),
            DQNConfig(
                demonstration_mix=args.demonstration_mix,
                demonstration_bc_weight=args.demonstration_bc_weight,
            ),
            seed=seed,
        )
        curriculum = _new_curriculum(seed)
        records: list[EpisodeRecord] = []
        evaluations: list[EvaluationRecord] = []
        replays: list[EvaluationReplay] = []
        start_episode = 1
        wall_time = 0.0
        pretraining: object = None
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint
            run_metadata = new_run_metadata("dqn", seed)
        else:
            run_metadata, checkpoint_path = create_run("dqn", seed)

    print(f"run {run_metadata.run_id} | checkpoint {checkpoint_path}")

    if args.demonstrations is not None:
        try:
            dataset = load_demonstration_dataset(args.demonstrations)
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        agent.set_demonstration_buffer(_demonstration_replay_items(dataset, agent.config))
        evaluation, replay, statistics = pretrain_dqn(
            agent,
            dataset,
            epochs=args.demonstration_epochs,
            seed=seed,
        )
        evaluations.append(evaluation)
        replays.append(replay)
        pretraining = dataset.provenance(
            method="behavior-cloning-and-replay",
            demonstration_replay_items=len(agent.demonstration_buffer),
            demonstration_mix=agent.config.demonstration_mix,
            demonstration_bc_weight=agent.config.demonstration_bc_weight,
            **statistics,
        )
        print(
            f"demonstrations | laps {len(dataset.laps)} | physics {dataset.transition_count} | "
            f"replay {len(agent.demonstration_buffer)} | "
            f"agreement {float(statistics['eligible_action_agreement']):.1%}"
        )
        print(
            f"evaluation @0 | progress {evaluation.mean_progress:.1%} | "
            f"laps {evaluation.lap_completions}"
        )
        _save(
            checkpoint_path,
            agent,
            curriculum,
            records,
            evaluations,
            replays,
            0,
            wall_time,
            seed,
            run_metadata.touch(),
            pretraining,
        )

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
                pretraining,
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


def _new_curriculum(seed: int) -> AdaptiveCurriculum:
    return AdaptiveCurriculum(
        Random(seed + 2_000_000),
        CurriculumConfig(final_rehearsal_probability=0.5),
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
    pretraining: object = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights, best_weights = _model_paths(path)
    state = {
        "agent": agent.snapshot(), "curriculum": curriculum.snapshot(), "records": records,
        "evaluations": evaluations, "replays": replays, "episode": episode, "wall_time": wall_time,
        "pretraining": pretraining,
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
            "pretraining": pretraining,
        },
        "agent": {
            "config": asdict(agent.config),
            "epsilon": agent.epsilon,
            "transitions": agent.transitions,
            "observation_size": OBSERVATION_SIZE,
            "learning_environment_version": LEARNING_ENVIRONMENT_VERSION,
        },
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
    save_trajectory_catalog(path, run_metadata.run_id, replays)


def _model_paths(path: Path) -> tuple[Path, Path]:
    if path.name == "checkpoint.json":
        return path.with_name("model.pt"), path.with_name("best-model.pt")
    return path.with_suffix(".pt"), path.with_name(f"{path.stem}-best.pt")


def _demonstration_replay_items(
    dataset: DemonstrationDataset,
    config: DQNConfig,
) -> tuple[ReplayItem, ...]:
    items: list[ReplayItem] = []
    for lap in dataset.laps:
        for start, transition in enumerate(lap.transitions):
            total = 0.0
            next_observation = transition.next_observation
            done = False
            used = 0
            for candidate in lap.transitions[start : start + config.n_step]:
                total += config.discount**used * candidate.reward
                used += 1
                next_observation = candidate.next_observation
                done = candidate.done
                if done:
                    break
            items.append(
                ReplayItem(
                    transition.observation,
                    int(transition.action),
                    total,
                    next_observation,
                    done,
                    config.discount**used,
                )
            )
    return tuple(items)


def pretrain_dqn(
    agent: DQNAgent,
    dataset: DemonstrationDataset,
    *,
    epochs: int,
    seed: int,
) -> tuple[EvaluationRecord, EvaluationReplay, dict[str, object]]:
    if epochs < 1:
        raise ValueError("demonstration epochs must be positive")
    samples = [
        (transition.observation, int(transition.action))
        for lap in dataset.laps
        for transition in lap.transitions
        if transition.action in agent.eligible_actions(transition.observation)
    ]
    if not samples:
        raise ValueError("DQN behavior cloning requires eligible demonstration actions")
    observations = torch.tensor([sample[0] for sample in samples], dtype=torch.float32)
    actions = torch.tensor([sample[1] for sample in samples], dtype=torch.int64)
    generator = torch.Generator().manual_seed(seed + 3_000_000)
    evaluation_interval = min(10, epochs)
    best_score: tuple[object, ...] | None = None
    best_weights: dict[str, torch.Tensor] | None = None
    best_evaluation: EvaluationRecord | None = None
    best_replay: EvaluationReplay | None = None
    selected_epoch = 0

    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(samples), generator=generator)
        for offset in range(0, len(samples), agent.config.batch_size):
            indices = order[offset : offset + agent.config.batch_size]
            loss = nn.functional.cross_entropy(
                agent.online(observations.index_select(0, indices)),
                actions.index_select(0, indices),
            )
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.online.parameters(), agent.config.gradient_clip)
            agent.optimizer.step()
        if epoch % evaluation_interval != 0 and epoch != epochs:
            continue
        evaluation, replay = evaluate_policy_with_replay(
            RacingEnv(), DQNPolicy(agent), 1, training_episode=0
        )
        score = (
            evaluation.lap_completions,
            evaluation.best_progress,
            evaluation.mean_return,
            -replay.simulated_duration,
        )
        if best_score is None or score > best_score:
            best_score = score
            best_weights = {
                name: value.detach().clone() for name, value in agent.online.state_dict().items()
            }
            best_evaluation = evaluation
            best_replay = replay
            selected_epoch = epoch

    if best_weights is None or best_evaluation is None or best_replay is None:
        raise RuntimeError("DQN demonstration pretraining did not produce an evaluation")
    agent.online.load_state_dict(best_weights)
    agent.target.load_state_dict(best_weights)
    agent.reset_optimizer()
    with torch.no_grad():
        agreement = float((agent.online(observations).argmax(dim=1) == actions).float().mean())
    return best_evaluation, best_replay, {
        "epochs": epochs,
        "selected_epoch": selected_epoch,
        "eligible_action_samples": len(samples),
        "eligible_action_agreement": agreement,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < 1:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
