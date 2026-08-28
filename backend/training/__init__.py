"""Headless agents, episode runners, and training metrics."""

from .agents import Agent, RandomAgent
from .checkpoint import CheckpointError, TrainingCheckpoint, load_checkpoint, load_checkpoint_replays, save_checkpoint
from .evaluator import EvaluationRecord, evaluate_policy, evaluate_policy_with_replay
from .replay import EvaluationReplay, ReplayState, ReplayTransition
from .runner import EpisodeRecord, RunSummary, run_episode, run_episodes, summarize_run
from .trainer import TrainingResult, run_training, run_training_episode

__all__ = [
    "Agent",
    "CheckpointError",
    "EpisodeRecord",
    "EvaluationRecord",
    "EvaluationReplay",
    "RandomAgent",
    "ReplayState",
    "ReplayTransition",
    "RunSummary",
    "TrainingResult",
    "TrainingCheckpoint",
    "evaluate_policy",
    "evaluate_policy_with_replay",
    "load_checkpoint",
    "load_checkpoint_replays",
    "run_episode",
    "run_episodes",
    "run_training",
    "run_training_episode",
    "save_checkpoint",
    "summarize_run",
]
