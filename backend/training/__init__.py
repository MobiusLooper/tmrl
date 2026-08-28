"""Headless agents, episode runners, and training metrics."""

from .agents import Agent, RandomAgent
from .evaluator import EvaluationRecord, evaluate_policy
from .runner import EpisodeRecord, RunSummary, run_episode, run_episodes, summarize_run
from .trainer import TrainingResult, run_training, run_training_episode

__all__ = [
    "Agent",
    "EpisodeRecord",
    "EvaluationRecord",
    "RandomAgent",
    "RunSummary",
    "TrainingResult",
    "evaluate_policy",
    "run_episode",
    "run_episodes",
    "run_training",
    "run_training_episode",
    "summarize_run",
]
