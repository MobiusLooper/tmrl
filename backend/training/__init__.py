"""Headless agents, episode runners, and training metrics."""

from .agents import Agent, RandomAgent
from .runner import EpisodeRecord, RunSummary, run_episode, run_episodes, summarize_run

__all__ = [
    "Agent",
    "EpisodeRecord",
    "RandomAgent",
    "RunSummary",
    "run_episode",
    "run_episodes",
    "summarize_run",
]
