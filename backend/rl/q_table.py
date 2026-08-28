from __future__ import annotations

from math import isfinite
from typing import Mapping, Sequence

from backend.env.environment import DiscreteAction

from .discretisation import DiscreteState

ACTION_COUNT = len(DiscreteAction)
ZERO_VALUES = (0.0,) * ACTION_COUNT


class QTable:
    def __init__(self) -> None:
        self._rows: dict[DiscreteState, list[float]] = {}

    def values(self, state: DiscreteState) -> tuple[float, ...]:
        row = self._rows.get(state)
        return ZERO_VALUES if row is None else tuple(row)

    def value(self, state: DiscreteState, action: DiscreteAction) -> float:
        row = self._rows.get(state)
        return 0.0 if row is None else row[int(action)]

    def set_value(self, state: DiscreteState, action: DiscreteAction, value: float) -> None:
        row = self._rows.setdefault(state, [0.0] * ACTION_COUNT)
        row[int(action)] = float(value)

    @property
    def visited_states(self) -> int:
        return len(self._rows)

    def snapshot(self) -> dict[DiscreteState, tuple[float, ...]]:
        return {state: tuple(values) for state, values in self._rows.items()}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[Sequence[int], Sequence[float]],
        *,
        bucket_count: int,
    ) -> QTable:
        if bucket_count < 2:
            raise ValueError("bucket_count must be at least 2")
        table = cls()
        for raw_state, raw_values in snapshot.items():
            state = tuple(raw_state)
            values = tuple(float(value) for value in raw_values)
            if len(state) not in {6, 7, 9} or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in state
            ):
                raise ValueError(
                    "Q-table states must contain six or nine legacy, or seven smooth bucket indices"
                )
            if len(state) == 7 and (
                any(value > 5 for value in state[:5])
                or state[5] > 4
                or state[6] > 3
            ):
                raise ValueError("smooth Q-table state indices are outside the tabular-v3 bounds")
            if len(values) != ACTION_COUNT or any(not isfinite(value) for value in values):
                raise ValueError(f"Q-table rows must contain {ACTION_COUNT} finite values")
            table._rows[state] = list(values)
        return table
