from __future__ import annotations

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
