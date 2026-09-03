from __future__ import annotations

import argparse
from typing import Sequence

from . import dqn_training, q_learning


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--algorithm", choices=("tabular", "dqn"), required=True)
    known, remaining = parser.parse_known_args(argv)
    return dqn_training.main(remaining) if known.algorithm == "dqn" else q_learning.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
