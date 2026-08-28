from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from secrets import token_hex


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
DEFAULT_RUNS_DIR = DEFAULT_ARTIFACTS_DIR / "runs"


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    created_at: str
    updated_at: str

    def touch(self) -> RunMetadata:
        return RunMetadata(self.run_id, self.created_at, utc_now())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_run(
    algorithm: str,
    seed: int,
    *,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> tuple[RunMetadata, Path]:
    created_at = utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{algorithm}-seed{seed}-{token_hex(4)}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return RunMetadata(run_id, created_at, created_at), run_dir / "checkpoint.json"


def new_run_metadata(algorithm: str, seed: int) -> RunMetadata:
    created_at = utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RunMetadata(
        f"{timestamp}-{algorithm}-seed{seed}-{token_hex(4)}",
        created_at,
        created_at,
    )


def resume_run_metadata(
    path: Path,
    *,
    run_id: str | None,
    created_at: str | None,
) -> RunMetadata:
    modified_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    inferred_id = run_id or inferred_run_id(path)
    return RunMetadata(inferred_id, created_at or modified_at, utc_now())


def inferred_run_id(path: Path) -> str:
    if path.name == "checkpoint.json" and path.parent.parent.name == "runs":
        return path.parent.name
    digest = sha256(str(path.resolve()).encode()).hexdigest()[:8]
    safe_stem = "".join(
        character if character.isalnum() else "-" for character in path.stem
    ).strip("-")
    return f"legacy-{safe_stem or 'run'}-{digest}"
