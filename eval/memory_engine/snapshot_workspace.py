from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_SQLITE_FILES = (
    "sessions.db",
    "memory/memory2.db",
)
OPTIONAL_SQLITE_FILES = (
    "observe/observe.db",
)
OPTIONAL_JSONL_FILES = (
    "observe/recall_inspector.jsonl",
)
SNAPSHOT_NAME_RE = re.compile(r"^\d{8}T\d{6}Z(?:-\d{2})?$")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(path: Path, *, root: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "kind": kind,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"{source.resolve().as_uri()}?mode=ro"
    src = sqlite3.connect(source_uri, uri=True)
    dst = sqlite3.connect(str(destination))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def _copy_complete_jsonl(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    with destination.open("rb+") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        while position:
            chunk_size = min(position, 8192)
            position -= chunk_size
            handle.seek(position)
            chunk = handle.read(chunk_size)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                handle.truncate(position + newline + 1)
                return
        handle.truncate(0)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _allocate_snapshot_dir(root: Path) -> Path:
    stamp = _now_stamp()
    candidate = root / stamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{stamp}-{suffix:02d}"
    return candidate


def _cleanup_old_snapshots(root: Path, *, keep: int) -> None:
    snapshots = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and SNAPSHOT_NAME_RE.fullmatch(path.name)
    )
    for path in snapshots[: max(0, len(snapshots) - max(1, keep))]:
        shutil.rmtree(path)


def create_snapshot(
    workspace: Path,
    output_root: Path,
    *,
    retention: int,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    final_dir = _allocate_snapshot_dir(output_root)
    partial_dir = output_root / f".{final_dir.name}.{os.getpid()}.partial"
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    partial_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    try:
        for relative in REQUIRED_SQLITE_FILES:
            source = workspace / relative
            if not source.exists():
                raise FileNotFoundError(f"required workspace file is missing: {source}")
            destination = partial_dir / relative
            _snapshot_sqlite(source, destination)
            entries.append(_manifest_entry(destination, root=partial_dir, kind="sqlite"))

        for relative in OPTIONAL_SQLITE_FILES:
            source = workspace / relative
            if not source.exists():
                continue
            destination = partial_dir / relative
            _snapshot_sqlite(source, destination)
            entries.append(_manifest_entry(destination, root=partial_dir, kind="sqlite"))

        for relative in OPTIONAL_JSONL_FILES:
            source = workspace / relative
            if not source.exists():
                continue
            destination = partial_dir / relative
            _copy_complete_jsonl(source, destination)
            entries.append(_manifest_entry(destination, root=partial_dir, kind="jsonl"))

        memory_dir = workspace / "memory"
        if memory_dir.exists():
            for source in sorted(memory_dir.glob("*.json")):
                destination = partial_dir / "memory" / source.name
                _copy_file(source, destination)
                entries.append(_manifest_entry(destination, root=partial_dir, kind="json"))

        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "source_workspace": str(workspace),
            "files": entries,
        }
        (partial_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    _cleanup_old_snapshots(output_root, keep=retention)
    return {
        "snapshot_dir": str(final_dir),
        "snapshot_name": final_dir.name,
        "file_count": len(entries),
        "manifest": str(final_dir / "manifest.json"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a consistent read-only snapshot of memory-engine workspace files."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retention", type=int, default=7)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = create_snapshot(
        args.workspace,
        args.output_root,
        retention=max(1, args.retention),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
