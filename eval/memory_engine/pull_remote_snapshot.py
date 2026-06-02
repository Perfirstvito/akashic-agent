from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 local fallback
    import tomli as tomllib  # type: ignore[no-redef]


DEFAULT_CONFIG = Path("eval/memory_engine/remote_sync.local.toml")
REMOTE_SCRIPT = Path(__file__).with_name("snapshot_workspace.py")
SNAPSHOT_NAME_RE = re.compile(r"^\d{8}T\d{6}Z(?:-\d{2})?$")
SSH_BATCH_OPTIONS = ("-o", "BatchMode=yes")


@dataclass(frozen=True)
class RemoteSettings:
    host: str
    workspace: str
    snapshot_root: str
    python: str
    ssh_options: tuple[str, ...]
    retention: int


@dataclass(frozen=True)
class LocalSettings:
    snapshot_root: Path
    retention: int


@dataclass(frozen=True)
class ExtractSettings:
    enabled: bool
    badcase_dir: str
    report: str
    no_source_messages: bool


@dataclass(frozen=True)
class SyncSettings:
    remote: RemoteSettings
    local: LocalSettings
    extract: ExtractSettings


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _required_text(data: dict[str, Any], key: str) -> str:
    value = str(data.get(key) or "").strip()
    if not value:
        raise ValueError(f"missing required config value: {key}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"config value must be one line: {key}")
    return value


def _load_settings(path: Path) -> SyncSettings:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    remote = _as_dict(payload.get("remote"))
    local = _as_dict(payload.get("local"))
    extract = _as_dict(payload.get("extract"))
    ssh_options = remote.get("ssh_options") or []
    if not isinstance(ssh_options, list) or not all(isinstance(item, str) for item in ssh_options):
        raise ValueError("remote.ssh_options must be a list of strings")
    return SyncSettings(
        remote=RemoteSettings(
            host=_required_text(remote, "host"),
            workspace=_required_text(remote, "workspace"),
            snapshot_root=_required_text(remote, "snapshot_root"),
            python=str(remote.get("python") or "python3").strip(),
            ssh_options=tuple(ssh_options),
            retention=max(1, int(remote.get("retention", 7))),
        ),
        local=LocalSettings(
            snapshot_root=Path(_required_text(local, "snapshot_root")),
            retention=max(1, int(local.get("retention", 30))),
        ),
        extract=ExtractSettings(
            enabled=bool(extract.get("enabled", False)),
            badcase_dir=str(
                extract.get("badcase_dir")
                or "eval/memory_engine/badcases/remote/{stamp}"
            ),
            report=str(
                extract.get("report")
                or "eval/memory_engine/reports/remote_extract_{stamp}.jsonl"
            ),
            no_source_messages=bool(extract.get("no_source_messages", False)),
        ),
    )


def _run_remote_snapshot(settings: SyncSettings) -> dict[str, Any]:
    remote = settings.remote
    remote_command = shlex.join(
        [
            remote.python,
            "-",
            "--workspace",
            remote.workspace,
            "--output-root",
            remote.snapshot_root,
            "--retention",
            str(remote.retention),
        ]
    )
    completed = subprocess.run(
        ["ssh", *SSH_BATCH_OPTIONS, *remote.ssh_options, remote.host, remote_command],
        input=REMOTE_SCRIPT.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("remote snapshot command returned no output")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("remote snapshot command returned an invalid payload")
    return payload


def _copy_remote_snapshot(
    settings: SyncSettings,
    *,
    remote_path: str,
    snapshot_name: str,
) -> Path:
    if not SNAPSHOT_NAME_RE.fullmatch(snapshot_name):
        raise ValueError(f"invalid remote snapshot name: {snapshot_name!r}")
    local_root = settings.local.snapshot_root.expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    final_dir = local_root / snapshot_name
    partial_dir = local_root / f".{snapshot_name}.{os.getpid()}.partial"
    if final_dir.exists():
        raise FileExistsError(f"local snapshot already exists: {final_dir}")
    if partial_dir.exists():
        shutil.rmtree(partial_dir)

    source = f"{settings.remote.host}:{remote_path.rstrip('/')}/"
    try:
        subprocess.run(
            [
                "scp",
                *SSH_BATCH_OPTIONS,
                *settings.remote.ssh_options,
                "-r",
                source,
                str(partial_dir),
            ],
            check=True,
        )
        _verify_manifest(partial_dir)
        partial_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    return final_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(raw: object) -> Path:
    pure = PurePosixPath(str(raw or ""))
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe manifest path: {raw!r}")
    return Path(*pure.parts)


def _verify_manifest(snapshot_dir: Path) -> None:
    manifest_path = snapshot_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("manifest files must be a list")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest file entry must be an object")
        relative = _safe_relative_path(item.get("path"))
        path = snapshot_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"snapshot file is missing: {path}")
        expected_size = int(item.get("bytes", -1))
        if path.stat().st_size != expected_size:
            raise ValueError(f"snapshot file size mismatch: {relative}")
        if _sha256(path) != str(item.get("sha256") or ""):
            raise ValueError(f"snapshot file checksum mismatch: {relative}")


def _update_latest(local_root: Path, snapshot_dir: Path) -> None:
    latest_txt = local_root / "latest.txt"
    latest_txt.write_text(snapshot_dir.name + "\n", encoding="utf-8")
    latest_link = local_root / "latest"
    temporary_link = local_root / ".latest.tmp"
    if temporary_link.exists() or temporary_link.is_symlink():
        temporary_link.unlink()
    try:
        temporary_link.symlink_to(snapshot_dir.name, target_is_directory=True)
        temporary_link.replace(latest_link)
    except OSError:
        if temporary_link.exists() or temporary_link.is_symlink():
            temporary_link.unlink()


def _cleanup_local_snapshots(root: Path, *, keep: int) -> None:
    snapshots = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and SNAPSHOT_NAME_RE.fullmatch(path.name)
    )
    for path in snapshots[: max(0, len(snapshots) - max(1, keep))]:
        shutil.rmtree(path)


def _extract_badcases(settings: SyncSettings, snapshot_dir: Path) -> None:
    stamp = snapshot_dir.name
    replacements = {"stamp": stamp}
    command = [
        sys.executable,
        "-m",
        "eval.memory_engine.daily_badcase_extract",
        "--workspace",
        str(snapshot_dir),
        "--badcase-dir",
        settings.extract.badcase_dir.format_map(replacements),
        "--report",
        settings.extract.report.format_map(replacements),
    ]
    if settings.extract.no_source_messages:
        command.append("--no-source-messages")
    subprocess.run(command, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a remote memory snapshot over SSH and pull it into an ignored local directory."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override extract.enabled from config.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = _load_settings(args.config.expanduser().resolve())
    remote_result = _run_remote_snapshot(settings)
    snapshot_name = _required_text(remote_result, "snapshot_name")
    remote_path = _required_text(remote_result, "snapshot_dir")
    snapshot_dir = _copy_remote_snapshot(
        settings,
        remote_path=remote_path,
        snapshot_name=snapshot_name,
    )
    local_root = settings.local.snapshot_root.expanduser().resolve()
    _update_latest(local_root, snapshot_dir)
    _cleanup_local_snapshots(local_root, keep=settings.local.retention)

    should_extract = settings.extract.enabled if args.extract is None else args.extract
    if should_extract:
        _extract_badcases(settings, snapshot_dir)

    print(
        json.dumps(
            {
                "snapshot_dir": str(snapshot_dir),
                "latest": str(local_root / "latest"),
                "extracted_badcases": should_extract,
                "remote": remote_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
