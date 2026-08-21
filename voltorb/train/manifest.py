"""`run.json` -- one manifest per run, written at launch and updated at close.

A file format, not a framework. The point is that a run directory can answer "what exactly
produced this?" without archaeology through `runs/*.log`. Metrics stay where they already
are: `metrics_ref` points at the tfevents directory rather than copying anything.

`verdict` is deliberately left null by the trainer. Whether a run beat, tied or lost is a
judgement made against `tools/eval.py` afterwards, not something the training loop knows.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata as _md
from pathlib import Path

PROJECT = "voltorb"
_PACKAGES = ("torch", "gymnasium", "numpy", "pyboy")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ("git", *args), capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.rstrip("\n") if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return ""


def _provenance(device: str) -> dict:
    dirty = _git("status", "--porcelain")
    return {
        "git_sha": _git("rev-parse", "HEAD").strip(),
        "dirty_files": [line[3:] for line in dirty.splitlines()] if dirty else [],
        "host": platform.node(),
        "device": device,
        "python": platform.python_version(),
        "package_versions": {
            name: (_md.version(name) if _has(name) else None) for name in _PACKAGES
        },
    }


def _has(name: str) -> bool:
    try:
        _md.version(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def rom_sha1(path) -> str | None:
    """The ROM is gitignored, so its hash is the only durable identifier of the data."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def write(
    path,
    *,
    run_id: str,
    parent: str | None,
    config: dict,
    config_source: dict,
    data_ref: dict,
    metrics_ref: dict,
    device: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "id": run_id,
                "project": PROJECT,
                "parent": parent,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": config,
                "config_source": config_source,
                "data_ref": data_ref,
                "provenance": _provenance(device),
                "launch": list(sys.argv),
                "metrics_ref": metrics_ref,
                "verdict": None,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    return path


def close(path, *, status: str, **fields) -> None:
    """Merge finish info into an existing manifest. `verdict` stays null: it is closed by
    hand once tools/eval.py has something to say."""
    path = Path(path)
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    blob["status"] = status
    blob["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    blob.update(fields)
    path.write_text(json.dumps(blob, indent=2, default=str) + "\n")
