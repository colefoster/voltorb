"""`run.json` -- one manifest per run, written at launch and updated at close.

**The writer itself lives in `runmanifest`**, the shared package in
`~/Dev/model-tuner` that voltorb, mimikyu and goldeneye all import
(`uv pip install -e ~/Dev/model-tuner`). This module is voltorb's half of that
contract: the repo root, the package list, the config namespaces, and the ROM
hash that only this project knows how to compute. See `model-tuner/SCHEMA.md`.

The point is that a run directory can answer "what exactly produced this?"
without archaeology through `runs/*.log`. Metrics stay where they already are:
`metrics_ref` points at the tfevents directory rather than copying anything.

`verdict` is deliberately left null by the trainer. Whether a run beat, tied or
lost is a judgement made against `tools/eval.py` afterwards, not something the
training loop knows -- `close()` here records only the finish *status*.

**If `runmanifest` is not installed**, this degrades to a no-op with one warning
rather than taking a training run with it.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT = "voltorb"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGES = ("torch", "gymnasium", "numpy", "pyboy")
# `vars(args)` versus the resolved env dataclass versus the reward kwargs. Three
# genuinely different namespaces, so the keys keep their prefix; the writer strips
# it only to match a key against the command line.
_NAMESPACES = ("args.", "env.", "reward.")

try:
    import runmanifest as _rm
except ImportError:  # pragma: no cover - only on a checkout without the install
    _rm = None

_WRITER = (
    _rm.ManifestWriter(PROJECT, _REPO_ROOT, packages=_PACKAGES,
                       config_namespaces=_NAMESPACES)
    if _rm else None
)


def _unavailable() -> None:
    print("run manifest skipped: `runmanifest` is not installed "
          "(uv pip install -e ~/Dev/model-tuner)", file=sys.stderr, flush=True)


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
    data_ref: dict,
    metrics_ref,
    device: str,
) -> Path | None:
    """Write `runs/<name>/run.json`. Returns None if it could not be written."""
    if _WRITER is None:
        _unavailable()
        return None
    return _WRITER.write(
        path,
        run_id=run_id,
        parent=parent,
        config=config,
        data_ref=data_ref,
        metrics_ref=metrics_ref,
        device=device,
    )


def close(path, *, status: str, **fields) -> None:
    """Merge finish info into an existing manifest. `verdict` stays null: it is closed by
    hand once tools/eval.py has something to say."""
    if _WRITER is None:
        return
    _WRITER.finish(path, status=status, **fields)
