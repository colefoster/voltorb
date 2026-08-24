"""Check the --out envelope from voltorb.tools.eval on synthetic results.

Evaluating for real needs the ROM and ~a minute per arm, so this exercises only
the serialization path -- the part that a downstream reader depends on.

    uv run python -m voltorb.tools.test_eval_json
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from voltorb.tools.eval import write_results


def _row(name: str, scale: float) -> dict:
    """The exact key set evaluate() returns, so a drift there fails here."""
    rng = np.random.default_rng(7)
    n = 5
    scores = rng.random(n) * 1e6 * scale
    frames = rng.random(n) * 1000 + 9000
    alley = rng.integers(0, 4, n).astype(float)
    arms = rng.integers(0, 3, n).astype(float)
    entries = rng.integers(0, 2, n).astype(float)
    dex = rng.integers(0, 2, n).astype(float)
    se = lambda a: a.std(ddof=1) / np.sqrt(n)
    rate = lambda a: a.sum() / frames.sum() * 10_000.0
    return {
        "checkpoint": name,
        "n": n,
        "frames_mean": frames.mean(),
        "frames_max": frames.max(),
        "frames_stderr": se(frames),
        "score_mean": scores.mean(),
        "score_median": np.median(scores),
        "score_max": scores.max(),
        "score_rate": rate(scores),
        "dex_mean": dex.mean(),
        "dex_stderr": se(dex),
        "dex_hit_rate": float((dex >= 1).mean()) * 100.0,
        "caught_mean": dex.mean(),
        "visits_mean": entries.mean(),
        "visits_stderr": se(entries),
        "visit_rate": rate(entries),
        "entries_mean": entries.mean(),
        "entries_stderr": se(entries),
        "entry_rate": rate(entries),
        "slots_entered_mean": entries.mean(),
        "alley_mean": alley.mean(),
        "alley_stderr": se(alley),
        "alley_rate": rate(alley),
        "arms_mean": arms.mean(),
        "arms_stderr": se(arms),
        "arms_rate": rate(arms),
        "_scores": scores,
        "_frames": frames,
        "_alley": alley,
        "_arms": arms,
        "_entries": entries,
        "_dex": dex,
    }


def main() -> None:
    rows = [_row("random", 1.0), _row("runs/fake-01/final.pt", 1.4)]
    comparison = {
        "baseline": "random",
        "metrics": {
            "score_rate": {
                "label": "score per 10k frames",
                "checkpoints": {
                    "runs/fake-01/final.pt": {
                        "mean": 1.0, "lo": 0.5, "hi": 1.5,
                        "sigma": 3.0, "significant": True,
                    }
                },
            }
        },
        "per_episode_mean": {},
    }
    args = argparse.Namespace(
        stage="dex", episodes=5, seed=1234, frame_skip=1, num_envs=8,
        rom="roms/does-not-exist.gbc",
        checkpoint=[r["checkpoint"] for r in rows],
    )

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "run.json"
        write_results(out, rows, args, comparison)
        payload = json.loads(out.read_text())

    # --- the shared envelope, as documented in eval.py and goldeneye's eval.py
    for key in ("project", "checkpoint", "n", "metrics", "comparison",
                "episodes", "params", "created_at"):
        assert key in payload, f"envelope is missing {key}"
    assert payload["project"] == "voltorb"
    # The candidate, not the baseline: `random` is what it is compared against.
    assert payload["checkpoint"] == "runs/fake-01/final.pt"
    assert payload["n"] == 5
    assert payload["created_at"].endswith("Z")

    # --- every scalar survives, with its stderr attached rather than orphaned
    cand = rows[1]
    scalars = {k for k in cand
               if not k.startswith("_") and k not in ("checkpoint", "n")
               and not k.endswith("_stderr")}
    assert set(payload["metrics"]) == scalars, (
        set(payload["metrics"]) ^ scalars)
    for name, m in payload["metrics"].items():
        assert set(m) == {"value", "stderr", "n"}, name
        assert m["n"] == 5
        assert isinstance(m["value"], float) and m["value"] == m["value"], name
    assert payload["metrics"]["dex_mean"]["stderr"] == cand["dex_stderr"]
    # A `_max` must not steal its family's standard error.
    assert payload["metrics"]["frames_max"]["stderr"] is None
    assert payload["metrics"]["score_rate"]["stderr"] is None

    # --- per-episode arrays, which are what make the bootstrap re-runnable
    eps = payload["episodes"]
    assert set(eps) == {"scores", "frames", "alley_shots", "arms",
                        "catch_entries", "dex"}, set(eps)
    for name, arr in eps.items():
        assert len(arr) == 5, name
        assert arr == [float(x) for x in cand["_" + {"scores": "scores",
                       "frames": "frames", "alley_shots": "alley",
                       "arms": "arms", "catch_entries": "entries",
                       "dex": "dex"}[name]]], name

    # --- comparison rides along untouched
    assert payload["comparison"] == comparison
    hit = payload["comparison"]["metrics"]["score_rate"]["checkpoints"]
    assert hit["runs/fake-01/final.pt"]["significant"] is True

    # --- params describe what produced the run
    p = payload["params"]
    assert p["stage"] == "dex" and p["episodes"] == 5 and p["seed"] == 1234
    assert p["checkpoints"] == ["random", "runs/fake-01/final.pt"]
    assert p["rom_sha256"] is None, "a missing ROM should be null, not a crash"

    # --- both arms are kept, so nothing the run measured is dropped
    assert set(payload["checkpoints"]) == {"random", "runs/fake-01/final.pt"}
    assert payload["checkpoints"]["random"]["episodes"]["scores"] == [
        float(x) for x in rows[0]["_scores"]]

    # --- comparison=None is legal (single-arm runs write no bootstrap)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "solo.json"
        write_results(out, [rows[0]], args, None)
        solo = json.loads(out.read_text())
    assert solo["comparison"] is None
    assert solo["checkpoint"] == "random", "a lone baseline is still the subject"

    print(f"ok: envelope has {len(payload['metrics'])} metrics, "
          f"{len(payload['episodes'])} episode arrays, "
          f"{len(payload['checkpoints'])} checkpoints")


if __name__ == "__main__":
    main()
