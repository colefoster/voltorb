"""Evaluate checkpoints properly, because the training scalars are a rolling window.

Compares any number of checkpoints (and a random baseline) over the same number of
episodes, and reports the numbers the project is actually judged on:

    uv run python -m voltorb.tools.eval --episodes 20 \\
        --checkpoint random \\
        --checkpoint runs/survive-03/final.pt \\
        --checkpoint runs/score-01/final.pt

The column that matters is `catchmd` -- catch-mode entries per episode. It decomposes as
`visits` (ball resting in the saucer at (124,120)) times whether the saucer was ready, and
~80% of entries become a catch unaided, so dex/game is mostly a noisy function of it. Random
measures 1.55 visits and 0.88 entries. The old "1 new species in >=90% of games" bar is
retired: random clears it 72-78% of the time.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from voltorb.env import OBS_DIM
from voltorb.train.ppo import (
    ActorCritic,
    _env_thunk,
    check_obs_dim,
    load_checkpoint,
)


def evaluate(checkpoint: str, args) -> dict:
    envs = gym.vector.AsyncVectorEnv(
        [
            functools.partial(_env_thunk, args.rom, args.stage, args.frame_skip, args.seed + i)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    # Seed torch too, not just numpy: the policy arms sample their actions from torch's
    # global RNG, so without this an arm's result depends on how many arms ran before it in
    # the same process -- saucer-01 measured 1.22 catch entries alone and 1.12 in a three-arm
    # invocation. Matched evals are the whole point of this tool.
    torch.manual_seed(args.seed)

    model = None
    if checkpoint != "random":
        model = ActorCritic()
        ckpt = load_checkpoint(checkpoint)
        check_obs_dim(ckpt, OBS_DIM, checkpoint)
        model.load_state_dict(ckpt["model"])
        model.eval()

    rng = np.random.default_rng(args.seed)
    obs, _ = envs.reset(seed=args.seed)
    episodes: list[dict] = []

    with torch.no_grad():
        while len(episodes) < args.episodes:
            if model is None:
                action = rng.integers(0, 4, size=args.num_envs)
            else:
                logits, _ = model(torch.as_tensor(obs, dtype=torch.float32))
                action = Categorical(logits=logits).sample().numpy()
            obs, _, terminated, truncated, infos = envs.step(action)
            final = infos.get("final_info")
            if not final:
                continue
            mask = np.asarray(
                final.get("_score", np.logical_or(terminated, truncated)), dtype=bool
            )
            for i in np.flatnonzero(mask):
                episodes.append(
                    {
                        "score": float(final["score"][i]),
                        "frames": float(final["frames"][i]),
                        "dex": float(final["dex_caught"][i]),
                        "caught": float(final["caught_in_session"][i]),
                        "visits": float(final["saucer_visits"][i]),
                        "catch_entries": float(final["catch_entries"][i]),
                        "slots_entered": float(final["slots_entered"][i]),
                        "alley_shots": float(final["alley_shots"][i]),
                        "arms": float(final["arms"][i]),
                    }
                )
    envs.close()

    e = episodes[: args.episodes]
    get = lambda k: np.asarray([x[k] for x in e])
    return {
        "checkpoint": checkpoint,
        "n": len(e),
        "frames_mean": get("frames").mean(),
        "frames_max": get("frames").max(),
        "score_mean": get("score").mean(),
        "score_median": np.median(get("score")),
        "score_max": get("score").max(),
        "dex_mean": get("dex").mean(),
        # Per-episode variance here is large (sd ~ 1 catch on a mean of ~1), so a raw
        # difference of 0.2 between checkpoints is well inside the noise. Always read the
        # standard error before claiming one policy beats another.
        "dex_stderr": get("dex").std(ddof=1) / np.sqrt(len(e)),
        "frames_stderr": get("frames").std(ddof=1) / np.sqrt(len(e)),
        "dex_hit_rate": float((get("dex") >= 1).mean()) * 100.0,
        "caught_mean": get("caught").mean(),
        # The bottleneck, and now the headline. dex/game is a floor-limited, high-variance
        # function of catch-mode entries -- which is why the random dex baseline wandered
        # 0.67/0.88/0.94 across the ledger.
        "visits_mean": get("visits").mean(),
        "visits_stderr": get("visits").std(ddof=1) / np.sqrt(len(e)),
        "entries_mean": get("catch_entries").mean(),
        "entries_stderr": get("catch_entries").std(ddof=1) / np.sqrt(len(e)),
        "slots_entered_mean": get("slots_entered").mean(),
        # Per-episode counts are confounded by episode length, and that confound has eaten
        # three experiments: every policy this project has trained buys survival time, which
        # buys more chances at the saucer without ever raising the chance per frame. The rate
        # is the honest version. Ratio of means, not mean of ratios -- short episodes would
        # otherwise dominate.
        # Score per frame, for the same reason: score/game rises with survival on its own.
        # Random measures 9,946,459 per 10k frames; an MPC planner gets 29,426,333 (3.0x).
        "score_rate": get("score").sum() / get("frames").sum() * 10_000.0,
        # Kept per-episode so score/10k can be bootstrapped. Pinball scores are heavy-tailed
        # -- the mean runs ~4x the median, and a single jackpot episode can move a ratio of
        # sums by tens of percent -- so a point estimate on this metric means nothing without
        # a resampled interval.
        "_scores": get("score"),
        "_frames": get("frames"),
        # Kept per-episode for the same reason as score: every rate in this table is a ratio
        # of sums, and the survival confound has faked three results in this project. A
        # per-episode standard error on a COUNT answers "does this policy do X more often per
        # game", which is not the question -- a policy that merely survives 16% longer scores
        # +4 sigma on that. These arrays let the per-10k-frame rate carry a resampled interval.
        "_alley": get("alley_shots"),
        "_arms": get("arms"),
        "_entries": get("catch_entries"),
        "_dex": get("dex"),
        "visit_rate": get("visits").sum() / get("frames").sum() * 10_000.0,
        "entry_rate": get("catch_entries").sum() / get("frames").sum() * 10_000.0,
        # The arm gate is the ceiling on everything downstream, and it is a count of ramp
        # shots, so it is the honest deciding metric for anything aimed at the dex objective.
        "alley_mean": get("alley_shots").mean(),
        "alley_stderr": get("alley_shots").std(ddof=1) / np.sqrt(len(e)),
        "alley_rate": get("alley_shots").sum() / get("frames").sum() * 10_000.0,
        "arms_mean": get("arms").mean(),
        "arms_stderr": get("arms").std(ddof=1) / np.sqrt(len(e)),
        "arms_rate": get("arms").sum() / get("frames").sum() * 10_000.0,
    }


# --- results file ------------------------------------------------------------
# `--out` writes one JSON envelope shared with goldeneye/harness/eval.py (which
# writes the same shape under GE_EVAL_OUT). Deliberately duplicated rather than
# factored into a package: two small writers, one documented format.
#
#   {"project": str, "checkpoint": str, "n": int,
#    "metrics": {<name>: {"value": float, "stderr": float|null, "n": int}},
#    "comparison": {...}|null,   # voltorb's bootstrap; null when there is none
#    "episodes": {<array name>: [...]},
#    "params": {...},            # what configured the run
#    "created_at": "...Z"}
#
# Extra top-level keys are project-specific: voltorb adds "checkpoints" (the
# same metrics/episodes for every arm, since one invocation evaluates several).

# The per-episode arrays evaluate() keeps under a leading underscore, and the
# names they get on disk. These are the whole point of the file: they make the
# bootstrap below re-runnable by anything downstream.
_ARRAYS = {
    "_scores": "scores",
    "_frames": "frames",
    "_alley": "alley_shots",
    "_arms": "arms",
    "_entries": "catch_entries",
    "_dex": "dex",
}


def _f(x):
    return float(x)


def _metrics(row: dict) -> dict:
    """Flatten a row into name -> {value, stderr, n}, pairing `*_mean` with `*_stderr`."""
    n = int(row["n"])
    out = {}
    for k, v in row.items():
        if k in ("checkpoint", "n") or k.startswith("_") or k.endswith("_stderr"):
            continue
        se = row.get(k[: -len("_mean")] + "_stderr") if k.endswith("_mean") else None
        out[k] = {"value": _f(v), "stderr": None if se is None else _f(se), "n": n}
    return out


def _episodes(row: dict) -> dict:
    return {name: [_f(x) for x in row[key]] for key, name in _ARRAYS.items() if key in row}


def _rom_sha256(path: str):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def write_results(path, rows: list, args, comparison) -> None:
    # The envelope needs one subject; the candidate is the interesting arm and
    # `random` is the baseline it is compared against. Every arm is still
    # written in full under "checkpoints", so nothing is lost.
    primary = next((r for r in rows if r["checkpoint"] != "random"), rows[0])
    payload = {
        "project": "voltorb",
        "checkpoint": primary["checkpoint"],
        "n": int(primary["n"]),
        "metrics": _metrics(primary),
        "comparison": comparison,
        "episodes": _episodes(primary),
        "params": {
            "stage": args.stage,
            "episodes": args.episodes,
            "seed": args.seed,
            "frame_skip": args.frame_skip,
            "num_envs": args.num_envs,
            "rom": args.rom,
            "rom_sha256": _rom_sha256(args.rom),
            "checkpoints": list(args.checkpoint),
        },
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoints": {
            r["checkpoint"]: {
                "n": int(r["n"]),
                "metrics": _metrics(r),
                "episodes": _episodes(r),
            }
            for r in rows
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path, or 'random' for the baseline; repeatable")
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--stage", default="dex",
                    choices=["survive", "score", "dex", "saucer", "catch", "shot", "alley"],
                    help="only affects reward bookkeeping, not the reported metrics")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", help="write the results envelope here as JSON")
    args = ap.parse_args()

    rows = [evaluate(c, args) for c in args.checkpoint]

    short = lambda n: n if len(n) <= 26 else "..." + n[-23:]

    print(
        f"\n{'checkpoint':26s} {'n':>3s} {'frames +- se':>19s} "
        f"{'visits':>7s} {'catchmd +- se':>15s} "
        f"{'score/10k':>12s} {'vis/10k':>8s} {'ent/10k':>8s} {'ramp/10k':>9s} {'arm/10k':>8s} "
        f"{'dex/game +- se':>17s} {'dex>=1':>7s}"
    )
    for r in rows:
        print(
            f"{short(r['checkpoint']):26s} {r['n']:3d} "
            f"{r['frames_mean']:11,.0f} +-{r['frames_stderr']:6,.0f} "
            f"{r['visits_mean']:7.2f} "
            f"{r['entries_mean']:9.2f} +-{r['entries_stderr']:4.2f} "
            f"{r['score_rate']:12,.0f} {r['visit_rate']:8.2f} {r['entry_rate']:8.2f} "
            f"{r['alley_rate']:9.2f} {r['arms_rate']:8.2f} "
            f"{r['dex_mean']:10.2f} +-{r['dex_stderr']:5.2f} "
            f"{r['dex_hit_rate']:6.0f}%"
        )

    comparison = None
    base = next((r for r in rows if r["checkpoint"] == "random"), None)
    if base and len(rows) > 1:
        # Collected as it prints, not recomputed: the bootstrap draws from one
        # rng in print order, so a second pass would not reproduce these numbers.
        comparison = {
            "baseline": base["checkpoint"],
            "method": "bootstrap ratio of sums, resampling episodes",
            "n_boot": 4000,
            "seed": 0,
            "ci": 95,
            "metrics": {},
            "per_episode_mean": {},
        }
        rng = np.random.default_rng(0)
        n_boot = 4000

        def boot_rate(r, key="_scores"):
            # Resample episodes, not frames: the numerator and denominator have to move
            # together or the interval is meaningless.
            num, fr = r[key], r["_frames"]
            idx = rng.integers(0, len(num), size=(n_boot, len(num)))
            return num[idx].sum(1) / fr[idx].sum(1) * 10_000.0

        for metric, label, key, fmt in (
            ("score_rate", "score per 10k frames", "_scores", ",.0f"),
            ("alley_rate", "ramp shots per 10k frames", "_alley", ".3f"),
            ("arms_rate", "arms per 10k frames", "_arms", ".3f"),
            ("entry_rate", "catch entries per 10k frames", "_entries", ".3f"),
            ("dex_rate", "dex per 10k frames", "_dex", ".3f"),
        ):
            b_base = boot_rate(base, key)
            print(f"\nvs random ({label}, bootstrapped over episodes):")
            comparison["metrics"][metric] = {"label": label, "checkpoints": {}}
            for r in rows:
                if r is base:
                    continue
                diff = boot_rate(r, key) - b_base
                lo, hi = np.percentile(diff, [2.5, 97.5])
                z = diff.mean() / diff.std() if diff.std() else 0.0
                verdict = "significant" if lo > 0 or hi < 0 else "NOT distinguishable from random"
                comparison["metrics"][metric]["checkpoints"][r["checkpoint"]] = {
                    "mean": float(diff.mean()),
                    "lo": float(lo),
                    "hi": float(hi),
                    "sigma": float(z),
                    "significant": bool(lo > 0 or hi < 0),
                }
                print(f"  {short(r['checkpoint']):26s} {diff.mean():+{fmt}} "
                      f"[95% CI {lo:+{fmt}}, {hi:+{fmt}}] = {z:+.1f} sigma  ({verdict})")
        # Per-EPISODE counts below. These are reported only because the ledger's older entries
        # use them; they are confounded by episode length and the rates above supersede them.
        for label, mean_key, se_key in (
            ("alley_shots/ep", "alley_mean", "alley_stderr"),
            ("arms/ep", "arms_mean", "arms_stderr"),
            ("catch_entries/ep", "entries_mean", "entries_stderr"),
            ("saucer_visits/ep", "visits_mean", "visits_stderr"),
            ("dex/game", "dex_mean", "dex_stderr"),
        ):
            print(f"\nvs random ({label}, in standard errors):")
            comparison["per_episode_mean"][mean_key] = {"label": label, "checkpoints": {}}
            for r in rows:
                if r is base:
                    continue
                diff = r[mean_key] - base[mean_key]
                se = float(np.hypot(r[se_key], base[se_key]))
                sigma = diff / se if se else 0.0
                verdict = "significant" if abs(sigma) >= 2 else "NOT distinguishable from random"
                comparison["per_episode_mean"][mean_key]["checkpoints"][r["checkpoint"]] = {
                    "mean": float(diff),
                    "stderr": se,
                    "sigma": float(sigma),
                    "significant": bool(abs(sigma) >= 2),
                }
                print(f"  {short(r['checkpoint']):26s} "
                      f"{diff:+.2f} = {sigma:+.1f} sigma  ({verdict})")

    if args.out:
        write_results(args.out, rows, args, comparison)


if __name__ == "__main__":
    main()
