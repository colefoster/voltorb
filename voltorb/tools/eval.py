"""Evaluate checkpoints properly, because the training scalars are a rolling window.

Compares any number of checkpoints (and a random baseline) over the same number of
episodes, and reports the numbers the project is actually judged on:

    uv run python -m voltorb.tools.eval --episodes 20 \\
        --checkpoint random \\
        --checkpoint runs/survive-03/final.pt \\
        --checkpoint runs/score-01/final.pt

The done bar for the project is "at least one NEW species per game in at least 90% of
games", so `dex>=1 %` is the column that matters in the end.
"""
from __future__ import annotations

import argparse
import functools

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from voltorb.train.ppo import ActorCritic, _env_thunk


def evaluate(checkpoint: str, args) -> dict:
    envs = gym.vector.AsyncVectorEnv(
        [
            functools.partial(_env_thunk, args.rom, args.stage, args.frame_skip, args.seed + i)
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    model = None
    if checkpoint != "random":
        model = ActorCritic()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path, or 'random' for the baseline; repeatable")
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--stage", default="dex", choices=["survive", "score", "dex", "catch"],
                    help="only affects reward bookkeeping, not the reported metrics")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [evaluate(c, args) for c in args.checkpoint]

    print(
        f"\n{'checkpoint':30s} {'n':>3s} {'frames +- se':>20s} "
        f"{'score med':>13s} {'dex/game +- se':>18s} {'dex>=1 %':>9s}"
    )
    for r in rows:
        name = r["checkpoint"]
        name = name if len(name) <= 30 else "..." + name[-27:]
        print(
            f"{name:30s} {r['n']:3d} "
            f"{r['frames_mean']:12,.0f} +-{r['frames_stderr']:6,.0f} "
            f"{r['score_median']:13,.0f} "
            f"{r['dex_mean']:11.2f} +-{r['dex_stderr']:5.2f} "
            f"{r['dex_hit_rate']:8.0f}%"
        )

    base = next((r for r in rows if r["checkpoint"] == "random"), None)
    if base and len(rows) > 1:
        print("\nvs random (difference in dex/game, in standard errors):")
        for r in rows:
            if r is base:
                continue
            diff = r["dex_mean"] - base["dex_mean"]
            se = float(np.hypot(r["dex_stderr"], base["dex_stderr"]))
            sigma = diff / se if se else 0.0
            verdict = "significant" if abs(sigma) >= 2 else "NOT distinguishable from random"
            name = r["checkpoint"]
            print(f"  {name if len(name) <= 30 else '...' + name[-27:]:30s} "
                  f"{diff:+.2f} = {sigma:+.1f} sigma  ({verdict})")


if __name__ == "__main__":
    main()
