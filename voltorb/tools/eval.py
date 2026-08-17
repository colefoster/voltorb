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
        "dex_hit_rate": float((get("dex") >= 1).mean()) * 100.0,
        "caught_mean": get("caught").mean(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path, or 'random' for the baseline; repeatable")
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--stage", default="dex", choices=["survive", "score", "dex"],
                    help="only affects reward bookkeeping, not the reported metrics")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [evaluate(c, args) for c in args.checkpoint]

    print(
        f"\n{'checkpoint':34s} {'n':>3s} {'frames':>9s} {'score mean':>13s} "
        f"{'score med':>13s} {'dex/game':>9s} {'dex>=1 %':>9s} {'caught':>7s}"
    )
    for r in rows:
        name = r["checkpoint"]
        name = name if len(name) <= 34 else "..." + name[-31:]
        print(
            f"{name:34s} {r['n']:3d} {r['frames_mean']:9,.0f} {r['score_mean']:13,.0f} "
            f"{r['score_median']:13,.0f} {r['dex_mean']:9.2f} {r['dex_hit_rate']:8.0f}% "
            f"{r['caught_mean']:7.2f}"
        )


if __name__ == "__main__":
    main()
