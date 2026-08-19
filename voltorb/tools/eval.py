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

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from voltorb.train.ppo import ActorCritic, _env_thunk, load_checkpoint


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
        model.load_state_dict(load_checkpoint(checkpoint)["model"])
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
        "visit_rate": get("visits").sum() / get("frames").sum() * 10_000.0,
        "entry_rate": get("catch_entries").sum() / get("frames").sum() * 10_000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", required=True,
                    help="path, or 'random' for the baseline; repeatable")
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--stage", default="dex",
                    choices=["survive", "score", "dex", "saucer", "catch", "shot"],
                    help="only affects reward bookkeeping, not the reported metrics")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [evaluate(c, args) for c in args.checkpoint]

    short = lambda n: n if len(n) <= 26 else "..." + n[-23:]

    print(
        f"\n{'checkpoint':26s} {'n':>3s} {'frames +- se':>19s} "
        f"{'visits':>7s} {'catchmd +- se':>15s} "
        f"{'score/10k':>12s} {'vis/10k':>8s} {'ent/10k':>8s} "
        f"{'dex/game +- se':>17s} {'dex>=1':>7s}"
    )
    for r in rows:
        print(
            f"{short(r['checkpoint']):26s} {r['n']:3d} "
            f"{r['frames_mean']:11,.0f} +-{r['frames_stderr']:6,.0f} "
            f"{r['visits_mean']:7.2f} "
            f"{r['entries_mean']:9.2f} +-{r['entries_stderr']:4.2f} "
            f"{r['score_rate']:12,.0f} {r['visit_rate']:8.2f} {r['entry_rate']:8.2f} "
            f"{r['dex_mean']:10.2f} +-{r['dex_stderr']:5.2f} "
            f"{r['dex_hit_rate']:6.0f}%"
        )

    base = next((r for r in rows if r["checkpoint"] == "random"), None)
    if base and len(rows) > 1:
        # catch_entries first: it is what the project is now judged on.
        for label, mean_key, se_key in (
            ("catch_entries/ep", "entries_mean", "entries_stderr"),
            ("saucer_visits/ep", "visits_mean", "visits_stderr"),
            ("dex/game", "dex_mean", "dex_stderr"),
        ):
            print(f"\nvs random ({label}, in standard errors):")
            for r in rows:
                if r is base:
                    continue
                diff = r[mean_key] - base[mean_key]
                se = float(np.hypot(r[se_key], base[se_key]))
                sigma = diff / se if se else 0.0
                verdict = "significant" if abs(sigma) >= 2 else "NOT distinguishable from random"
                print(f"  {short(r['checkpoint']):26s} "
                      f"{diff:+.2f} = {sigma:+.1f} sigma  ({verdict})")


if __name__ == "__main__":
    main()
