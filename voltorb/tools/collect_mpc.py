"""Distil the MPC planner: record (observation, action executed) from planner play.

This is the one imitation variant the project has NOT tried, and the distinction matters.
`tools/collect_demos.py` + `train/bc.py` (runs/bc-01.pt, bc-02.pt) fit advantage-weighted
targets drawn from *random search*, and that failed for a measured reason: many different
action sequences reach the saucer from a given state, so P(action | obs, win) is nearly
uniform and the fit recovered 0.007 nats over uniform. The targets were multimodal.

The planner does not have that problem. At every frame it has committed to exactly one action,
so the label is a point mass. If a policy can be fit to those labels and reproduce the planner's
saucer rate, the shot is learnable from the 34-float observation and every null so far was the
learner. If it cannot -- if the fit is accurate on held-out frames but the rollout saucer rate
collapses to random's -- then the observation underdetermines the shot, and that is the argument
for pixels or explicit target geometry, bought for one afternoon instead of another 20M steps.

    uv run python -m voltorb.tools.collect_mpc --episodes 4 --out data/mpc_00.npz
    uv run python -m voltorb.train.bc --demos data/mpc_demos.npz --out runs/bc-mpc.pt
"""
from __future__ import annotations

import argparse
import io

import numpy as np

from voltorb.env import EnvConfig, PinballEnv


def collect_episode(env, args, seed, obs_buf, act_buf, stats) -> dict:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    from voltorb.tools.mpc import _points_rollout, _score_rollout

    plan: list[int] = []
    keep_plan = False
    while True:
        if not plan:
            snap = io.BytesIO()
            env.pyboy.save_state(snap)
            # Every counter the planner's rollouts mutate has to be restored, or the planning
            # frames leak into the episode's own statistics. Same list as tools/mpc.py.
            held, launched = env._held, env._launched
            visits, dwell, counted = env._saucer_visits, env._saucer_dwell, env._saucer_counted
            idle, entries = env._catch_idle_frames, env._catch_entries
            alley, arms = env._alley_shots, env._arms
            prev_alley, prev_ready = env._prev_alley_count, env._prev_ready
            frames0, prev_xy = env._frames, env._prev_xy
            best_score, best_actions = -1e18, None
            scorer = _points_rollout if args.objective == "score" else _score_rollout
            for _ in range(args.rollouts):
                actions = [int(rng.integers(0, 4)) for _ in range(args.horizon)]
                score = scorer(env, snap, actions, args.horizon)
                if score > best_score:
                    best_score, best_actions = score, actions
            snap.seek(0)
            env.pyboy.load_state(snap)
            env.pyboy.button_release("left")
            env.pyboy.button_release("a")
            env._held, env._launched = held, launched
            env._saucer_visits, env._saucer_dwell, env._saucer_counted = visits, dwell, counted
            env._catch_idle_frames, env._catch_entries = idle, entries
            env._alley_shots, env._arms = alley, arms
            env._prev_alley_count, env._prev_ready = prev_alley, prev_ready
            env._frames, env._prev_xy = frames0, prev_xy
            plan = best_actions[: args.replan]
            stats["decisions"] += 1
            informative = best_score > (0.0 if args.objective == "score" else 500.0)
            stats["with_hit"] += informative
            # THE labelling decision, and the smoke test made it obvious. The planner picks the
            # best of N *random* sequences, so on a decision where no rollout reached the
            # saucer it is only avoiding the drain and its choice among the rest is arbitrary.
            # Keeping those frames reproduces exactly the failure that killed bc-01/bc-02: the
            # first run of this collector logged an action histogram of [741, 749, 748, 762] --
            # uniform -- because 46 of 50 decisions had nothing to discriminate. Keep only the
            # decisions where the planner actually saw the shot.
            keep_plan = informative or not args.only_informative

        action = plan.pop(0)
        # Label the state the action was actually taken in, before stepping. Every executed
        # frame is kept, not just the replan boundary: the planner commits to the whole prefix,
        # and one label per 60 frames is not enough data to fit anything.
        if keep_plan:
            obs_buf.append(env._observe(env._raw_state()))
            act_buf.append(action)
        _, _, term, trunc, info = env.step(action)
        if term or trunc:
            return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--replan", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--objective", default="saucer", choices=["saucer", "score"])
    ap.add_argument("--out", default="data/mpc_demos.npz")
    ap.add_argument("--only-informative", action="store_true", default=True,
                    help="keep only frames from decisions whose best rollout reached the goal")
    ap.add_argument("--all-frames", dest="only_informative", action="store_false")
    args = ap.parse_args()

    env = PinballEnv(EnvConfig(rom_path=args.rom, stage="saucer", max_frames=args.max_frames))
    obs_buf: list = []
    act_buf: list = []
    stats = {"decisions": 0, "with_hit": 0}
    rows = []
    for ep in range(args.episodes):
        info = collect_episode(env, args, args.seed + ep, obs_buf, act_buf, stats)
        rows.append(info)
        print(f"  ep {ep}: frames {info['frames']:,} visits {info['saucer_visits']} "
              f"arms {info['arms']} dex {info['dex_caught']} samples {len(act_buf):,}", flush=True)
    env.close()

    obs = np.asarray(obs_buf, dtype=np.float32)
    act = np.asarray(act_buf, dtype=np.int64)
    # Uniform weights: bc.py is advantage-weighted for the search data, and there is no
    # advantage here -- the planner's action IS the target.
    weight = np.ones(len(act), dtype=np.float32)
    np.savez_compressed(args.out, obs=obs, act=act, weight=weight)

    f = sum(r["frames"] for r in rows)
    v = sum(r["saucer_visits"] for r in rows)
    print(f"\n{len(act):,} samples -> {args.out}")
    print(f"planner: {f:,} frames, {v} saucer visits = {v / f * 10_000:.2f} per 10k "
          f"(random 1.17), {stats['with_hit']}/{stats['decisions']} decisions had a hit rollout")
    print("action histogram:", np.bincount(act, minlength=4).tolist())


if __name__ == "__main__":
    main()
