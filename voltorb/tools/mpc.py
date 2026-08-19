"""Use the search at decision time instead of trying to distill it into a policy.

The emulator is a perfect forward model and savestates make it resettable, so the shot can be
solved by planning rather than learning: snapshot the current state, roll K random futures,
execute the first `replan` actions of the best one, repeat.

This exists because distillation failed in a specific, measured way. tools/shotsearch.py showed
an open-loop winning sequence usually exists (a 90-frame prefix triples the hit rate, +9.0
sigma), but advantage-weighted behaviour cloning on that same search data recovers only 0.007
nats over a uniform policy -- because *many different* sequences reach the saucer from a given
state, so P(action | observation, win) is nearly uniform even though the joint sequence matters.
A per-frame Markov policy is the wrong object to fit; a planner is not.

What this measures is the headroom: if MPC clears random's 1.18 saucer visits per 10,000 frames
by a wide margin, the control shotsearch found is real and exploitable and the gap is purely the
learner. If even MPC cannot, that control is too thin to matter in practice.

    uv run python -m voltorb.tools.mpc --episodes 8 --rollouts 12 --horizon 240 --replan 60

Cost is roughly rollouts * horizon / replan times real time, so this is an evaluation
instrument, not a policy you would ship.
"""
from __future__ import annotations

import argparse
import io

import numpy as np

from voltorb.env import EnvConfig, PinballEnv
from voltorb.env.pinball_env import SAUCER_X, SAUCER_Y


def _points_rollout(env: PinballEnv, snap: io.BytesIO, actions, horizon: int) -> float:
    """Score a future by points gained, with losing the ball charged against it.

    Unlike the saucer, points are dense -- most rollouts differ -- so the planner has a usable
    signal at nearly every decision instead of the ~35% it gets aiming at the saucer.
    """
    snap.seek(0)
    env.pyboy.load_state(snap)
    env.pyboy.button_release("left")
    env.pyboy.button_release("a")
    env._held = 0
    env._launched = True
    start_score = env.gw.score
    prev_balls = env.gw.balls_left
    for i in range(horizon):
        env.step(actions[i])
        if env.gw.balls_left < prev_balls or env.gw.game_over:
            # A ball is worth ~6.7M to a random policy, so losing one dwarfs any 400-frame gain.
            return float(env.gw.score - start_score) - 2_000_000.0
    return float(env.gw.score - start_score)


def _score_rollout(env: PinballEnv, snap: io.BytesIO, actions, horizon: int) -> float:
    """Replay `actions` and score the future: a saucer visit is worth far more than anything
    else, and closest approach breaks ties so the planner still has a gradient when no rollout
    reaches the saucer. Losing the ball is scored below any surviving future."""
    snap.seek(0)
    env.pyboy.load_state(snap)
    env.pyboy.button_release("left")
    env.pyboy.button_release("a")
    env._held = 0
    env._launched = True
    prev_balls = env.gw.balls_left
    best_dist = 1e9
    dwell = 0
    for i in range(horizon):
        env.step(actions[i])
        mem = env.pyboy.memory
        x = (mem[0xD4B3] | (mem[0xD4B4] << 8)) / 256.0
        y = (mem[0xD4B5] | (mem[0xD4B6] << 8)) / 256.0
        ready = mem[0xD532] == 128
        if env.gw.current_stage == 0 and ready:
            best_dist = min(best_dist, float(np.hypot(x - SAUCER_X, y - SAUCER_Y)))
        if np.hypot(x - SAUCER_X, y - SAUCER_Y) < 2.0 and mem[0xD54B] == 0:
            dwell += 1
            if dwell >= 5:
                return 1000.0 - i * 0.01  # sooner is better
        else:
            dwell = 0
        if env.gw.balls_left < prev_balls or env.gw.game_over:
            return -100.0
    if best_dist > 1e8:
        return -1.0  # never on the saucer's screen with the saucer ready
    return -best_dist


def run_episode(env: PinballEnv, args, seed: int, stats: dict) -> dict:
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    frames = 0
    plan: list[int] = []
    while True:
        if not plan:
            snap = io.BytesIO()
            env.pyboy.save_state(snap)
            held, launched = env._held, env._launched
            visits, dwell, counted = env._saucer_visits, env._saucer_dwell, env._saucer_counted
            idle, entries = env._catch_idle_frames, env._catch_entries
            # _frames must be saved too: the rollouts run through env.step(), so without this
            # the planning frames count against the episode's own budget and it truncates after
            # two decisions -- which also poisons the visits-per-10k denominator.
            frames0, prev_xy = env._frames, env._prev_xy
            best_score, best_actions = -1e18, None
            for k in range(args.rollouts):
                actions = [int(rng.integers(0, 4)) for _ in range(args.horizon)]
                scorer = _points_rollout if args.objective == "score" else _score_rollout
                score = scorer(env, snap, actions, args.horizon)
                if score > best_score:
                    best_score, best_actions = score, actions
            # Put the real game back exactly as it was before planning.
            snap.seek(0)
            env.pyboy.load_state(snap)
            env.pyboy.button_release("left")
            env.pyboy.button_release("a")
            env._held, env._launched = held, launched
            env._saucer_visits, env._saucer_dwell, env._saucer_counted = visits, dwell, counted
            env._catch_idle_frames, env._catch_entries = idle, entries
            env._frames, env._prev_xy = frames0, prev_xy
            plan = best_actions[: args.replan]
            stats["decisions"] += 1
            if args.objective == "score":
                stats["with_hit"] += best_score > 0.0
                stats["blind"] += best_score <= 0.0
            else:
                stats["with_hit"] += best_score > 500.0
                stats["blind"] += best_score <= -99.0 or best_score == -1.0
        action = plan.pop(0)
        _, _, term, trunc, info = env.step(action)
        frames += 1
        if term or trunc:
            return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=240)
    ap.add_argument("--replan", type=int, default=60)
    ap.add_argument("--max-frames", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--objective", default="saucer", choices=["saucer", "score"],
                    help="what the planner optimises")
    ap.add_argument("--random-baseline", action="store_true",
                    help="same episode count with uniform actions, for a matched comparison")
    args = ap.parse_args()

    env = PinballEnv(EnvConfig(rom_path=args.rom, stage="saucer", max_frames=args.max_frames))
    rows = []
    # If almost no decision has a rollout that reaches the saucer, the planner is not planning
    # the shot at all -- it is only avoiding the drain, and the horizon is too short.
    stats = {"decisions": 0, "with_hit": 0, "blind": 0}
    for ep in range(args.episodes):
        if args.random_baseline:
            env.reset(seed=args.seed + ep)
            rng = np.random.default_rng(args.seed + ep)
            while True:
                _, _, term, trunc, info = env.step(int(rng.integers(0, 4)))
                if term or trunc:
                    break
        else:
            info = run_episode(env, args, args.seed + ep, stats)
        rows.append(info)
        print(f"  ep {ep}: frames {info['frames']:,} score {info['score']:,} "
              f"visits {info['saucer_visits']} dex {info['dex_caught']}", flush=True)
    env.close()

    f = sum(r["frames"] for r in rows)
    v = sum(r["saucer_visits"] for r in rows)
    c = sum(r["catch_entries"] for r in rows)
    d = sum(r["dex_caught"] for r in rows)
    k = 10_000.0 / f
    label = "random" if args.random_baseline else "MPC"
    print(f"\n{label}: {len(rows)} episodes, {f:,} frames")
    scores = np.asarray([r["score"] for r in rows], dtype=float)
    print(f"  visits {v} = {v * k:.2f}/10k   entries {c} = {c * k:.2f}/10k   "
          f"dex {d} = {d / len(rows):.2f}/game")
    # Score per frame, because score/game is confounded by survival exactly like dex/game was.
    print(f"  score median {np.median(scores):,.0f}  mean {scores.mean():,.0f}  "
          f"per 10k frames {scores.sum() * k:,.0f}")
    print("  reference: random 1.18 visits/10k, 0.59 entries/10k, ~0.88 dex/game")
    if stats["decisions"]:
        print(f"  planner: {stats['decisions']} decisions, "
              f"{stats['with_hit'] / stats['decisions'] * 100:.1f}% had a rollout that scored a "
              f"visit, {stats['blind'] / stats['decisions'] * 100:.1f}% had no usable signal")


if __name__ == "__main__":
    main()
