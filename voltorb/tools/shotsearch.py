"""Is the saucer shot controllable, or is it luck? Answer it without a human and without PPO.

Three trained runs and thirteen hand-written policies all failed to raise saucer visits per
10,000 frames above the ~1.2 a random policy gets. That leaves two possibilities that no
further training can distinguish: the shot is aimable and PPO cannot find it, or the shot is
not aimable at frame-level flipper granularity.

This settles it with savestates instead of learning. From one fixed game state, run K random
action sequences forward. Then take the sequences that reached the saucer, keep only their
first `prefix` actions, and replay those with *fresh* randomness afterwards.

  * If the prefix replays hit far more often than the base rate, the early actions caused the
    hit. The shot is controllable and this is a credit-assignment problem.
  * If the prefix replays fall back to the base rate, the hit came from downstream chance. The
    saucer is luck given time on the table, and no reward shaping will move it.

**The baseline must be state-matched.** Prefix retests only exist for states that produced a
winner, and those states are favourable by selection -- some states have the ball nowhere near
the saucer and hit 0/32 no matter what. Comparing prefix replays against the base rate over
*all* states manufactures a +2.0 sigma "result" out of that selection alone; against the same
states it was -0.6 sigma. Only the state-matched number is reported.

Sweeping the prefix length shows *when* the outcome is decided. A prefix as long as the
horizon is a deterministic replay and reproduces trivially, so the question is where between
0 and the horizon the reproduction rate rises: early saturation means an early decision
controls the shot; a rise only near the horizon means the outcome is settled late, by ball
physics rather than by a choosable action.

    uv run python -m voltorb.tools.shotsearch --states 24 --rollouts 32 --prefix 30,90,200
"""
from __future__ import annotations

import argparse
import io

import numpy as np

from voltorb.env import EnvConfig, PinballEnv
from voltorb.env.pinball_env import SAUCER_X, SAUCER_Y


def _visited(env: PinballEnv, actions, horizon: int, rng) -> bool:
    """Roll forward and report whether the ball settled in the saucer. `actions` is a prefix
    replayed verbatim; anything past it is drawn from `rng`."""
    dwell = 0
    for i in range(horizon):
        a = actions[i] if i < len(actions) else int(rng.integers(0, 4))
        env.step(a)
        mem = env.pyboy.memory
        x = (mem[0xD4B3] | (mem[0xD4B4] << 8)) / 256.0
        y = (mem[0xD4B5] | (mem[0xD4B6] << 8)) / 256.0
        if np.hypot(x - SAUCER_X, y - SAUCER_Y) < 2.0 and mem[0xD54B] == 0:
            dwell += 1
            if dwell >= 5:
                return True
        else:
            dwell = 0
    return False


def _restore(env: PinballEnv, snap: io.BytesIO) -> None:
    snap.seek(0)
    env.pyboy.load_state(snap)
    env.pyboy.button_release("left")
    env.pyboy.button_release("a")
    env._held = 0
    env._launched = True
    env._frames = 0
    env._saucer_dwell = 0
    env._saucer_counted = False
    env._catch_idle_frames = env.CATCH_ENTRY_GAP_FRAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--states", type=int, default=24, help="game states to branch from")
    ap.add_argument("--rollouts", type=int, default=32, help="random continuations per state")
    ap.add_argument("--horizon", type=int, default=400, help="frames per rollout")
    ap.add_argument("--prefix", default="30,90,200",
                    help="comma-separated prefix lengths to sweep")
    ap.add_argument("--retests", type=int, default=16, help="fresh continuations per prefix")
    ap.add_argument("--gap", type=int, default=500, help="frames between sampled states")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env = PinballEnv(EnvConfig(rom_path=args.rom, stage="saucer"))
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    prefixes = [int(v) for v in str(args.prefix).split(",")]
    base_hits = base_total = 0
    # State-matched accounting: only states that produced at least one winner contribute, and
    # they contribute to BOTH sides of the comparison.
    matched_base_hits = matched_base_total = 0
    prefix_hits = {p: 0 for p in prefixes}
    prefix_total = {p: 0 for p in prefixes}
    states_with_a_hit = 0
    per_state = []

    for s in range(args.states):
        # Advance to a fresh state that is actually a shot opportunity: ball in play on the
        # screen that has the saucer, and the saucer ready. Only ~17% of frames qualify
        # (stage 0 is 31% of frames, ready is ~56%), so seek rather than skip.
        for i in range(args.gap * 20):
            _, _, term, trunc, _ = env.step(int(rng.integers(0, 4)))
            if term or trunc:
                env.reset(seed=args.seed + 1000 + s)
            if i >= args.gap and env.gw.current_stage == 0 and env.pyboy.memory[0xD532] == 128:
                break
        else:
            continue
        snap = io.BytesIO()
        env.pyboy.save_state(snap)

        winners, hits = [], 0
        for k in range(args.rollouts):
            _restore(env, snap)
            roll = np.random.default_rng(args.seed * 100003 + s * 997 + k)
            actions = [int(roll.integers(0, 4)) for _ in range(args.horizon)]
            replay = np.random.default_rng(0)  # unused: actions are fully specified
            if _visited(env, actions, args.horizon, replay):
                hits += 1
                winners.append(actions)
        base_hits += hits
        base_total += args.rollouts
        states_with_a_hit += bool(winners)
        if not winners:
            print(f"  state {s:3d}: 0/{args.rollouts} random rollouts hit", flush=True)
            continue
        matched_base_hits += hits
        matched_base_total += args.rollouts

        # Replay each winning prefix with fresh randomness after it, at each prefix length.
        row = {}
        for pfx in prefixes:
            p_hits = p_total = 0
            for w_i, w in enumerate(winners[:4]):
                for r in range(args.retests):
                    _restore(env, snap)
                    fresh = np.random.default_rng(
                        args.seed * 7919 + s * 31 + w_i * 17 + r + pfx * 104729
                    )
                    p_hits += _visited(env, w[:pfx], args.horizon, fresh)
                    p_total += 1
            prefix_hits[pfx] += p_hits
            prefix_total[pfx] += p_total
            row[pfx] = (p_hits, p_total)
        per_state.append((s, hits, row))
        detail = "  ".join(f"p{p}={row[p][0]}/{row[p][1]}" for p in prefixes)
        print(f"  state {s:3d}: {hits:2d}/{args.rollouts} random rollouts hit   {detail}",
              flush=True)

    env.close()
    base = base_hits / max(base_total, 1)
    matched = matched_base_hits / max(matched_base_total, 1)
    print(f"\nstates branched from          {states_with_a_hit + (base_total // args.rollouts - states_with_a_hit)}")
    print(f"states where any rollout hit  {states_with_a_hit}")
    print(f"base rate, all states         {base_hits}/{base_total} = {base:.3f}")
    print(f"base rate, matched states     {matched_base_hits}/{matched_base_total} = {matched:.3f}"
          "   <- the only valid baseline")
    print(f"\n{'prefix':>7s} {'frames':>7s} {'retests':>16s} {'rate':>7s} {'vs matched':>12s}")
    for pfx in prefixes:
        h, t = prefix_hits[pfx], prefix_total[pfx]
        if not t:
            continue
        rate = h / t
        p = (matched_base_hits + h) / (matched_base_total + t)
        se = np.sqrt(p * (1 - p) * (1 / matched_base_total + 1 / t)) if matched_base_total else 0
        z = (rate - matched) / se if se else 0.0
        print(f"{pfx:7d} {pfx:7d} {h:9d}/{t:<6d} {rate:7.3f} {z:+11.1f}s")
    print("\nA prefix equal to the horizon is a deterministic replay and reproduces trivially.")
    print("Reproduction that only appears at long prefixes means the outcome is settled late,")
    print("by ball physics rather than by a choosable action.")


if __name__ == "__main__":
    main()
