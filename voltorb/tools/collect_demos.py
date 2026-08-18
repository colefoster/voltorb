"""Generate demonstrations for the saucer shot by search, with a per-state baseline.

tools/shotsearch.py established that an open-loop winning action sequence usually exists from a
given state (a 90-frame prefix reproduces at 0.379 against a state-matched base of 0.127) while
shot-01 established that PPO does not find one from experience. That is the classic case for
imitation: generate the behaviour by search, then fit a policy to it.

The naive version -- keep the trajectories that succeeded -- is weak here, because the actions
in them were drawn uniformly at random. What makes a random trajectory a winner is mostly the
state it started in, so filtering on success alone mostly relabels uniform actions as "good".

So this collects **contrastive** data instead: from each start state, run K rollouts, and keep
winners and losers from the SAME state. An action is labelled by how its rollout did relative to
that state's own success rate, which cancels the state's difficulty. That is advantage-weighted
regression with an exact per-state baseline, and it is what makes the weak per-sample signal
usable.

    uv run python -m voltorb.tools.collect_demos --states 60 --rollouts 16 --out data/demos.npz
"""
from __future__ import annotations

import argparse
import io

import numpy as np

from voltorb.env import OBS_DIM, EnvConfig, PinballEnv
from voltorb.env.pinball_env import SAUCER_X, SAUCER_Y


def _rollout(env: PinballEnv, snap: io.BytesIO, actions, horizon: int):
    """Replay `actions` from `snap`, returning (observations, hit)."""
    snap.seek(0)
    env.pyboy.load_state(snap)
    env.pyboy.button_release("left")
    env.pyboy.button_release("a")
    env._held = 0
    env._launched = True
    env._frames = 0
    obs_seq = []
    dwell = 0
    hit = False
    for i in range(horizon):
        raw = env._raw_state()
        obs_seq.append(env._observe(raw))
        env.step(actions[i])
        mem = env.pyboy.memory
        x = (mem[0xD4B3] | (mem[0xD4B4] << 8)) / 256.0
        y = (mem[0xD4B5] | (mem[0xD4B6] << 8)) / 256.0
        if ((x - SAUCER_X) ** 2 + (y - SAUCER_Y) ** 2) ** 0.5 < 2.0 and mem[0xD54B] == 0:
            dwell += 1
            if dwell >= 5:
                hit = True
                break
        else:
            dwell = 0
        if env.gw.game_over:
            break
    return obs_seq, hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--rollouts", type=int, default=16, help="rollouts per state")
    # 1,200 frames is where the random hit rate is 12.5%; at 300 it is ~2% and almost every
    # state comes back 0/K, which is no contrast and therefore no training signal.
    ap.add_argument("--horizon", type=int, default=1_200, help="frames scored per rollout")
    # Keep only the opening frames of each winning rollout. shotsearch measured the causal
    # content there: replaying a winner's first 90 frames triples the hit rate (+9.0 sigma),
    # while a whole 1,200-frame rollout is ~90 frames of decision and ~1,100 frames of ball
    # physics. Cloning all of it buries a real signal under a 12:1 dilution -- the first
    # attempt fit 81,015 whole-rollout samples and landed at val loss 1.3876 against a uniform
    # reference of 1.3863, i.e. nothing.
    ap.add_argument("--credit-frames", type=int, default=90,
                    help="frames kept from the start of each winning rollout; 0 = all")
    ap.add_argument("--gap", type=int, default=400, help="frames between sampled states")
    ap.add_argument("--out", default="data/demos.npz")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    env = PinballEnv(EnvConfig(rom_path=args.rom, stage="saucer"))
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    all_obs, all_act, all_w = [], [], []
    kept_states = 0
    for s in range(args.states):
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

        rollouts = []
        for k in range(args.rollouts):
            roll = np.random.default_rng(args.seed * 7717 + s * 131 + k)
            actions = [int(roll.integers(0, 4)) for _ in range(args.horizon)]
            obs_seq, hit = _rollout(env, snap, actions, args.horizon)
            rollouts.append((obs_seq, actions[: len(obs_seq)], hit))

        rate = sum(r[2] for r in rollouts) / len(rollouts)
        # A state where everything or nothing works carries no contrast, so it teaches nothing.
        if rate in (0.0, 1.0):
            continue
        kept_states += 1
        for obs_seq, actions, hit in rollouts:
            adv = (1.0 if hit else 0.0) - rate
            if adv <= 0:
                continue  # keep the winners, weighted by how surprising the win was
            keep = len(obs_seq) if not args.credit_frames else min(
                args.credit_frames, len(obs_seq)
            )
            all_obs.extend(obs_seq[:keep])
            all_act.extend(actions[:keep])
            all_w.extend([adv] * keep)
        print(f"  state {s:3d}: success {rate:.2f}, samples so far {len(all_obs):,}", flush=True)

    env.close()
    obs = np.asarray(all_obs, dtype=np.float32).reshape(-1, OBS_DIM)
    act = np.asarray(all_act, dtype=np.int64)
    w = np.asarray(all_w, dtype=np.float32)
    np.savez_compressed(args.out, obs=obs, act=act, weight=w)
    print(f"\nstates with contrast {kept_states}/{args.states}")
    print(f"saved {len(act):,} weighted samples -> {args.out}")
    if len(act):
        print("action histogram:", np.bincount(act, minlength=4).tolist())
        print("weight mean %.3f max %.3f" % (w.mean(), w.max()))


if __name__ == "__main__":
    main()
