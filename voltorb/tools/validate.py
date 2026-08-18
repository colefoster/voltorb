"""Validation harness. Run this before trusting anything the env reports.

Answers, empirically:
  1. Does every observation field actually move, and is its scale sane?
  2. Are the wrapper's ball velocities signed, or do we need position deltas?
  3. How many env steps/sec do we get at 1..14 parallel processes?
  4. How long is a ball, and a game? (Undocumented anywhere — sets PPO rollout length.)
  5. Does reset() really hand back a fresh, empty Pokedex?

Usage: uv run python -m voltorb.tools.validate [--frames N] [--games N]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from voltorb.env import OBS_FIELDS, EnvConfig, PinballEnv

ROM = "roms/pokemon_pinball.gbc"

# Fields that legitimately sit at their bound a lot (flags, full meters). Hitting 1.0 here
# is correct, not a scaling bug, so they are exempt from the saturation warning.
BOUNDED_FIELDS = frozenset(
    {
        "special_mode_active",
        "timer_active",
        "rare_pokemon_flag",
        "target_already_caught",
        "pikachu_saver_charge",
        "saver_seconds_left",
        "catch_tiles_flipped",
    }
)


def _make_env(stage: str = "dex", frame_skip: int = 1) -> PinballEnv:
    return PinballEnv(EnvConfig(rom_path=ROM, frame_skip=frame_skip, stage=stage))


# ---- 1 & 2: field coverage and velocity signedness -----------------------------------


def check_fields(frames: int, seed: int = 0) -> None:
    print(f"\n=== field coverage over {frames:,} frames (random policy) ===")
    env = _make_env()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset()

    raws: list[dict[str, float]] = []
    obses = [obs]
    for _ in range(frames):
        obs, _, terminated, truncated, _ = env.step(rng.integers(0, 4))
        raws.append(env._raw_state())
        obses.append(obs)
        if terminated or truncated:
            obs, _ = env.reset()

    arr = np.asarray(obses, dtype=np.float32)
    print(f"{'field':26s} {'raw min':>10s} {'raw max':>10s} {'obs min':>8s} {'obs max':>8s} {'%clip':>6s}  note")
    for i, name in enumerate(OBS_FIELDS):
        rawvals = np.asarray([r[name] for r in raws], dtype=np.float64)
        col = arr[:, i]
        clipped = float(np.mean((np.abs(col) >= 0.999))) * 100.0
        notes = []
        if rawvals.min() == rawvals.max():
            notes.append("DEAD (constant)")
        if clipped > 5.0 and name not in BOUNDED_FIELDS and not name.startswith("tile_"):
            notes.append("SATURATED - rescale")
        if rawvals.min() < 0:
            notes.append("signed")
        print(
            f"{name:26s} {rawvals.min():10.1f} {rawvals.max():10.1f} "
            f"{col.min():8.3f} {col.max():8.3f} {clipped:6.1f}  {', '.join(notes)}"
        )

    # Velocity signedness: if the wrapper never reports a negative, direction is lost and
    # the derived position deltas are the only usable velocity signal.
    vx = np.asarray([r["ball_vx_raw"] for r in raws])
    vy = np.asarray([r["ball_vy_raw"] for r in raws])
    dx = np.asarray([r["ball_dx"] for r in raws])
    dy = np.asarray([r["ball_dy"] for r in raws])
    print(
        f"\nvelocity signedness: vx_raw<0 in {np.mean(vx < 0) * 100:.1f}% of frames, "
        f"vy_raw<0 in {np.mean(vy < 0) * 100:.1f}%"
    )
    print(
        f"derived deltas:      dx<0 in {np.mean(dx < 0) * 100:.1f}%, dy<0 in {np.mean(dy < 0) * 100:.1f}%"
    )
    moving = (np.abs(dx) > 0) | (np.abs(dy) > 0)
    print(f"ball moved on {np.mean(moving) * 100:.1f}% of frames")
    if np.any(vx < 0) or np.any(vy < 0):
        print("  -> wrapper velocities ARE signed; keep them")
    else:
        print("  -> wrapper velocities look UNSIGNED; rely on ball_dx/ball_dy for direction")
    env.close()


# ---- 3: throughput -------------------------------------------------------------------


def _bench_worker(args) -> int:
    frames, seed = args
    env = _make_env()
    rng = np.random.default_rng(seed)
    env.reset()
    for _ in range(frames):
        _, _, terminated, truncated, _ = env.step(rng.integers(0, 4))
        if terminated or truncated:
            env.reset()
    env.close()
    return frames


def bench(frames_per_env: int, workers: tuple[int, ...]) -> None:
    print(f"\n=== throughput ({frames_per_env:,} steps per worker) ===")
    print(f"{'workers':>8s} {'steps/sec':>12s} {'per-worker':>12s} {'x realtime':>11s}")
    for n in workers:
        t0 = time.perf_counter()
        if n == 1:
            _bench_worker((frames_per_env, 0))
        else:
            with mp.get_context("spawn").Pool(n) as pool:
                pool.map(_bench_worker, [(frames_per_env, i) for i in range(n)])
        dt = time.perf_counter() - t0
        total = frames_per_env * n
        print(f"{n:8d} {total / dt:12,.0f} {total / dt / n:12,.0f} {total / dt / 60:11.0f}")


# ---- 4 & 5: episode structure and fresh-save guarantee -------------------------------


def measure_episodes(games: int, seed: int = 0) -> None:
    print(f"\n=== episode structure over {games} games (random policy) ===")
    env = _make_env()
    rng = np.random.default_rng(seed)

    game_frames, game_scores, game_caught, ball_frames = [], [], [], []
    dex_at_reset = []

    for _ in range(games):
        obs, _ = env.reset()
        dex_at_reset.append(int(round(env._raw_state()["dex_caught_frac"] * 151)))
        frames = 0
        last_ball_change = 0
        prev_balls = env.gw.balls_left
        while True:
            _, _, terminated, truncated, info = env.step(rng.integers(0, 4))
            frames += 1
            if env.gw.balls_left != prev_balls:
                ball_frames.append(frames - last_ball_change)
                last_ball_change = frames
                prev_balls = env.gw.balls_left
            if terminated or truncated:
                game_frames.append(frames)
                game_scores.append(info.get("score", 0))
                game_caught.append(info.get("dex_caught", 0))
                break

    def stats(name, vals, unit=""):
        if not vals:
            print(f"  {name:22s} no samples")
            return
        a = np.asarray(vals, dtype=np.float64)
        print(
            f"  {name:22s} n={len(a):4d}  min={a.min():10.1f}  median={np.median(a):10.1f}  "
            f"mean={a.mean():10.1f}  max={a.max():10.1f} {unit}"
        )

    stats("frames per game", game_frames, "frames")
    stats("seconds per game", [f / 60 for f in game_frames], "sec")
    stats("frames per ball", ball_frames, "frames")
    stats("score per game", game_scores)
    stats("dex caught per game", game_caught)
    print(f"  dex at reset (must be 0): {sorted(set(dex_at_reset))}")
    if set(dex_at_reset) != {0}:
        print("  !! FRESH-SAVE GUARANTEE BROKEN - Pokedex is leaking across episodes")
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=20_000)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--bench-frames", type=int, default=20_000)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    check_fields(args.frames)
    measure_episodes(args.games)
    if not args.skip_bench:
        bench(args.bench_frames, workers=(1, 4, 8, 10, 14))


if __name__ == "__main__":
    main()
