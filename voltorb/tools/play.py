"""Play the game yourself, scored on the metric the agent is judged on.

Three trained runs and thirteen hand-written policies have all failed to raise saucer visits
per 10,000 frames above the ~1.2 a uniform random policy gets. That leaves two very different
worlds, and no amount of further training distinguishes them:

  * A human gets clearly more than 1.2 visits/10k -> the shot IS aimable with two flippers,
    and this is a credit-assignment problem worth attacking with a different algorithm.
  * A human gets about 1.2 too -> catch mode is luck given time on the table, dex/game is
    survival x luck, and maximising frames (which every run already learns) is close to
    optimal for RAM + flippers. The objective needs a different lever, not a better learner.

Controls are PyBoy's own: arrow keys + A/S for A/B. Left arrow is the left flipper, A is the
right flipper and also launches the ball.

    uv run python -m voltorb.tools.play [--frames 60000]

Counting matches env/pinball_env.py exactly -- same saucer coordinate, radius, dwell and
catch-mode debounce -- so the number is comparable to tools/eval.py output.
"""
from __future__ import annotations

import argparse

from pyboy import PyBoy

from voltorb.env import pinball_env as pe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--frames", type=int, default=60_000, help="~16 min of game time")
    args = ap.parse_args()

    pyboy = PyBoy(args.rom, window="SDL2", sound_emulated=False)
    gw = pyboy.game_wrapper
    gw.start_game()
    for i in range(pe.N_SPECIES):
        pyboy.memory[pe.ADDR_POKEDEX + i] = 0

    frames = visits = entries = 0
    dwell = idle = 0
    counted = False
    idle = pe.PinballEnv.CATCH_ENTRY_GAP_FRAMES

    print("play. ctrl-c or close the window to stop.")
    try:
        for frames in range(1, args.frames + 1):
            if not pyboy.tick(1, True, True):
                break
            mem = pyboy.memory
            x = (mem[pe.ADDR_BALL_X] | (mem[pe.ADDR_BALL_X + 1] << 8)) / 256.0
            y = (mem[pe.ADDR_BALL_Y] | (mem[pe.ADDR_BALL_Y + 1] << 8)) / 256.0
            dist = ((x - pe.SAUCER_X) ** 2 + (y - pe.SAUCER_Y) ** 2) ** 0.5
            in_catch = mem[0xD54B] != 0 and mem[0xD550] == pe.SPECIAL_MODE_CATCH

            if in_catch:
                if idle >= pe.PinballEnv.CATCH_ENTRY_GAP_FRAMES:
                    entries += 1
                idle = 0
            else:
                idle += 1

            if dist < pe.PinballEnv.SAUCER_RADIUS and not in_catch:
                dwell += 1
                if dwell >= pe.PinballEnv.SAUCER_DWELL_FRAMES and not counted:
                    visits += 1
                    counted = True
                    print(f"  saucer visit {visits} at frame {frames:,}", flush=True)
            else:
                dwell = 0
                counted = False

            if gw.game_over:
                print(f"  game over at frame {frames:,}")
                break
    except KeyboardInterrupt:
        pass
    finally:
        dex = sum(1 for v in pyboy.memory[pe.ADDR_POKEDEX : pe.ADDR_POKEDEX + pe.N_SPECIES]
                  if v & 2)
        k = 10_000.0 / max(frames, 1)
        print(f"\nframes {frames:,}  visits {visits}  catch-mode entries {entries}  dex {dex}")
        print(f"visits/10k {visits * k:.2f}   entries/10k {entries * k:.2f}")
        print("random policy reference: visits/10k 1.23, entries/10k 0.55")
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
