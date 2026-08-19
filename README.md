# voltorb

An RL agent that plays **Pokémon Pinball** (GBC) to complete the Pokédex — not to maximize
score. Score is instrumental; catching species is the objective.

## Setup

```sh
uv sync
# supply your own ROM dump:
cp "Pokemon Pinball (USA, Australia) (Rumble Version)....gbc" roms/pokemon_pinball.gbc
uv run python -m voltorb.tools.validate --frames 40000 --games 3
```

Expected ROM: SHA1 `9402014d14969432142abfde728c6f1a10ee4dac` (matches pret/pokepinball's
byte-matching build). ROMs are gitignored and never committed.

## Design

| | |
|---|---|
| Observation | RAM only — 30 floats, no pixels (`OBS_FIELDS` in `env/pinball_env.py`) |
| Actions | 4 held states: `{none, left, right, both}`, re-declared every frame |
| Episode | One game (3 balls) from a fresh save, so the Pokédex starts empty |
| Curriculum | `survive` → `score` → `dex` → `saucer`, weights carried forward |
| Algorithm | PPO, single-file, ~8 vectorized envs |
| Headline metric | catch-mode entries **per 10,000 frames** (random: 0.55) |

The done bar used to be "≥1 new species per game in ≥90% of games". It is **retired**: a
uniform random policy clears it 72–78% of the time, so it measured the game's generosity
rather than the agent.

**The objective has almost no headroom, and that is the project's main finding.** A catch needs
the saucer to be armed (`0xD532 == 128`). It arms once at frame 0 and re-arms only **0.42 times
per episode** (measured over 12 episodes; 696–12,595 frames after disarming, no trigger found),
so **any policy gets ~1.4 catch opportunities per game.** At ~80% conversion that caps dex/game
near 1.1–1.2, and a uniform random policy already scores 0.88.

A planning agent (`tools/mpc.py`, no training) confirms it: it reaches the saucer **5.9× as
often** as random and enters catch mode **slightly less often per frame**, because the extra
visits land while the saucer is disarmed. Its higher dex/game is purely survival — it plays 1.8×
longer. So **dex/game is a survival metric in disguise**: (arms per game) × 0.8, and arms accrue
with time on the table. That is why 190M+ training steps across eight runs all read as noise.

**So train for points.** Score is ungated and dense, and it is the only objective here with room:
`score-02` measures **+2.4σ on score per 10,000 frames** (20.0M vs random's 15.2M, bootstrapped
95% CI [+0.7M, +8.6M]) and **+2.3σ on dex/game** (1.06 vs 0.79) — the first time anything in this
project has beaten random on the Pokédex objective, achieved by not targeting it.

**Judge on rates, not per-episode counts.** Every policy trained here buys survival time,
which buys more chances at the objective without improving the chance per frame — and that
confound has absorbed three experiments. `saucer-03` reads +1.9σ on dex/game and exactly 0.00σ
on saucer visits per 10,000 frames. `tools/eval.py` reports both.

## Measured facts

Everything here came out of `tools/validate.py`; none of it is documented anywhere public.

- **Throughput**: 9,542 steps/sec single worker; **64,320 at 10 workers** (peak efficiency
  is ~8 workers — 14 workers adds only 3% over 10). Raw PyBoy with no env wrapper is 22,896
  fps, so the observation layer costs ~60% and is the place to optimize if needed.
- **Game length**: 12k–24k frames (~3.5–7 min). **~4,000 frames per ball.**
- **Random policy** scores 1.8M–79M and catches **zero** species — the sparse-reward
  problem is real, and the curriculum is not optional.
- **A catch attempt starts when the ball comes to rest in the saucer at ball position
  `(124, 120)` while `0xD532 == 128`.** That byte is 128 on the first frame of every episode
  and drops to 0 the moment catch mode starts. Of 31 measured saucer visits, all 16 taken
  while it was 128 started catch mode and all 15 taken while it was 0 did not.
- **The whole objective is that one shot.** A random policy visits the saucer 1.55 times per
  episode, 0.88 of those while ready, and ~80% of attempts end in a catch with no further
  help — `catch_tiles_flipped` maxes out in every attempt, for a random policy, because the
  attempt lasts ~5,000 frames. So dex/game ≈ 0.88 for random, and every reward shaped at the
  in-mode end of the chain has produced a null result.
- **This corrects the claim this file used to make.** It said a random policy "is in a
  special mode ~32% of frames, so entering catch mode is easy and finishing it is the hard
  part". That 32% is a *duration* read as a *frequency*: one ~5,000-frame attempt inside a
  ~17,000-frame game. Entries happen ~0.9 times per game, identically for every policy
  trained so far including uniform random. The inverted claim shaped four runs.
- **The roulette slots are not the trigger.** `roulette_slots_opened` / `..._entered` (1.3–1.8
  and 1.0–1.5 per game) sit at a similar rate to catch attempts, which is a coincidence:
  measured entries occur with zero slots opened, and slot events land thousands of frames
  away from mode entry.
- **`0xD586` ("tile illumination", 48 bytes) is downstream, not upstream.** It is all zeros
  until `special_mode_active` goes 1, its bit count tracks `catch_tiles_flipped`, and its bits
  blink as an animation several times a second — unusable as a reward signal.
- `start_game()` leaves the game **pre-launch**; **A** launches the ball and also serves as
  the right flipper. But launching needs a press **edge**, not a held button: a policy that
  holds A from before the serve never launches at all (measured: every fixed action sits at
  stage 1, score 0, for 3,000 frames, while a random policy launches at frame 40). So the
  4-action space can launch unaided only if the policy is stochastic; the env toggles A on a
  15-frame duty cycle after a grace window to cover deterministic ones.
- **Holding both flippers up while the ball is low is a stable deadlock** — the ball rests on
  a raised flipper, stays low, and a policy that keeps holding never loses it. Episodes run to
  `max_frames` with zero events. Watch for `ep_len` pinned at the cap.
- **Pokémon Pinball GBC has no tilt/nudge mechanic** — no such address exists in the ROM
  map, which is why flippers alone are the complete action set.

## PyBoy wrapper bugs worked around

PyBoy 2.7.0's `game_wrapper_pokemon_pinball`:

- **`ball_x` / `ball_y` return the wrong byte.** Position is 16-bit fixed point at `0xD4B3`
  / `0xD4B5` (high byte = pixel, low byte = subpixel); the wrapper returns the **low** byte,
  i.e. the fraction. Verified identical to `raw16 & 0xFF` across 20,000 consecutive frames.
  Using it as position feeds the policy noise. We read the full word instead.
- **`ball_x_velocity` / `ball_y_velocity` are unsigned** — never negative in 40,000 frames,
  so direction is unrecoverable from them. We derive velocity from wrap-corrected position
  deltas and keep the raw values only as auxiliary features.

Useful things the wrapper does get right, and that saved real work: `pokedex` (per-species,
`2` == caught, from `0xD962`), plus `start_catch_mode(pokemon, unlimited_time=True)` and
`enable_evolve_hack()` — direct curriculum levers for stage 3.

## Attribution

`env/pinball_env.py` derives from NicoleFaye/pokemon-pinball-gym (MIT). See `NOTICE`.
