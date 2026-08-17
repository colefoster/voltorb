# Run ledger

One line of hypothesis per run. Newest first.

| run | stage | steps | hypothesis | result |
|---|---|---|---|---|
| `dex-03` | dex | 40M | Fine-tune `catch-01` on the full game. The catch skill is real but does not transfer, so the missing step is keeping it while learning to *reach* catch mode. 12 envs, not 16, since both mid-flight crashes used 16. | running |
| `catch-01` | catch | 50M | Warm-started from `score-01`. Train the catch sub-task in isolation: every episode starts inside a forced catch attempt and ends when it resolves. Random baseline is 12% catch rate over ~1,965-frame episodes, so there is finally a gradient. | **WORKS (+3.2 sigma).** Crashed at 26M/50M but the checkpoint holds: catch rate **0.20 +- 0.02 vs random 0.12 +- 0.02** over 400 episodes each, a 67% relative gain. First statistically real result on the objective. |
| `dex-02` | dex | 50M | Rebalanced: new species 300 (was 100), catch progress 2.0 (was 0.5), height 0.001 (was 0.01), score off. dex-01's shaping paid ~100/episode against 100 for a whole species, so the objective was drowned. | **null again, stopped at 31M/50M.** dex/game 0.82 +- 0.10 vs random 0.94 +- 0.07 = -0.9 sigma, i.e. slightly worse. `dex>=1` 61% vs 78%. The reward-balance hypothesis is disproven: 3x the species bonus with shaping off changed nothing. |
| `dex-01` | dex | 50M | Warm-started from `score-01`. The real objective: reward only species not already in the Pokedex this episode. Target is dex/game clearly above random's 0.67. | **null result.** dex/game 1.05 +- 0.09 vs random 0.94 +- 0.07 = +1.0 sigma, indistinguishable. `dex>=1` 76% vs random's 78%. Training showed `game/dex_caught` rising 0.75 -> 0.99 but that was the rolling window, not a real gain. |
| `score-01` | score | 50M | Warm-started from `survive-03`. Adding log-scaled score deltas on top of height shaping teaches the agent to hit things on purpose without losing the retention it already has. Watch that `ep_len` does not regress below ~20k while `game/score` climbs. | **worked.** 24-episode eval: 20,438 frames (+34% over random), score median 22.4M (+19%), mean 70.6M (3.7x), dex 1.00/game vs random 0.67. Best checkpoint so far. |
| `survive-03` | survive | 50M | With `ent_coef=0.001` and the height-shaped reward, ball retention actually learns. | **worked.** Monotone across all five 10M buckets: 17,938 / 18,127 / 18,852 / 19,089 / 20,646 mean `ep_len`. Final 20-ep window **23,955 vs 16,891 random (+42%)**, best 26,731. Entropy 1.386 -> 1.251, `explained_variance` -0.124 -> 0.826, `advantage_std` 0.088 -> 0.151. |
| `survive-01` | survive | 50M | Frame-level 4-action control learns ball retention from 26 RAM floats. | **failed — no learning.** Entropy 1.385 -> 1.381 (max is ln 4 = 1.386), i.e. still uniform random after 50M steps. `ep_len` 17.9k -> 19.4k is noise. Cause: `ent_coef=0.01` overwhelmed a weak advantage signal, and the constant `+0.01`/frame alive bonus was information-free. |

## Established facts

- Baseline to beat: **16,891 frames/episode** (random policy, `tools/validate.py`).
- **A constant per-frame reward cannot train anything.** It is identical in every state and
  for every action, so the critic learns a constant and advantages collapse. Measured under
  the old reward: `advantage_std` 0.13 vs `return_mean` 8.15, `explained_variance` 0.955.
  Replaced with ball-height shaping, which varies with state.
- **`ent_coef=0.01` is too high for this task.** Sweep at 600k steps, frames survived:
  | ent_coef | frame_skip | entropy | frames |
  |---|---|---|---|
  | 0.01 | 1 | 1.385 -> 1.368 | 18,319 |
  | **0.001** | **1** | **1.385 -> 1.250** | **21,516** |
  | 0.001 | 4 | 1.386 -> 1.352 | 18,568 |
  | 0.0 | 4 | 1.386 -> 1.280 | 15,980 |
  | 0.001 | 8 | 1.386 -> 1.054 | 15,624 |
- **Higher frame_skip survives worse** — independent confirmation that frame-level control
  was the right call, not just a preference.
- **`clipfrac` is 0.0000 in every config, learning or not.** It is not a useful health
  metric here; `losses/entropy` is. Do not read clipfrac 0 as "the policy is frozen".
- Compare `ep_len` across frame_skip settings only after multiplying by frame_skip.

## Evaluation (24 episodes each, `tools/eval.py`, corrected dex counting)

| checkpoint | frames | score mean | score median | dex/game | dex>=1 % |
|---|---|---|---|---|---|
| random | 15,244 | 19,096,573 | 18,846,675 | 0.67 | 58% |
| survive-03 | 17,702 | 27,815,144 | 20,241,425 | 0.83 | 71% |
| score-01 | 20,438 | 70,585,021 | 22,450,375 | 1.00 | 62% |

- **Trust `tools/eval.py`, not the training scalars.** The `charts/episodic_length` rolling
  window flattered survive-03 badly: it reported +42% over random where a clean 24-episode
  eval says +16%.
- **The done bar needs revisiting.** A random policy already catches 0.67 species/game and
  clears >=1 in 58% of games, so "at least 1 new species in >=90% of games" is a much
  weaker target than it sounded when set. Judge on dex/game against 0.67.
- Score means are outlier-skewed; the median is the honest comparison.

## Bugs that silently corrupted earlier results

- **Pokedex bytes are a bitfield, not an enum**: bit 0 = seen, bit 1 = caught, so caught
  reads 2 or 3. PyBoy's `has_pokemon()` tests `== 2` and misses most real catches. Every
  dex number before this fix read 0.00.
- **PyBoy persists SRAM to `<rom>.ram` and reloads it on boot**, so each run inherited the
  previous run's catches — 8 species already caught at reset. The env now zeroes the dex
  region before snapshotting the boot state.
- **`gw.pokedex` is a tick-hook cache**, so reading it right after `load_state()` returns
  the *previous* episode's values. The env reads Pokedex bytes straight from memory now.
- **The ball never launched for deterministic policies** — pre-launch coordinate bytes hold
  garbage, so a `ball_y > 0` test latched "launched" immediately and disabled the fallback.
- **Terminal info arrives under `infos["final_info"]`**, not at the top level, so
  `game/score` logged nothing for an entire 50M-step run.

## Definitive evaluation (80 episodes each, with standard errors)

| checkpoint | frames +- se | score median | dex/game +- se | dex>=1 % |
|---|---|---|---|---|
| random | 17,200 +- 637 | 20,748,100 | 0.94 +- 0.07 | 78% |
| score-01 | 21,204 +- 1,021 | 22,902,350 | 1.18 +- 0.10 | 76% |
| dex-01 | 20,240 +- 1,425 | 22,164,825 | 1.05 +- 0.09 | 76% |

**Ball control works; catching does not.** score-01 survives +23% longer than random at
+3.3 sigma -- a real result. But on the actual objective, score-01 is +1.9 sigma and dex-01
is +1.0 sigma, both inside the noise. No checkpoint meets the done bar, and random has the
best `dex>=1` rate of the four.

**Why dex-01 failed, and it is a design error not a bug.** Over a ~20,000-frame episode the
height shaping paid `0.01 * ~0.5 * 20,000 ~= 100` reward. One new species was also worth
100. The shaping term was worth as much as the entire objective and the score term added
more on top, so the agent optimised survival and bumpers, which is exactly what it was paid
to do. Fixed in dex-02: species 300, height 0.001 (~10/episode), score 0.

**Always eval with standard errors.** At n=24 random scored 0.67 dex/game; at n=80 it scored
0.94. Per-episode variance is ~1 catch on a mean of ~1, so anything under 2 sigma is noise.

## Where the dex objective actually stands

Three attempts (dex-01, dex-02, and score-01 incidentally) all land within noise of random
on catches, while ball control improves significantly. The reward-weight explanation is
dead -- dex-02 tripled the species bonus, zeroed score, and cut height shaping 10x, and
came out marginally worse.

**The leading explanation is now horizon, not weighting.** `gamma=0.999` gives an effective
horizon of ~1,000 frames. A catch requires: enter a special mode, flip ~6 catch tiles, hit
the target ~6 times to reveal the silhouette, then hit it 3-4 more times -- a sequence
spanning several thousand frames. A 300-point payout that far downstream is discounted to
nothing by the time credit reaches the flipper actions that caused it. The agent is not
being stubborn; it cannot see the connection.

Two candidate fixes, in order of promise:

1. **Train the sub-task directly.** `gw.start_catch_mode(pokemon, unlimited_time=True)` forces
   catch mode on demand. Reset straight into it so every episode is a short, dense catch
   attempt instead of a 20,000-frame game where catches are incidental. This converts a
   sparse long-horizon problem into a dense short-horizon one, then transfers back.
2. **Raise gamma to 0.9999** (horizon ~10,000 frames) for the dex stage alone. Cheaper to
   try, but slower to learn and it does not fix the underlying rarity.

Do (1) before spending another 50M steps on the full game.

## The catch sub-task (stage `catch`)

Every episode starts already inside a catch attempt via `gw.start_catch_mode()`, with a
randomly chosen species so the policy learns the mechanic and not one target.

Getting a usable baseline took two fixes, both found by measuring instead of training:

- **Terminating on `special_mode_active` going false was wrong.** It flickers off mid-attempt
  -- observed at frame 2,940 with 96 seconds still on the clock -- which cut episodes to a
  1,320-frame median and produced a **0/30** catch rate. Termination is now catch success,
  ball lost, or game over.
- **The natural 120s mode timer ends attempts before a learning policy could finish one**, so
  the sub-task uses `unlimited_time=True`. Put the timer back once catches are reliable.

Random baseline in this stage: **12% catch rate (5/40)**, episodes ~1,965 frames, and catch
progress (tiles + hits) nonzero in 40% of episodes. That is the first version of this
objective with a gradient to climb -- full-game episodes hid one catch inside 20,000 frames,
well beyond what gamma=0.999 can assign credit across.

## First real result on the objective

`catch-01`, evaluated on the catch sub-task, 400 episodes per arm:

| | frames +- se | catch rate +- se | >=1 catch |
|---|---|---|---|
| random | 1,790 +- 59 | 0.12 +- 0.02 | 12% |
| catch-01 (26M steps) | 2,205 +- 64 | **0.20 +- 0.02** | 20% |

**+3.2 sigma.** The horizon hypothesis was correct: three full-game attempts failed because a
catch spans several thousand frames and gamma=0.999 reaches back ~1,000. Shrink the episode
to match the credit window and the agent learns the mechanic.

Open question: does it transfer? A high catch rate inside forced catch mode is only useful
if it raises dex/game in real games, where the agent must also *reach* catch mode.

## Run stability

Two runs have now died mid-flight -- `dex-02` stopped at 31M, `catch-01` crashed at 26M with
an EOFError from a vector-env worker (a worker died and took the pipe with it). Both used 16
envs and both failed in the 26-31M range, which smells like a slow leak in the PyBoy worker
processes rather than a one-off. Checkpoints survived both times, so it costs progress
rather than results. Worth periodic worker recycling if it keeps happening.

## The catch skill does not transfer (yet)

Evaluated on full games, 100 episodes per arm:

| | frames +- se | dex/game +- se | >=1 catch |
|---|---|---|---|
| random | 16,945 +- 549 | 0.91 +- 0.06 | 77% |
| score-01 | 21,568 +- 1,037 | 1.10 +- 0.10 | 72% |
| catch-01 | 19,127 +- 862 | 0.97 +- 0.07 | 78% |

catch-01 is +0.7 sigma in full games despite being +3.2 sigma at the sub-task. This is a
useful decomposition rather than a dead end: the binding constraint in a real game is **not
finishing a catch** -- the agent demonstrably does that better than chance now -- it is
**reaching catch mode at all**, which forced-mode training never exercised.

Next: `dex-03` fine-tunes the catch-trained weights on the full game, which is the curriculum
step that was missing. If that also fails to move dex/game, the honest follow-up is a
sub-task for *entering* catch mode, rewarding the shots that trigger it.
