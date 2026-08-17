# Run ledger

One line of hypothesis per run. Newest first.

| run | stage | steps | hypothesis | result |
|---|---|---|---|---|
| `dex-01` | dex | 50M | Warm-started from `score-01`. The real objective: reward only species not already in the Pokedex this episode. Target is dex/game clearly above random's 0.67. | running |
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
