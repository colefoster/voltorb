# Does the saucer shot have structure near contact?

Measurement only. No training, no env or reward changes. Tool: `voltorb/tools/saucer_granularity.py`
(`collect` / `analyze` / `band` / `hazard`). Datasets: `data/saucer_gran/` and
`data/saucer_gran_pos/` (10 npz shards each, not committed).

## Verdict

**Near-contact structure exists, it replicates, and it is visible in the observation.** The
action played **53-66 frames (0.9-1.1 s) before the ball settles in the saucer** carries real
information about whether the shot lands, at up to **+9.4 sigma** on a single cell with an exact
null, **replicated on 19 held-out start states (z = +8.63, one pre-specified test)** and positive
in **37 of 39 states individually**. Conditioned on the ball being in the flipper region it is a
large effect: `P(both flippers | win)` is **0.334 against 0.25**, i.e. `P(win | both) = 0.238`
against `P(win | none) = 0.125` from **one frame's action**.

**Macro-level structure does not exist at this resolution.** Which 90-frame block was played --
either the opening prefix or the 90 frames before the event -- does not move the win rate:
Fisher-combined permutation p = **0.85** (prefix) and **0.27** (pre-event), with 1 of 39 states
individually significant against 2 expected.

**The `bc-mpc` conclusion does not survive as written.** "No per-frame conditional carries
information" is false. What survives is the weaker and quantitatively different claim that *the
per-frame conditional is small and concentrated in time*: pooled over a whole 90-frame window it
averages to nothing (`H = 1.3863`, ln 4, at every window length), which is exactly what the
first-action test measured, and it is far too weak for a supervised learner to pick up from the
sample sizes used so far. "Pixels would not fix it" is also unsupported -- the signal is already
present in a field the observation carries (`ball_y`).

## What was collected

Start states come from the env's own `EnvConfig(stage="shot")` pool builder
(`PinballEnv._build_shot_pool`, real shot opportunities harvested from random play: stage 0,
`catch_ready`, ball in play). Nothing synthesised -- `catch-01` is on record as invalidated by
made-up states. 10 worker processes, each built its own 48-state pool from its own seed and took
every 6th snapshot, giving **40 states**. Hit test is the env's own debounced counter
(`_saucer_visits`), never a hand-rolled one -- the `bc-mpc` row records that a hand-rolled test
returned 0 winners in 1,920 rollouts.

| | |
|---|---|
| rollouts | **24,000** (40 states x 600), uniform random policy, horizon 1,200 frames |
| episode end | env's `shot` rules: saucer visit, ball lost, or 1,200 frames |
| mean rollout | 868 frames; **17.5M emulator steps per collection** |
| win rate | 0.199 pooled raw; **0.178 after exclusion** (below) |
| ball lost | 9,167 of 24,000 |
| second collection | identical seeds/params **plus per-frame ball (x, y)**, reproduced the run exactly (4,766 winners both times) |
| wall clock | ~9 min per collection, 10 procs, ~3,950 steps/s each (~40k/s aggregate) |

**Explicit exclusion:** 1 of 40 states (`3018`) won **600/600** with median closest approach 0.0 --
the pool builder snapshotted a frame with the ball already at the saucer. It is dropped from every
test below (600 rollouts). Nothing else is capped or truncated. Per-state win rates for the
remaining 39 span 0.040 to 0.260, mean 0.178.

**Power.** Per state there are ~107 winners, so a single (offset, action) cell has
`se = 0.042` on `P(a | win)`: per-state single-cell tests resolve only ~0.084 deviations (2 sigma).
Pooled over 39 states (4,166 winners) `se = 0.0067`, resolving ~0.013. The effect found is
**0.049**, i.e. 1.2 sigma per state and 7.5 sigma pooled -- and this is why the `bc-mpc`
measurement could not have seen it: its 16 states x 32 rollouts gave ~89 winners in total, where
the same effect is 1.1 sigma. For the macro test, 8 categories x ~75 rollouts per state resolves a
cluster win-rate contrast of ~0.088 absolute (0.178 -> 0.266, a 1.5x relative gain); the Fisher
combination over 39 states resolves a consistent contrast of ~0.02. The forward-hazard analysis
uses 19.4M frames and 108,316 events, where a 1.05x hazard contrast is ~5 sigma.

Alignment choice: losers are aligned on their **closest approach** to the saucer (their nearest
analogue of the event; median closest approach 5.2 px against the 2.0 px visit radius), with a
**random frame** as a control. Both are reported. The primary tests -- 1 and 1b -- use **winners
only** and need no loser alignment at all: the generating policy is iid uniform over 4 actions, so
`P(a | win) = 0.25` exactly under H0, the actions are exogenous, and any deviation is causal.

## 1. Window entropy: null, and this is what `bc-mpc` measured

Entropy of the pooled action counts in the last N frames before the visit, against a
count-matched multinomial null (exact, since the policy *is* uniform), 2,000 draws.

| N | winners | samples | H_obs | H_null | sd | z | per-state Stouffer (39 states) |
|---|---|---|---|---|---|---|---|
| 5 | 4,166 | 20,830 | 1.3862 | 1.3862 | 0.0001 | -0.73 | +0.45 |
| 10 | 4,166 | 41,660 | 1.3863 | 1.3863 | 0.0000 | -0.17 | +0.53 |
| 20 | 4,166 | 83,320 | 1.3863 | 1.3863 | 0.0000 | -0.86 | +0.15 |
| 40 | 4,166 | 166,640 | 1.3863 | 1.3863 | 0.0000 | +0.79 | -0.27 |
| 90 | 4,166 | 374,940 | 1.3863 | 1.3863 | 0.0000 | -3.65 | -0.64 |

The N=90 pooled cell is nominally p = 0.007 on chi-square, but the deviation is
`p(a) = [0.2498, 0.2483, 0.2522, 0.2497]` -- **0.2 percentage points** -- and the per-state
Stouffer is -0.64. Read as: **averaged over a window, the winning action distribution is uniform to
within a fifth of a percent, at every window length from 5 to 90.** This reproduces `bc-mpc` and
explains why a global action-frequency shift (which is all `alley-sym` and the flipper-cost runs
ever did) buys nothing.

## 1b. The same data, one offset at a time: this is where the signal is

`P(action | win)` per offset before the visit, pooled winners, exact binomial null.

| offset | p(none) | p(left) | p(right) | p(both) | z(none) | z(left) | z(right) | z(both) |
|---|---|---|---|---|---|---|---|---|
| -66 | 0.2715 | 0.2302 | 0.2568 | 0.2415 | **+3.20** | -2.95 | +1.02 | -1.27 |
| -64 | 0.2458 | 0.2333 | 0.2789 | 0.2420 | -0.63 | -2.49 | **+4.31** | -1.20 |
| -61 | 0.2576 | 0.2295 | 0.2784 | 0.2345 | +1.13 | **-3.06** | **+4.24** | -2.31 |
| -60 | 0.2703 | 0.2273 | 0.2768 | 0.2256 | **+3.02** | **-3.38** | **+3.99** | **-3.63** |
| -58 | 0.2650 | 0.2223 | 0.2825 | 0.2302 | +2.24 | **-4.13** | **+4.85** | -2.95 |
| -56 | 0.2283 | 0.2525 | 0.2362 | 0.2830 | **-3.24** | +0.38 | -2.06 | **+4.92** |
| **-55** | 0.2091 | 0.2751 | 0.2156 | **0.3003** | **-6.10** | **+3.74** | **-5.13** | **+7.50** |
| **-54** | **0.1973** | 0.2837 | 0.2204 | 0.2986 | **-7.85** | **+5.03** | **-4.42** | **+7.25** |
| -53 | 0.2415 | 0.2590 | 0.2292 | 0.2703 | -1.27 | +1.34 | **-3.09** | **+3.02** |
| -40..-1 | | | | | all \|z\| < 2.2 | | | |

Summary over all 90 x 4 = 360 cells: max \|z\| = **7.85** at offset -54; **24 cells with
\|z\| > 3** against 1.0 expected; largest deviation from 0.25 is **0.053**.

Note the **sign flip** between the two sub-bands: at -58 to -66 the favoured actions are *right*
and *none* and the penalised ones are *left* and *both*; at -53 to -56 it reverses. Any policy
change that shifts action frequencies uniformly in time cancels this out exactly, which is a
mechanical reason every global-frequency intervention in the ledger was null.

### Is it real?

| check | result |
|---|---|
| synthetic iid control, same shape, 20 draws | cells \|z\| > 3: mean **0.95**, max 3 (observed 24) -- machinery is calibrated |
| random split-half of winners, 3 draws | corr(z, z') = **+0.466 / +0.464 / +0.463** |
| split by **state group** (19 vs 20 states, disjoint start states) | corr = **+0.454** |
| **held-out, one pre-specified test**: discover the pattern on group A, project group B's deviation onto it | group A offset -54 `p(a) = [0.196, 0.285, 0.219, 0.299]`; group B (2,051 winners) `p(a) = [0.198, 0.282, 0.221, 0.298]`, **z = +8.63** |
| **per-state (primary)**: contrast `p(both) - p(none)` over offsets -53..-56 | mean **+0.069**, positive in **37/39** states, per-state z mean +1.99 sd 1.22, **Stouffer +12.45** |

Implied one-frame effect at offset -54, base win rate 0.178:
`P(win | none) = 0.141`, `P(win | right) = 0.157`, `P(win | left) = 0.202`,
`P(win | both) = 0.213`. **A 1.51x relative swing from a single frame's action.**

## 2. Mutual information with losers aligned: weak, and the control shows why to distrust it

`sum_k I(a at offset -k ; win)`, per state, permutation null (2,000 label shuffles per state).

| N | closest-approach alignment | random-frame control |
|---|---|---|
| 5 | Stouffer +1.73 (obs 0.5407 vs null 0.4917 nats) | -0.53 |
| 10 | +2.15 (1.0948 vs 1.0086) | +0.25 |
| 20 | +2.30 (2.2719 vs 2.1244) | +0.97 |
| 40 | +0.41 (4.3366 vs 4.3099) | -2.11 |
| 90 | +2.41 (10.5091 vs 10.2357) | +2.14 |

Marginal either way, and the random-frame control moves nearly as much at N=90, so the aligned
version is not cleanly interpretable -- aligning losers on an *extremum* of the trajectory selects
frames where the ball's motion, and therefore the recent action, is non-generic. This is why the
winners-only tests are primary. The state-stratified win-rate version of the same scan (test 2b in
the tool) agrees with 1b about where the signal is (max \|z\| 6.53 at offset -54, largest shift
0.035 win-rate points) but inherits the same alignment caveat.

## 3. Macro-level separation: clean null

90-frame blocks reduced to 9 interpretable features (per-action frequency, switch rate,
any-flipper rate, any-flipper rate in each third), standardised within state, k-means into
**8 categories per state**, chi-square of win rate across categories with a permutation null
(2,000 within-state label shuffles).

| block | states | Fisher on permutation p | states with p_perm < 0.05 |
|---|---|---|---|
| opening 90-frame prefix | 39 | chi2 65.09, **p = 0.85** | 1 (expected 2.0) |
| 90 frames before the event | 39 | chi2 85.23, **p = 0.27** | 1 (expected 2.0) |

Per-state cluster win rates scatter within the binomial noise band (e.g. state 12:
0.238 / 0.178 / 0.250 / 0.239 / 0.186 / 0.230 / 0.227 / 0.200 against a 0.218 base).
**At n = 600 per state, 8 categories, this resolves a 0.178 -> 0.266 contrast per state and ~0.02
combined, and there is nothing there.** The "options / macro-actions" direction is retired by this:
whole-block identity does not predict the outcome; only *when* a specific frame's action lands
does.

## 4. Out-of-sample prediction from the window: null, and consistent with the above

Logistic regression on one-hot(action x offset) for the last-N window, 5-fold stratified CV, per
state, with 20 label-permutation nulls.

| N | delta nats/rollout vs base rate (mean over 39 states) | best state | per-state z mean | Stouffer | states z>2 |
|---|---|---|---|---|---|
| 5 | -0.0057 | +0.0096 | +0.14 | +0.86 | 2 |
| 10 | -0.0120 | +0.0081 | +0.15 | +0.91 | 2 |
| 20 | -0.0252 | +0.0005 | +0.15 | +0.93 | 1 |
| 40 | -0.0538 | -0.0158 | -0.10 | -0.61 | 0 |
| 90 | -0.1012 | -0.0169 | +0.17 | +1.08 | 2 |

Every arm is negative: from 600 rollouts per state a learner cannot recover the effect, because the
effect is ~0.01-0.03 nats concentrated on ~4 of ~890 frames, and it does not know which frames.
**A real, causally identified signal and an unlearnable-at-this-sample-size signal are the same
thing here** -- which is the honest reconciliation of "the mechanism exists" with three failed BC
attempts, rather than "no information exists".

## 5. Is it in the observation? Yes -- and here is the exploitable size

The second collection recorded ball (x, y) at every applied action.

Conditioning the band on ball height (`ball_y` is observation field 1):

| offset | subset | n | p(a\|win) = [none, left, right, both] | z | H | KL vs uniform |
|---|---|---|---|---|---|---|
| -54 | all | 4,166 | 0.209 / 0.275 / 0.216 / 0.300 | -6.1 / +3.7 / -5.1 / +7.5 | 1.3743 | 0.0120 |
| -54 | **ball y >= 110 (flipper region)** | 2,344 | **0.176 / 0.295 / 0.195 / 0.334** | **-8.3 / +5.1 / -6.2 / +9.4** | **1.3508** | **0.0355** |
| -54 | ball y < 110 | 1,822 | 0.251 / 0.249 / 0.243 / 0.257 | +0.1 / -0.1 / -0.7 / +0.7 | 1.3861 | 0.0002 |
| -61 | ball y >= 110 | 2,352 | 0.287 / 0.216 / 0.269 / 0.228 | +4.2 / -3.9 / +2.1 / -2.5 | 1.3794 | 0.0069 |
| -61 | ball y < 110 | 1,814 | 0.239 / 0.253 / 0.263 / 0.246 | -1.1 / +0.2 / +1.3 / -0.4 | 1.3857 | 0.0006 |

**The entire effect lives in the frames where the ball is near the flippers, and it is exactly zero
where it is not (KL 0.0002 nats).** So the informative frames are identifiable from the
observation, not only from "how long until the event", and a per-frame conditional policy on the
existing 34-float observation *can* express this. That kills the "34 floats underdetermine the shot,
this argues for pixels" reading of `bc-mpc`.

Forward, exploitable version -- for every frame of every rollout, label = a debounced visit lands
in the next 45-70 frames, bucketed by ball position (8 x 8 grid over 0-160 px):

* 19,399,297 frames, 108,316 events, base hazard 0.00558.
* **Within-bucket action effect: chi2 = 469.6 on 114 df, p = 1.1e-44** over the 38 buckets with
  >= 2,000 frames. The action changes the hazard *given the observable ball position*.
* Largest buckets: `x[40,60) y[120,140)` n=352,226, hazard by action
  [0.0353, 0.0276, 0.0347, 0.0282] (**1.28x** best/worst); `x[60,80) y[120,140)` n=669,600,
  [0.0221, 0.0262, 0.0225, 0.0259] (1.18x). Note the best action differs by bucket -- it is
  aiming, not a global preference.
* **Held-out greedy rule** (fit the best action per bucket on half the rollouts, score on the
  other half, 2.44M frames on the rule's arm): hazard **0.00586 vs 0.00563 base = 1.041x**.

That last number is the honest ceiling for *this* representation of the conditional: a
position-only, 8x8, single-frame greedy rule buys **~4%**, where `tools/mpc.py` buys **7x**
(8.33 vs 1.17 visits/10k). So the near-contact conditional is real, causal, observable -- and, on
its own, small. The gap between 1.04x and 7x is the part that still needs the joint sequence. What
is untested is the obvious next refinement: the buckets here use position only, while the
observation also carries velocity, and the effect is plainly a contact-timing effect.

### Why BC measured 0.0004 nats

`KL(P(a | win) || uniform)` at the peak offset is **0.0144 nats/frame** (0.0355 in the flipper
region). Diluted over a ~890-frame rollout:

| informative frames | mean nats/frame over the rollout |
|---|---|
| 4 | 0.000065 |
| 15 | 0.000243 |
| 30 | 0.000487 |

`bc-mpc` measured **0.0004 nats** over uniform on 19,500 filtered MPC samples. The band found here
is ~14 offsets wide, so the predicted dilution and the measured BC gap agree to within a factor of
about two. **The BC nulls are quantitatively consistent with dilution, not with absence of signal.**

## Caveats

* One start-state pool per worker, one policy (uniform random). All numbers describe the shot as
  reachable from random play; a trained policy's state distribution is different.
* The held-out replication is across disjoint *start states* within one collection, not across
  independent collections. The second collection was seed-identical by design (to attach ball
  positions to the same rollouts), so it is a determinism check, not an independent replication.
* The 45-70 frame lookahead in the hazard analysis was chosen from the band found in test 1b, so
  its p-value is conditional on that discovery; the held-out 1.041x is not.
* `P(win | a)` figures are one-frame contrasts under a uniform-random continuation. They are not a
  promise about a policy that plays that action at every opportunity -- playing it changes the
  state distribution downstream.
