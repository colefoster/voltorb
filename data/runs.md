# Run ledger

One line of hypothesis per run. Newest first.

| run | stage | steps | hypothesis | result |
|---|---|---|---|---|
| `alley-sym` | alley | 60M | **The policy only ever flips both flippers at once, and the flipper cost is why.** Measured on `alley-long/final.pt` over 123 live samples: mean action probabilities [none 0.758, **left 0.000, right 0.000**, both 0.242], entropy 0.315 nats against ln4 = 1.386. The cause is `pinball_env.py:448`, which set `flipper_held` as a **boolean** (`float(self._held != ACTION_NONE)`) while `rewards.py:57` charges `flipper_cost * flipper_held` -- so holding BOTH costs exactly as much as holding one, and `ACTION_BOTH` **strictly dominates** `ACTION_LEFT`/`ACTION_RIGHT`: same price, more ball contact. The ledger's own histograms corroborate the causal story -- `alley-02`, before `--flipper-cost` existed, ran left 62.8% / right 34.6% / both 2.2% and *did* use single flippers; the collapse to none/both appears only from `alley-03`, the run that introduced the flat charge. Hypothesis: an aimed shot is unrepresentable with a single both-flippers stroke, which would explain why `vis/10k` has never moved in any run. Fix is one line -- count raised flippers (0/1/2) so BOTH costs 2x -- verified at `flipper_cost=1.0`: none 0.000, left -1.000, right -1.000, both -2.000. Everything else identical to `alley-long`, from scratch, seed 1. Deciding metric **vis/10k** (random 1.17), with ramp/10k as the guard that the arm gain survives. | **Null, and it refutes the hypothesis rather than failing to confirm it: `both` is load-bearing.** The action space did reopen -- single-flipper mass went 0.000 -> 0.225 -- but it re-collapsed the other way (final mix [none 0.551, left 0.225, **right 0.000**, both 0.224], entropy 0.393), swapping which flipper it abandoned. And every honest metric got worse. n=300, seed 1234, bootstrapped per 10k frames, against `alley-long`: **ramp/10k +4.4 sigma (was +10.6)**, **arm/10k +1.2 sigma, NOT significant (was +5.0)**, **entries/10k +0.6 sigma, NOT significant (was +6.1)**, **dex/10k +0.2 sigma, NOT significant (was +5.2)**, and **score/10k -3.5 sigma -- significantly WORSE than random** (was +2.5). `vis/10k` is **1.06 against random's 1.17**, i.e. below it: pricing the double-flip did not buy aim, it lost ground. **Do not quote the per-episode block:** frames are 21,807 +- 971 vs random's 17,932 (+22%), so `dex/game` +3.3 sigma and `arms/ep` +3.6 sigma are the survival confound that already faked `saucer-01`, `saucer-03` and `dex-03`. This policy survives longer and does less per frame. **What is established:** both flippers together is the *effective stroke* for the right ramp, not an artefact of mispricing -- the flat charge was letting the policy find the right action, and doubling the price of its best action simply degraded it. The `{none, both}` collapse is therefore real but benign, and **'the collapsed action space is why the shot never trained' is dead**. The saucer wall is not the action space. `alley-long` stands as the project's best result, unchallenged. Reverted in `pinball_env.py:448` with the reasoning recorded in-line so this is not re-attempted. Caveat: one training seed, one eval seed -- but four independent metrics moved the same way with large effects, so a seed rerun is unlikely to rescue it. |
| `alley-b1/b2/b3` | alley | 3x20M | **Ball loss is barely punished, so punish it properly.** `ball_lost_penalty` is **1.0** against episode returns of ~130-450, and a game has 3 balls, so draining the whole game costs **-3, under 1% of the return** -- the only thing discouraging it is the forgone future reward. The MPC planner, the one component that plays like a pinball player, charges the equivalent of -100 and survives 1.8x longer. Raise it to **50**, comparable to the alley bonus, on top of the `alley-03` config, at 3 seeds. | **Refuted, and it broke the thing that was working.** Frames **17,516 / 17,224 / 18,112 against random's 17,932** -- a 50x larger drain penalty bought **no survival at all**, which kills the premise directly. It also destroyed the ramp gain: ramp/10k **0.99 / 1.04 / 1.14** against `alley-03`'s 1.16 and random's 0.97, and entries/10k went **-1.9 / +0.6 / -0.3 sigma** against `alley-03`'s +2.6. dex/game 0.65 / 0.82 / 0.92 against random's 0.85. Best read: a -50 spike on drain dominates the return and makes the policy risk-averse, and a ramp shot is exactly a risk -- it sends the ball up the side where it can drain. So it traded away the objective for safety **it did not even obtain**. The implicit cost of ending the episode was already doing this job; an explicit penalty this size only adds variance. |
| `alley-s2/s3/s4` | alley | 3x20M | **The `alley-03` recipe at three fresh training seeds.** `score-02` (+2.4 sigma) did not survive a seed change (`score-04` +0.9, `score-05` -0.0), so no single-seed result in this ledger can be trusted. This is the same config, seeds 2/3/4, judged together with `alley-03` as four samples of one recipe. | **It replicates -- the first result in this project that survives a multi-seed test.** n=300 per arm, per 10k frames. **ramp/10k significant in all four: +4.1 / +3.8 / +6.0 / +3.9 sigma**, mean effect +0.221 with between-seed sd 0.048. **entries/10k positive in all four (+2.6 / +2.0 / +2.3 / +1.5 sigma), mean +0.046 with between-seed sd 0.0081** -- individually marginal, but the scatter is a fifth of the effect, which is what a real effect looks like and the exact opposite of score's +2.4 / +0.9 / -0.0. arms/10k +1.4 / +1.8 / +3.9 / +2.0, all positive. **Unexpected and it matters: score/10k is negative** (-2.3, -0.3, -1.7 sigma) -- these policies trade points away for ramp shots, so they are specialists, not general improvements. Saucer aim (`vis/10k`) remains flat in every seed. |
| `alley-03` | alley | 20M | **The policy never times a flip -- it holds one.** Actions are held flipper *states*, not presses, so sitting on a raised flipper is free, and `alley-02` found that: measured over 3 episodes it has a flipper up on **99.6% of frames** (none 0.4%, left 62.8%, right 34.6%, both 2.2%) and switches every ~2.6 frames, against random's 74.7% up. That is not aiming, it is resting on the flippers and jittering, and it is a plausible direct cause of `vis/10k` never moving: a ball that lands on an already-raised flipper cannot be shot deliberately. New adjustable `--flipper-cost`, off by default so every earlier run reproduces. At **0.001** it charges ~20 over a 20,000-frame episode against ~10 for the height term and ~100 for the alley bonus -- enough to price loitering, not enough to outbid the objective. Everything else identical to `alley-02`. Deciding metric **vis/10k** (random 1.17, `alley-02` 1.14 -- flat, which is the whole point), with ramp/10k as the guard that the arm gain is not lost. Success: vis/10k clear of random at >=2 sigma bootstrapped. Watch the action histogram: if `none` does not rise well off 0.4%, the cost is too small; if `ep_len` collapses, it is too large. | **Failed its own bar, replicated a real result underneath it, and caught a false positive in the act.** The flipper cost worked behaviourally and dramatically: `none` went **0.4% -> 45.6%**, any-flipper 99.6% -> 54.4% (random 74.7%), so the policy now actually releases the flippers. But the pre-registered metric **failed**: `vis/10k` is **1.14 vs random 1.17**, flat, exactly as `alley-02`. Pricing the hold does not make it aim at the saucer. Two independent eval seeds, n=300 each (1234 / 777), and the difference between them is the point: **ramp/10k +4.1 / +6.5 sigma**, **arms/10k +1.4 / +2.7 sigma**, **entries/10k +2.6 / +3.1 sigma** all replicate, while **dex/10k came in at +2.6 sigma on the first seed and +0.8 sigma on the second -- it does not replicate.** The first seed's dex result was flagged as suspect before the second was run, on the grounds that it was one of 10 significance tests and this project has watched exactly that kind of number evaporate before (`score-01`). It duly evaporated. **What is established:** `alley-03` reliably beats random per frame on ramp shots, arms, and **catch-mode entries -- which the ledger already names as the headline metric** -- with frames at 17,976 vs 17,476 (+2.9%), so it is not the survival confound. **What is not:** dex/10k, and saucer aim, which is flat in both seeds. Note also that the random arm itself moved between eval seeds (ramp/10k 0.97 -> 0.91, dex/game 0.85 -> 0.91), so single-seed evals of the floor-limited dex metric are worth little. |
| `score-04` | score | 20M | **Housekeeping, and it is a regression I caused.** Adding the four alley inputs took the observation 30 -> 34, so `runs/score-02/final.pt` no longer loads at all (size mismatch on `trunk.0.weight`) and the project's best score checkpoint is currently unrunnable. Retrain the exact `score-02` recipe -- log transform, `score_weight=0.1`, from scratch -- at the new observation width. Not a hypothesis; the point is to restore a loadable baseline before anything else is measured against it. Success is simply landing near `score-02`'s 20,019,247 per 10k; a large gap either way means the four new inputs changed the score stage, which would itself be worth knowing. | **It did not reproduce, and that is a problem for the ledger's best result rather than for this run.** n=200: score/10k **16,940,368 vs random 15,505,125 = +1,402,447, 95% CI [-1,785,406, +4,574,226], +0.9 sigma -- not distinguishable from random**, where `score-02` on the same recipe measured **+2.4 sigma**. `diag/advantage_std` ran 0.284 -> 0.327 against `score-02`'s 0.277 -> 0.321 and `explained_variance` 0.87 -> 0.92, so the run is a faithful reproduction of the training dynamics; it is the *outcome* that did not come back. Two readings. Either the four alley inputs dilute a 34-wide input layer on a stage where they are irrelevant, or **`score-02`'s +2.4 sigma was substantially seed luck** -- its CI was [+715,053, +8,645,541] and barely excluded zero. `score-05` runs the same thing at seed 2 to separate them. **The methodological point is the bigger one:** every result in this ledger is bootstrapped over *episodes* at n=100-300 but trained at **n=1 seed**, and episode-level resampling cannot see training-seed variance at all. Two runs of one recipe landing at +2.4 and +0.9 sigma is the first direct measurement of how large that unmeasured term is. |
| `bc-mpc` | (imitation) | -- | **Distil the MPC planner instead of the random search.** `bc-01`/`bc-02` fit advantage-weighted targets from `collect_demos.py` and recovered 0.007 nats over uniform; the diagnosis on record was that many sequences reach the saucer so `P(a | obs, win)` is nearly uniform. The planner does not obviously have that problem -- at each frame it has committed to one action, so the label is a point mass. New `tools/collect_mpc.py` records (obs, executed action) from planner play. Falsifier as stated going in: if the fit beats uniform the shot is learnable and every null was the learner; if it does not, 34 floats underdetermine the shot and that argues for pixels. | **Null, and it refutes its own falsifier -- the problem is not the representation.** First collection came back with an action histogram of `[741, 749, 748, 762]`, dead uniform, because only **4 of 50 decisions had a rollout that reached the saucer**: the planner is best-of-N *random* sequences, so on the other 92% it is only avoiding the drain and its choice is an arbitrary tie-break. Added `--only-informative` and raised rollouts 12 -> 32, which lifted the teacher to **8.33 saucer visits/10k against random's 1.17**, matching the independent MPC figure of 8.93. Fit on 19,500 filtered samples: **val loss 1.3859 against the uniform reference 1.3863 -- 0.0004 nats -- and val accuracy 0.26 against chance 0.25.** Three distinct labelling schemes have now failed identically. **The mechanism, measured directly:** from 16 real shot-opportunity states, 32 rollouts of 1,200 frames each (win rate 0.174, matching the 12.5% on record), the entropy of the first action **among winners** is **1.1995 against a count-matched uniform null of 1.0867 +- 0.060, z = +1.88** -- the winning action is if anything *less* concentrated than chance. Bucketed by ball height to test whether the signal is merely diluted by frames where the ball is airborne, it is flat everywhere: z between -0.32 and +1.09 across five buckets at n = 5,000-16,000 each. **From a given state the set of winning sequences is diverse enough that no per-frame conditional carries information. Winning is a property of the joint sequence, so per-frame behaviour cloning cannot work at any representation, and adding pixels would not fix it.** Caveat on the second test: it pools across states, which can wash out state-dependent structure -- the load-bearing measurement is the per-state one, which controls for that. Also recorded: my first version of the identifiability script returned 0 winners in 1,920 rollouts against a 2.5% expected base rate; the hand-rolled hit test was wrong and was replaced by the env's own debounced visit counter. |
| `alley-02` | alley | 20M | `alley-01` moved ramp shots +2.0 sigma with episode length held level -- the first aiming gain in the project that is not survival in disguise -- but it paid for the wrong events. `wRightAlleyCount` runs 0..3, the arm fires at **2**, and catch mode zeroes the byte, so the reward schedule should be **progress toward the threshold and nothing past it**: pay 50 on 0->1 and 1->2, pay **zero** on 2->3, and pay again after the reset. One-line change (`min(count, alley_target)` before the diff) and it removes a third of `alley-01`'s payouts, all of them noise. Everything else identical, from scratch. Deciding metric stays **ramp/10k at n=300** (random 0.97), but the real question is now conversion, so **arms/10k (random 0.46) is the confirmation** and it needs the ramp gain to roughly double before n=300 can see it: at 2 ramps per arm, ramp/10k 1.6 predicts arms/10k ~0.75. Success: ramp/10k >= 1.6 **and** arms/ep clear of random at >=2 sigma. Kill: if ramp/10k lands at ~1.1 again, the ceiling is not the reward schedule and the next lever is the MPC distillation, not another shaping term. | **WORKS on the arm gate -- the first per-frame aiming gain in the project -- and it locates the remaining bottleneck precisely.** 300 episodes per arm. The per-episode numbers are large and **must not be quoted**: frames are 20,810 vs 17,932 (+16%), so ramp +6.0 sigma / arms +3.8 sigma / entries +4.1 sigma / dex/game +3.0 sigma are all riding the survival confound that faked three earlier results. Bootstrapped **per 10,000 frames**, which is the honest test: **ramp/10k +0.208 [95% CI +0.117, +0.297] = +4.5 sigma**, **arm/10k +0.094 [+0.022, +0.166] = +2.5 sigma**, entries/10k +0.040 [-0.001, +0.081] = +1.9 sigma, dex/10k +0.023 [-0.024, +0.068] = +1.0 sigma, score/10k +0.2 sigma. So `dex/game`'s +3.0 sigma **was survival**, and the objective itself is still not beaten. Against its own bar it is a **split**: arms cleared 2 sigma on the honest metric, ramp/10k landed at **1.18, well short of the 1.6 target**. The threshold fix is what did it -- `game/alley_shots` rose monotonically 1.53 -> 2.10 -> 2.81 where `alley-01` oscillated 1.85 / 1.26 / 2.10 with no trend, from a one-line `min(count, 2)`. **Re-scored, `alley-01` was not a null either**: its ramp/10k is +0.130 [+0.032, +0.225] = +2.6 sigma. It was null on *conversion* (arm/10k -0.4 sigma), which is exactly what paying for the worthless 2->3 increment predicts. **The chain attenuates, and the leak is identifiable.** Ramps convert to arms almost perfectly: +0.208 ramp/10k at 2 ramps per arm predicts +0.104 arm/10k and measured +0.094. Arms to catch-mode entries lose about half: +0.094 arms should give ~+0.075 entries at the 80% conversion rate, measured +0.040. And **`vis/10k` is 1.14 vs random's 1.17, dead flat** -- `alley-02` opens the gate 20% more often and is no better at putting the ball in the saucer. **The arm gate is no longer the bottleneck; the shot is**, which is the one thing `shot-01` (+1.2 sigma) and `shot-02` (+0.3 sigma) both failed at. Next lever is the MPC/DAgger distillation, not another shaping term: it tests whether 34 floats can express the shot at all. Also fixed here: `tools/eval.py` now bootstraps ramp/arm/entry/dex **per 10k frames** and marks the per-episode block superseded -- those counts had no interval at all before, which is how +6.0 sigma and +4.5 sigma got to look like the same claim. |
| `alley-01` | alley | 20M | **The arm gate is a ramp shot, and the ceiling that justified abandoning the dex objective does not exist.** From the pret/pokepinball disassembly, verified against the ROM: `0xD532` is `wIndicatorStates + 3`, and `HandleSecondaryRightAlleyTrigger_RedField` / `HandleThirdRightAlleyTrigger_RedField` both do `ld a,[wRightAlleyCount] / cp $3 / ret z / inc a / ... / cp $2 / ret c / ld a,$80 / ld [wIndicatorStates+3],a`. So **catch-ready is exactly `wRightAlleyCount >= 2`**, that byte increments when the ball passes the secondary right alley trigger after the main one latched `wRightAlleyTrigger`, it caps at 3, zeroes at stage init, and catch mode consumes it. Over 53,205 frames the arm transitions and the count increments run in lockstep; the 4.7% of frames that break the biconditional are catch-mode bookkeeping lag. **Two errors are retired at once.** The `0.42 re-arms/episode` behind '~1.4 opportunities per game for any policy' was measured at **n=12**; at **n=120 it is 0.708 +- 0.075**, a 69% underestimate and 3.8 sigma out. And the quantity was never policy-invariant -- it is a count of ramp shots, so 'no trigger found, it happens on maps 2/3/5 at any balls_left' was looking at the wrong end of the causal chain. **Why every saucer run was null is now mechanical, not statistical:** `saucer-01/02/03` all aimed at the saucer, which has no controllable precursor, while the agent's observation carried only `catch_ready` -- a *derived bit*. It could see whether the shot was live and never how close it was to making it live, so 'get one more ramp' was literally unrepresentable. Obs 30 -> 34 adds `right_alley_count`, `right_alley_trigger`, `left_alley_count`, `left_alley_trigger`; the reward is `DexReward` plus **50 per `wRightAlleyCount` increment** (~2/episode under random, so ~100 against 300 for a species). Deliberately **not** proximity-shaped at the ramp -- three runs already went to distance-to-a-fixed-point and all three were null on aim. Deciding metric is **ramp shots per 10,000 frames**, random **1.08**, with arms/10k (random **0.61**) as the confirmation that the ramps convert. Judge at **n=300, not 100**: these are Poisson-ish counts with sd ~ mean, and n=100 cannot resolve the 1.2-1.9 sigma band that three earlier runs died in. Success: ramp/10k >= 1.6 at >=3 sigma. Kill criterion: if ramp/10k is flat at 20M, the ramp is not more learnable than the saucer was and the constraint is the 4-action frame-level policy, not the target. | **null against its own bar, but it is the first survival-neutral aiming gain in the project.** 300 episodes per arm: **ramp/10k 1.10 vs random 0.97**, `alley_shots/ep` **+0.24 = +2.0 sigma (significant)** -- and **frames are 17,971 vs 17,932, dead level**, so for the first time a movement on an aiming quantity is not the survival confound that ate `saucer-01`, `saucer-03` and `dex-03`. It missed the >=1.6-at-3-sigma bar and it did **not** convert: `arms/ep` -0.02 = -0.3 sigma, `dex/game` +0.01 = +0.1 sigma, score/10k +0.1 sigma. `game/alley_shots` in training oscillated 1.85 / 1.26 / 2.10 / 1.60 with no trend, so most of the eval gain is not a learning curve. **The cause is a specification error I put in, and the disassembly names it exactly.** `wRightAlleyCount` counts 0..3 while the gate opens at **2**, and catch mode ends with `xor a / ld [wRightAlleyCount], a` -- it zeroes the byte, so each arm costs **two fresh ramps** and the 2->3 increment buys nothing. `alley-01` paid 50 for every increment, so **a third of the events it paid for were worthless**, and nothing in the reward let the agent prefer the two that matter. **The conversion test was also underpowered, not failed:** +0.24 ramps/ep at 2 ramps per arm predicts +0.12 arms/ep against a combined se of ~0.10, i.e. ~1.2 sigma -- n=300 cannot resolve it either way, so 'the ramps did not convert' is not yet a measurement. One thing genuinely new in the diagnostics: `diag/explained_variance` ran 0.57 -> 0.79 against 0.90-0.96 in every other run, with `advantage_std` at 2.7-3.8 against ~0.3. The critic could not predict this reward, which is what a *controllable* signal looks like and the opposite of what `score-03` produced. See `alley-02`. |
| `score-03` | score | 20M | `score-02` worked but its reward transform is nearly rank-blind. Measured over 1,099 random-play scoring events: **45% pay 100, 21% pay 5,000, 21% pay 50, and the 3% paying >=1,000,000 carry 71% of all the points.** Under `log10(1+delta)` a 100-point bumper pays 2.0 and a 3,000,000-point jackpot pays 6.5 -- a **3.3x reward ratio for 30,000x the points**, so the policy is nearly indifferent between them, and `diag/advantage_std` sat flat at 0.27-0.31 for all of `score-02`, which is what that compression looks like from the critic. Switch to **`sqrt`** with `score_weight=0.003`, chosen so the *total* score budget per episode is unchanged (0.003 x sum(sqrt) = 67/ep against 0.1 x sum(log10) = 72/ep under random) -- only the ranking inside the budget changes. Now a jackpot is worth **173x** a bumper instead of 3.3x, while a 100-point event still pays 0.03, ~3x a max-height frame, so the **97.9% reward density that makes this stage trainable survives**. A per-million linear term was rejected for the opposite reason: it pays ~0.0001 for 97% of events and throws the density away, and raw linear deltas caused +90 sigma advantage spikes in earlier runs. Per-frame magnitude stays bounded near the 1.0 ball-lost penalty (3M -> 5.2, the largest event ever measured, 11.4M -> 10.1) with a `score_clip=15.0` backstop that never binds on measured data. From scratch, everything else identical to `score-02`, so the transform is the only variable. Success: score/10k clearly above `score-02`'s 20,019,247 at >=2 sigma bootstrapped, judged per 10,000 frames and never per game. Watch `diag/advantage_std`: if it is still flat at ~0.3 the compression was not the binding constraint. | **null, and it disproves the compression hypothesis outright.** 100 episodes per arm, matched seeds: score/10k **18,583,587 vs random 15,191,818 = +3,374,874, 95% CI [-1,501,235, +8,278,975], +1.3 sigma -- not distinguishable from random**, where `score-02` on the identical eval is +2.4 sigma. Head to head, **score-03 minus score-02 is -1,367,935/10k [-6,214,645, +3,662,776] = -0.5 sigma**: the two policies are not separable from each other, but only one of them separates from random, so the transform bought nothing and cost the result. dex/game fell to 0.84 +- 0.07 (+0.5 sigma) from `score-02`'s 1.06 (+2.3 sigma), and frames fell to 18,536 from 20,433. **The mechanism did fire and it did not help:** `diag/advantage_std` ran 0.376 -> 0.444 against `score-02`'s 0.277 -> 0.321, ~40% more advantage magnitude, exactly what re-ranking predicted -- while `diag/explained_variance` sat at 0.90 -> 0.96, *unchanged*. More advantage variance that the critic still fully explains is more gradient noise on returns the policy does not control. **The payout mix is the decisive measurement.** Re-measuring the delta stream per ~90k frames: random gets 22 events of 1,000,000 and 11 of 3,000,000; `score-02` gets **54 and 25**; `score-03` gets **14 and 7** -- fewer jackpots than random, while its 50/100-point event count went *up* (1,130 vs random's 728). A reward that pays a jackpot 173x a bumper produced a policy that collects more bumpers and fewer jackpots. So the big payouts are not a separately targetable action the policy was declining to take because it was underpaid; `score-02` reached them by playing well and surviving, and re-weighting toward them pushed the policy off the behaviour that produces them. **Log-compression was not the binding constraint, and `score-02` remains the best checkpoint.** |
| `score-02` | score | 20M | The dex objective is capped at ~1.4 arms/game and random already takes 0.88 of the ~1.1 available, so it is over. **Points are not gated, and they are dense.** An MPC planner pointed at score gets **29,426,333 per 10k frames vs random's 9,946,459 (3.0x)** and finds a usable rollout in **97.9% of decisions against 35% for the saucer** -- the reward-density problem that produced eight nulls simply is not present here. `score-01` already worked once on this stage (+19% median, mean 3.7x) and was abandoned mid-curriculum, not because it failed. Re-run the established recipe, but **from scratch** -- `survive-03` predates the observation change and its 26-input trunk no longer loads against 30 -- with log-scaled score deltas and `score_weight=0.1` and judge on **score per 10,000 frames**, not per game, since score/game rises with survival on its own. Success: >= 1.5x random's 9.95M/10k at >=3 sigma. Note for later: log10 compression makes a 10k event and a 10M event nearly equal in reward, which is wrong for maximising score -- but it also caps the advantage spikes that a linear transform would create, so change it only if this plateaus. | **WORKS, and it is the first result on the dex objective too.** 100 episodes per arm: **score/10k 20,019,247 vs random 15,191,818 = +4,791,231, 95% CI [+715,053, +8,645,541], +2.4 sigma** (bootstrapped over episodes -- scores are heavy-tailed enough that a ratio of sums needs a resampled interval, so the point estimate of +32% carries a real range of +5% to +57%). Frames 20,433 vs 17,871, so the gain survives the survival confound that ate three earlier experiments. **dex/game 1.06 +- 0.10 vs 0.79 +- 0.06 = +2.3 sigma, significant -- the first time anything in this project has beaten random on the stated objective**, and it came from not aiming at it: `ent/10k` is 0.59 vs 0.59, dead flat, so it did not get better at catching, it got more chances. Exactly what "dex = arms x 0.8, arms accrue with time" predicts. Against the MPC ceiling of 29.4M/10k this captures ~16% of the available headroom. |
| `shot-02` | shot | 20M | `shot-01` proved the shot is not found from a cold start, while `shotsearch` proved an open-loop winning sequence exists (90-frame prefix reproduces at 0.379 vs 0.127). That is exploration, so start at the goal and walk backwards. Start states are harvested free rather than searched: play randomly with a ring of savestates and, on each observed saucer visit, keep the snapshots from 45/90/180/360/720 frames earlier. Measured random hit rate per level: **0.700 / 0.325 / 0.275 / 0.175 / 0.125** -- a smooth ladder from trivial to the full task. Each worker promotes when its own last-50-episode rate clears 0.70 and demotes below 0.25. Evaluated on the **plain** shot task so the number is comparable to `shot-01`'s 0.19 vs random 0.15. Success: hit rate >= 0.25 at >=3 sigma over 300 episodes. Watch `game/shot_level`: if it never leaves 0, the ladder is too steep; if it pins at 4 early with a flat hit rate, the curriculum is not the binding constraint either. | **null, and it is worse than `shot-01`.** 300 episodes per arm on the plain shot task: hit rate **0.16 vs random 0.15 = +0.3 sigma**, against `shot-01`'s +1.2 sigma, so the curriculum made it slightly worse rather than better. **The ladder is the diagnostic and it never left the bottom rung:** `game/shot_level` averaged 0.59 at the first bucket and 0.98 at the last, i.e. mean level ~1 of 5 after 20M steps, and `game/saucer_visits` on the *eased* task ran 0.428 -> 0.415 -> 0.338, flat-to-falling for the whole run. Random hits 0.700 at level 0 and 0.325 at level 1; the curriculum policy sits at ~0.41 on that mixture, so **it is at or below random even 45 frames from the goal, with the ball already most of the way there.** That is not an exploration failure the curriculum could fix -- starting the agent 45 frames from a known-reachable saucer is as close to a solved problem as this task has, and it still did not clear the 0.70 promotion gate. Combined with `shot-01`, the reading is that the 4-action frame-level policy cannot express the shot from the 30-float observation, not that it cannot find it. The remaining untried lever on this sub-task is the one that was always paired with the curriculum and never run: **behaviour-clone the winning `shotsearch` prefixes** (`runs/bc-01.pt`, `bc-02.pt` exist and were never evaluated) and check whether a supervised policy can reach 0.379 where PPO reaches 0.19. If BC cannot either, the observation is the constraint. |
| `shot-01` | shot | 20M | The shot is controllable (`tools/shotsearch.py`: a 90-frame action prefix triples the hit rate, +9.0 sigma state-matched), so three saucer nulls are credit assignment, not a ceiling. Train the shot in isolation, exactly the shape that made `catch-01` the only run that ever worked: episodes start from savestates of **real** shot opportunities sampled from random play in this same env (ball in play, stage 0, saucer ready), last <=1,200 frames, and end on the visit. Random base rate **12.5% +- 2.3%** measured at that horizon, against 12% for the catch sub-task. Reward is deliberately narrow: +10 for the saucer visit, ~1.0/episode of stage-gated proximity shaping, -1 for losing the ball, nothing else. Unlike `catch-01` the start states are real, so no observation field is frozen at a value that never occurs. Deciding metric: hit rate over 300 episodes vs a matched random arm; then transfer -- does it raise `ent/10k` in full games? | **null.** 300 episodes per arm: hit rate **0.19 vs random 0.15 = +1.2 sigma**. Training scalars were flat for 20M steps -- `game/saucer_visits` 0.130 at the first bucket and 0.139 at the last, with entropy oscillating 0.96 -> 0.57 -> 1.17 rather than settling. Transfer was not measured; there is nothing to transfer. **This is the sharpest null in the ledger**, because the task was built to remove every excuse: real start states, 12.5% base rate, 625-frame mean episode inside the credit window, a 10:1 objective-to-shaping ratio, and a measured 0.379 achievable at the 90-frame prefix. PPO reached 0.19. |
| `saucer-03` | saucer | 20M | Both earlier saucer runs paid the distance term **ungated by stage**, and `current_stage` indexes two different screens: over 121,752 frames the ball is at (124,120) in 1,437 frames of stage 0 and **zero** frames of stage 1, all visits and entries are stage 0, and stage 1 is **69% of all frames**. So most of the shaping budget paid for proximity to a coordinate on the wrong screen. Same as `saucer-02` with the raw term gated on `current_stage == 0` and the weight raised 0.002 -> 0.005 to keep ~20/episode through a 31% duty cycle. Deciding metric `saucer_visits/ep` vs random 2.11 +- 0.10. | **null, and it is the clearest null of the three.** 100 episodes per arm, torch now seeded per arm: `saucer_visits` **2.40 vs random 2.11 = +1.2 sigma**, `catch_entries` 1.24 +- 0.08 vs 1.05 +- 0.07 = **+1.8 sigma**, dex/game 1.00 +- 0.09 vs 0.79 +- 0.06 = +1.9 sigma, `dex>=1` 70% vs 69%, frames 20,270 vs 17,871 (**+13%**). Everything points the right way and nothing clears 2 sigma. **Normalise by frames and the gain vanishes:** visits per 10,000 frames 1.18 for saucer-03 and 1.18 for random; entries per 10,000 frames 0.61 vs 0.59. The gate fix was real and correct, and it bought episode length, not aim. |
| `saucer-02` | saucer | 20M | `saucer-01` moved `catch_entries` +1.7 sigma but `saucer_visits` **not at all** (-0.4 sigma), so the potential-based distance term did not make it aim. PBRS is policy-invariant by construction and telescopes to ~phi(end)-phi(start) per episode; the one shaping term this project has ever gotten to work (ball height, `survive-03`) is a **raw** per-frame state term. Same setup, `saucer_shaping="raw"`: +0.002 per frame times proximity to the saucer, ~22/episode against 100 per entry and 300 per species. Hovering near the saucer is not farming here -- it is the precondition. Deciding metric `saucer_visits/ep` (random 2.11 +- 0.10). | **null, marginally negative.** 100 episodes per arm: `saucer_visits` **2.04 vs random 2.11 = -0.3 sigma**, `catch_entries` 0.96 +- 0.07 vs 1.05 +- 0.07 = **-1.0 sigma**, dex/game 0.73 vs 0.79, `dex>=1` 59% vs 69%, frames 17,706 vs 17,871 (flat). Raw proximity shaping did not make it aim either -- and unlike `saucer-01` it did not even buy retention. Cause found afterwards and it is a defect, not a ceiling: the term was ungated by stage, so ~69% of what it paid was proximity on a screen where the saucer does not exist. See `saucer-03`. |
| `saucer-01` | saucer | 20M | Catch mode is one shot: the ball at rest in the saucer at `(124,120)` while `0xD532 == 128`. Random gets 1.55 visits/ep and 0.88 entries/ep. Given `catch_ready` plus the ball's offset to the saucer in the observation, and potential-based shaping on distance to it (+100 per catch-mode entry, species 300, no in-mode progress bonus), the agent should aim. From scratch: the obs changed, and the survive/score warm start buys ball-holding, not aiming. Deciding metric `catch_entries/ep` over 100 episodes vs a matched random arm; success >= 1.5 at >=3 sigma, kill if inside 2 sigma of 0.88 at 10M. | **null on the deciding metric, and it says something specific.** Ran 20M/20M with no crash (first completed run since `033e2ef`). 100 episodes per arm, matched: `catch_entries` **1.22 +- 0.07 vs random 1.05 +- 0.07 = +1.7 sigma**, `saucer_visits` **2.03 vs 2.11 = -0.4 sigma**, dex/game 0.91 +- 0.08 vs 0.79 +- 0.06 = +1.2 sigma, `dex>=1` 71% vs 69%. Frames 20,469 vs 17,871 (+15%). So it learned ball retention *again* -- the one thing every run learns -- and the extra catch entries are the longer episodes, not aim: **visits per episode did not move at all.** The dense potential-based distance signal did not produce aiming. |
| `dex-03` | dex | 40M | Fine-tune `catch-01` on the full game. The catch skill is real but does not transfer, so the missing step is keeping it while learning to *reach* catch mode. 12 envs, not 16, since both mid-flight crashes used 16. | **null, and it crashed too** — stopped at 19.8M/40M, a third mid-flight crash. 40-episode eval: dex/game **0.88 +- 0.09 vs random 0.88 +- 0.11 = +0.0 sigma**, dead flat. `dex>=1` 78% vs 72%. Survival regressed: 16,775 frames vs random's 19,245. Entropy had collapsed to 0.32 (max 1.386), so the policy went nearly deterministic and bought nothing for it. The catch skill still does not transfer, and fine-tuning on the full game is not the missing step. |
| `catch-01` | catch | 50M | Warm-started from `score-01`. Train the catch sub-task in isolation: every episode starts inside a forced catch attempt and ends when it resolves. Random baseline is 12% catch rate over ~1,965-frame episodes, so there is finally a gradient. | **WORKS (+3.2 sigma).** Crashed at 26M/50M but the checkpoint holds: catch rate **0.20 +- 0.02 vs random 0.12 +- 0.02** over 400 episodes each, a 67% relative gain. First statistically real result on the objective. |
| `dex-02` | dex | 50M | Rebalanced: new species 300 (was 100), catch progress 2.0 (was 0.5), height 0.001 (was 0.01), score off. dex-01's shaping paid ~100/episode against 100 for a whole species, so the objective was drowned. | **null again, stopped at 31M/50M.** dex/game 0.82 +- 0.10 vs random 0.94 +- 0.07 = -0.9 sigma, i.e. slightly worse. `dex>=1` 61% vs 78%. The reward-balance hypothesis is disproven: 3x the species bonus with shaping off changed nothing. |
| `dex-01` | dex | 50M | Warm-started from `score-01`. The real objective: reward only species not already in the Pokedex this episode. Target is dex/game clearly above random's 0.67. | **null result.** dex/game 1.05 +- 0.09 vs random 0.94 +- 0.07 = +1.0 sigma, indistinguishable. `dex>=1` 76% vs random's 78%. Training showed `game/dex_caught` rising 0.75 -> 0.99 but that was the rolling window, not a real gain. |
| `score-01` | score | 50M | Warm-started from `survive-03`. Adding log-scaled score deltas on top of height shaping teaches the agent to hit things on purpose without losing the retention it already has. Watch that `ep_len` does not regress below ~20k while `game/score` climbs. | **worked.** 24-episode eval: 20,438 frames (+34% over random), score median 22.4M (+19%), mean 70.6M (3.7x), dex 1.00/game vs random 0.67. Best checkpoint so far. |
| `survive-03` | survive | 50M | With `ent_coef=0.001` and the height-shaped reward, ball retention actually learns. | **worked.** Monotone across all five 10M buckets: 17,938 / 18,127 / 18,852 / 19,089 / 20,646 mean `ep_len`. Final 20-ep window **23,955 vs 16,891 random (+42%)**, best 26,731. Entropy 1.386 -> 1.251, `explained_variance` -0.124 -> 0.826, `advantage_std` 0.088 -> 0.151. |
| `survive-01` | survive | 50M | Frame-level 4-action control learns ball retention from 26 RAM floats. | **failed — no learning.** Entropy 1.385 -> 1.381 (max is ln 4 = 1.386), i.e. still uniform random after 50M steps. `ep_len` 17.9k -> 19.4k is noise. Cause: `ent_coef=0.01` overwhelmed a weak advantage signal, and the constant `+0.01`/frame alive bonus was information-free. |

## Established facts

- **Single-seed training is the biggest unmeasured error term in this project.** Every entry
  above bootstraps over episodes (n=100-300) while training exactly **one seed**, and episode
  resampling cannot see training-seed variance. `score-02` measured +2.4 sigma on score/10k;
  `score-04`, the same recipe re-run, measured **+0.9 sigma and not distinguishable from
  random**. Treat any single-seed result in the 1-3 sigma band as provisional until a second
  seed agrees. `alley-03` is the first entry judged against two independent eval seeds, and it
  cost one headline claim: dex/10k read +2.6 sigma on one and +0.8 sigma on the other.

- **Per-frame imitation is closed, for a measured reason.** `bc-01` (0.007 nats), `bc-02`, and
  `bc-mpc` (0.0004 nats) all failed identically. From a fixed shot-opportunity state, the
  entropy of the winning first action is **1.1995 against a count-matched uniform null of
  1.0867 +- 0.060** -- no concentration, and the wrong sign. Bucketing by ball height to test
  the "actions only matter near the flippers" dilution hypothesis gives z between -0.32 and
  +1.09 at n = 5,000-16,000 per bucket: flat everywhere. Winning is a property of the joint
  1,200-frame sequence and many sequences win, so there is no per-frame target to fit. **This
  also rules out the representation explanation**: no observation, pixels included, makes a
  non-identifiable label identifiable. Do not spend another run on BC/DAgger/AWR.

- **RETRACTED: "a random policy is in a special mode ~32% (or 58%) of frames, so entering
  catch mode is easy and finishing it is the hard part."** That is a duration statistic read
  as a frequency: one ~5,000-frame catch attempt inside a ~17,000-frame game. Catch-mode
  entries happen **~0.9 times per game, for every policy ever trained including uniform
  random.** The inverted claim was written into `rewards.py` as "reward in-mode progress,
  never mode entry" and steered `dex-01`, `dex-02`, `dex-03` and `catch-01` -- four nulls
  aimed at the wrong end of the chain.
- **RETRACTED: "the dex objective is hard-capped at ~1.4 arms per game for any policy, so
  the headroom is ~25% and that is why eight runs produced nothing."** Two stacked errors,
  both found 2026-08-19. The `0.42 re-arms/episode` it rested on was **n=12**; the n=120
  value is **0.708 +- 0.075**. And the quantity is not a property of the game at all --
  `0xD532` is `wIndicatorStates + 3` and equals `wRightAlleyCount >= 2`, a count of right-ramp
  shots that caps at 3 and is consumed by catch mode. The arm is **earned, repeatable, and
  aimable**. "No trigger found, it happens on maps 2/3/5 at any balls_left, median 7,320
  frames after disarm" was a description of how often a *random* policy happens to shoot the
  right ramp twice, read as a game-imposed cooldown.

- **Read the disassembly before measuring.** `pret/pokepinball` resolves in minutes what
  twelve episodes of polling could not, and the polling produced a confident wrong answer that
  redirected the whole project away from its stated objective. Any future "what causes X"
  question should start at `wram.asm` and `engine/pinball_game/`, not at a sampling loop.

- **The chain is one shot, measured 2026-08-18.** `0xD532 == 128` means catch mode is ready;
  it is 128 on the first frame of every episode and drops to 0 the instant catch mode starts.
  Catch mode begins when the ball comes to rest in the saucer at ball position `(124, 120)`.
  Of 31 saucer visits, **all 16 while ready started catch mode and all 15 while not ready did
  not** -- perfect separation. Random: **1.55 saucer visits/ep, 0.88 catch-mode entries/ep,
  ~80% of entries end in a catch unaided** (`catch_tiles_flipped` maxes out in every attempt).
- **The roulette slots are not the trigger.** Similar rate (1.0-1.5 entered/game) but no
  causal relation: measured catch-mode entries happen with zero slots opened, and slot events
  land thousands of frames from mode entry. A planned `slot` experiment was built and then
  abandoned before it ran, on this measurement.
- **`0xD586` (48-byte "tile illumination") is downstream of the bottleneck.** All zeros until
  `special_mode_active` goes 1, bit count tracks `catch_tiles_flipped`, and its bits blink as
  an animation -- a per-bit reward would have paid for the blink, not for progress.
- **The random baseline has to be measured at the same n as the thing it judges.** Quoted at
  0.67, 0.88 and 0.94 dex/game in different entries above, purely from episode count. The
  40-episode number is 0.88 +- 0.11; treat smaller ones as noise.
- **`score-01`'s dex advantage was small-sample noise.** Recorded above as 1.00/game vs
  random 0.67 over 24 episodes. Re-run at 40 episodes against a matched baseline it is
  **0.75 +- 0.11 vs 0.88 +- 0.11 = -0.8 sigma** — indistinguishable, and pointing the wrong
  way. Nothing has beaten random on the dex objective yet; only `catch-01`, on the isolated
  sub-task, has beaten anything.
- **The done bar is retired, not unmet.** ">=1 new species in >=90% of games" is cleared
  72-78% of the time by a uniform random policy, so it measured the game rather than the
  agent. The headline metric is now **catch-mode entries per game** (random 0.88 +- 0.19 at
  n=8, 1.00 including a re-arm); dex/game is its noisy, floor-limited consequence.

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

## What was actually wrong (2026-08-18)

The chain was measured end to end instead of assumed. Findings, in order of how much they
mattered:

1. **The founding fact was inverted** (see Established facts). Entering catch mode is the hard
   part; finishing it is free. Four runs were shaped at the free end.
2. **The trigger is a single fixed shot.** Ball at rest at `(124, 120)` while `0xD532 == 128`.
   Neither the readiness byte nor the ball's offset to that spot was in the observation, so
   the agent was asked to hit a target it could not see, for a payout it was never given.
3. **The roulette-slot chain in the previous handoff was a coincidence**, and `0xD586`
   ("tile illumination") is the in-mode animation, not the upstream light state. Both were
   about to be built into a reward; measurement killed the experiment before it ran.
4. **`catch-01` trained on a distribution that does not exist.** `start_catch_mode(
   unlimited_time=True)` never sets `ADDR_TIMER_ACTIVE`, so `timer_active` and
   `timer_remaining` were frozen at values seen in 0% of real attempts -- two of the
   observation's inputs were systematically wrong for that whole run. The `catch` stage is
   left in place but should not be built on until this is fixed.
5. **A silent action-space bug.** The launch fallback called `pyboy.button("a", 5)`, which
   queues a release 5 frames later, while `_apply_action` still believed A was held. After one
   fallback press the right flipper could stay released-but-believed-held for the rest of the
   episode, cutting the action space to two states. Triggers whenever the policy fails to
   press A within 120 frames -- common early in training. Fixed by declaring the press through
   `_held`.
6. **`dex_caught_frac` was a dead input** (`/151` put one catch at 0.0066 against features of
   order 0.5). Now `dex_caught / 8`.

### Where this leaves the project

**Train for points, and report the Pokedex as a consequence.** `score-02` is +2.4 sigma on
score/10k and +2.3 sigma on dex/game, the latter without ever targeting catches. Points are
ungated and dense where the dex objective is capped at ~1.4 arms per game, so score is the only
objective here with room to work in -- MPC says 3.0x is available and `score-02` took 1.32x.

The next thing to change is the reward transform. `ScoreReward` uses `log10(1 + delta)`, which
scores a 10,000-point event at 4 and a 10,000,000-point event at 7, so the policy is nearly
indifferent between a bumper and a jackpot -- while MPC's advantage came mostly from big
payouts (one episode hit 271M). `diag/advantage_std` sat at 0.27-0.31 and flat all run, which is
what a compressed reward looks like from the critic's side. A `sqrt` or per-million linear term
with clipping keeps more of the difference without reintroducing the +90 sigma advantage spikes
that a raw linear term would cause.

### THE ANSWER: the objective has almost no headroom over random

`tools/mpc.py` plans with the emulator as a forward model -- snapshot, roll 12 random futures,
execute the best one's first 100 actions, replan -- and it solves the aiming problem outright.
It does not improve the objective. 8 episodes per arm, uncapped:

| | frames/ep | visits/10k | **entries/10k** | dex/game | dex/10k |
|---|---|---|---|---|---|
| random (matched) | 15,771 | 1.51 | 0.63 | 0.38 | 0.24 |
| random (n=100 reference) | 17,871 | 1.18 | 0.59 | 0.88 | 0.49 |
| **MPC** | 28,965 | **8.93** | **0.52** | 1.38 | 0.47 |

**MPC lands in the saucer 5.9x as often and enters catch mode slightly LESS often per frame.**
Its higher dex/game is entirely survival: it plays 1.8x longer because the planner scores ball
loss at -100. Per frame it catches at exactly the random rate.

**SUPERSEDED 2026-08-19 -- the paragraph below is wrong in both its number and its premise.
See the `alley-01` entry: 0xD532 is `wIndicatorStates + 3` and equals `wRightAlleyCount >= 2`,
a count of right-ramp shots, and the re-arm rate is 0.708 +- 0.075 at n=120, not 0.42 at n=12.
There is no cap and no cooldown; there is a shot nobody was aiming at.**

The reason is the arm gate. Measured over 12 episodes: **0xD532 re-arms 0.42 times per episode**,
at 696-12,595 frames after it disarms (median 7,320, sd 4,357), with no trigger found -- not a
map change, not a new ball, and it happens on maps 2/3/5 at any balls_left. With the arm at
frame 0 that is **~1.4 catch opportunities per game, for any policy**, and MPC measured 1.5.

At ~80% conversion that caps dex/game at **~1.1-1.2 at natural episode length. Random already
gets 0.88.** The entire headroom on the stated objective is roughly 25%, which is inside the
noise of a 100-episode eval. **That is why 190M+ steps across eight runs produced nothing: there
was almost nothing there to find.**

**dex/game is a survival metric wearing a costume.** It is (arms per game) x 0.8, arms accrue
with time on the table, so the only real lever is keeping the ball alive -- which `survive-03`
and `score-01` already pull, at +42% and +23%. The project's two "wins" were the honest ones all
along.

If the objective is to stay, it has to be **dex per 10,000 frames**, and the target has to be the
arm gate: what re-arms 0xD532 is now the only question whose answer could change the ceiling.

### The shot IS controllable (2026-08-18, `tools/shotsearch.py`)

Settled without a human and without learning, using savestates. From a real in-play state with
the saucer ready, run 32 random 400-frame action sequences; take the ones that reached the
saucer, keep only their first N actions, and replay those with **fresh** randomness after.
32 states sampled, 15 produced at least one hit:

| replayed prefix | retest hit rate | vs state-matched base 0.127 |
|---|---|---|
| 30 frames (0.5 s) | 0.194 | **+2.8 sigma** |
| 90 frames (1.5 s) | 0.379 | **+9.0 sigma** |
| 200 frames (3.3 s) | 0.575 | +14.6 sigma |

**The first 90 frames of action triple the chance of reaching the saucer.** The information is
in the action sequence, so the shot is aimable and this is a credit-assignment / exploration
problem -- not the ceiling of RAM-only observation. (200 of a 400-frame horizon is half the
rollout fixed, so read 30 and 90; those are far from trivial replay and both clearly positive.)

**The baseline nearly produced the opposite answer.** Prefix retests only exist for states that
produced a winner, and those states are favourable by selection -- 10 of 22 states in the first
run hit 0/32 whatever the actions. Against the base rate pooled over *all* states the first run
read **+2.0 sigma "controllable"**; against the same states it read **-0.6 sigma**. The tool now
reports only the state-matched comparison. A selection-biased baseline is the same class of
error as the duration-read-as-frequency one that started this whole detour.

### Open-loop control exists; closed-loop control was not learned

`shotsearch` shows a winning 90-frame prefix reproduces at 0.379 against a state-matched base
of 0.127, and a full-length replay reproduces at 1.0 by construction -- so **from any given
state an open-loop action sequence that reaches the saucer usually exists.** `shot-01` then
failed to learn a state-conditioned policy that does the same thing, on exactly that
distribution of states, at 20M steps.

That is the gap worth attacking next, and the honest reading is that it is an exploration
problem: a random policy finds the winning sequence in ~13% of episodes, and PPO has no
mechanism to notice that the winner differed from the loser in its first 90 frames.

**Next, and it follows directly from the prefix measurement: a backward start-state
curriculum.** Use `shotsearch` to find states from which a winning path exists, then start
episodes 90 frames before the hit -- where the base rate is high -- and walk the start
backwards as the success rate holds. The demonstrations are already a by-product of the
search, so behaviour cloning on winning prefixes is the same data, and the two compose.

### What the sub-task was built from

A **saucer sub-task**, and the precedent is exact. `catch-01` is the only thing that ever
trained (+3.2 sigma) and it worked because the sub-task had a short horizon and a base rate
around 12%. A shot sub-task has the same shape: reset from a sampled shot-opportunity
savestate (in play, stage 0, saucer ready), episode of ~400 frames, reward the saucer visit,
**base rate 12.7% measured**. Everything that made the full game hopeless -- 17,000-frame
episodes, one opportunity per game, credit spanning thousands of frames -- is gone.

The alternative, and they compose: `shotsearch` already generates winning action prefixes, so
those are demonstrations. Seed a replay buffer or behaviour-clone them before fine-tuning.

### The earlier scripted-policy sweep, and why it proved less than it looked

Three trained runs move episode length and leave the rate alone. So the rate was attacked
without any learning involved -- thirteen hand-written policies, scored on saucer visits per
10,000 frames against random's 1.23:

| policy | frames/ep | visits/10k | entries/10k |
|---|---|---|---|
| random | 17,289 | **1.23** | 0.55 |
| flip only on stage 0 | 17,576 | 1.07 | 0.64 |
| random, right-biased | 17,344 | 1.06 | 0.70 |
| hold flippers up while ball low (4 variants) | 30,000 (truncated) | 0.00 | 0.00 |
| 3-frame tap on ball-low (6 variants) | 5,522-30,000 | 0.00 | 0.00 |

**Nothing beat random.** Two findings inside that, both about the env rather than the game:

- **Holding the flippers up while the ball is low is a stable deadlock.** The ball rests on a
  raised flipper, stays low, the policy keeps holding, and the episode runs to `max_frames`
  with zero events. This is a live hazard for training: a cradling policy earns a long episode
  and, under per-frame proximity shaping, gets paid for it. `saucer-03` did not fall in
  (20,270 frames/ep against a 108,000 cap), but an `ep_len` pinned at `max_frames` means this.
- **The tap policies got zero visits because they keep the ball alive far worse than random**
  (5,522 frames/ep against 17,289), not because they aimed badly. Visits need a live ball
  bouncing in the upper field, so a policy that flips rarely never gets there. That makes this
  sweep evidence about survival, **not** evidence that aiming is impossible.

So the honest position is: **no policy, learned or hand-written, has raised the rate -- and the
aim question is still open**, because every policy that raised it would first have to match
random's ball retention. The cheap instrument that settles it is a human: `tools/play.py`
scores a human game on exactly the same counters. If a human clears ~1.2 visits/10k the shot is
aimable and this is a credit-assignment problem for a different algorithm; if a human lands
near 1.2 as well, catch mode is luck given time on the table, dex/game is survival x luck, and
maximising frames -- which every run already learns -- is close to optimal for RAM + flippers.

### Two launch bugs, one of them introduced this session

- **Launching needs a press EDGE, not a held button.** Measured: with A held from before the
  serve, every fixed action -- including the two that hold A forever -- sits at stage 1 with
  score 0 for 3,000 frames, while a random policy launches at frame 40. An input sweep
  confirms only A launches (frame 50 as a tap; A toggled at 2/4/8-frame periods launches at
  frames 32-38). `README.md`'s "the 4-action space can launch unaided" is true only for
  policies that generate A edges, i.e. stochastic ones.
- **The fallback was applied after the tick**, so the next step's action released A before the
  emulator advanced a frame with it down and the press was invisible. Fixed by overriding the
  action *before* the tick and toggling A on a 15-frame duty cycle through `_apply_action`, so
  `_held` stays truthful. All four fixed actions now launch at frame 165. **This never affected
  a training run** -- policies with entropy 1.1-1.3 launch at ~frame 40 -- but it silently
  invalidated the first scripted-policy sweep, and it would invalidate any greedy-policy eval.

### The known ceiling

Arm events measured **1.62 per episode** including the one at frame 0, so catch attempts per
episode are capped near ~1.6 unless re-arming is faster when the saucer is consumed early
(the two episodes that re-armed had both consumed it early, so this is plausible but
unmeasured). At ~80% conversion that projects to dex/game ~1.3. The retired 90% bar needed
lambda ~2.9. **If `saucer-01` works, the next question is what re-arms `0xD532`.**

### Crash class, closed by hardening rather than by diagnosis

Three long runs died mid-flight with no logs. The memory-leak hypothesis in the previous
handoff was disproven by measurement (per-process RSS flat across ~40 `record_video` cycles),
and there is no macOS crash report at any of the three times. Two of three died within 16
updates of a video, and `record_video` is the only path in the loop that boots a second PyBoy
in the parent process and spawns ffmpeg. So: the video call is now wrapped in `try/except` and
logged, checkpoints carry optimizer state and the update counter with `--resume-from`, and
`env.close()` uses `pyboy.stop(save=False)` so a video no longer rewrites the `.ram` file that
the next boot reads. Runs launch under `nohup ... | tee runs/<name>.log`.
