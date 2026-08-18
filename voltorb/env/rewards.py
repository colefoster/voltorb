"""Curriculum reward stages: survive -> score -> dex -> saucer.

Each stage is a superset of the one before it, so a policy trained on an earlier stage
transfers its weights forward instead of starting over.
"""
from __future__ import annotations

import math

N_SPECIES = 151
PLAYFIELD_HEIGHT = 172.0  # max observed ball_y; see tools/validate.py


class Reward:
    def reset(self, raw, gw) -> None:
        self._prev_score = gw.score
        self._prev_caught = raw["dex_caught"]
        self._prev_evolutions = gw.evolution_success_count

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        raise NotImplementedError


class SurviveReward(Reward):
    """Stage 1: keep the ball alive.

    A constant per-frame alive bonus does NOT work here, and the first 50M-step run proved
    it: a reward identical in every state and for every action makes the critic learn a
    constant, advantages collapse to ~1% of return scale, and the entropy bonus pins the
    policy at uniform random forever. Measured: advantage_std 0.13 vs return_mean 8.15,
    explained_variance 0.955, clipfrac 0.000 for all 6,103 updates.

    So the signal has to vary with state. Ball height does: the ball is lost at the bottom,
    so "keep it high" is dense, action-sensitive, and points the same direction as survival.
    """

    def __init__(
        self,
        ball_lost_penalty: float = 1.0,
        height_weight: float = 0.01,
        alive_bonus: float = 0.0,
    ):
        self.ball_lost_penalty = ball_lost_penalty
        self.height_weight = height_weight
        self.alive_bonus = alive_bonus

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        height = 1.0 - min(raw["ball_y"], PLAYFIELD_HEIGHT) / PLAYFIELD_HEIGHT
        reward = self.alive_bonus + self.height_weight * height
        if ball_lost:
            reward -= self.ball_lost_penalty
        return reward


class ScoreReward(SurviveReward):
    """Stage 2: survival plus points. Score deltas are log-scaled — raw pinball payouts
    span several orders of magnitude and would otherwise swamp every other term."""

    def __init__(self, score_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.score_weight = score_weight

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        reward = super().step(raw, gw, ball_lost=ball_lost)
        delta = gw.score - self._prev_score
        self._prev_score = gw.score
        if delta > 0:
            reward += self.score_weight * math.log10(1.0 + delta)
        return reward


class DexReward(ScoreReward):
    """Stage 3: the real objective. Big payout for a species not already in the Pokedex,
    with score kept as a weak shaping term so the agent still has dense signal to follow.

    Deliberately rewards `dex_caught` deltas rather than the wrapper's
    `pokemon_caught_in_session` counter, so re-catching the same species pays nothing.
    """

    def __init__(
        self,
        new_species_bonus: float = 300.0,
        evolution_bonus: float = 50.0,
        catch_progress_bonus: float = 2.0,
        score_weight: float = 0.0,
        height_weight: float = 0.001,
        **kwargs,
    ):
        # dex-01 failed because the shaping outweighed the objective. Over a ~20,000-frame
        # episode the old height term paid 0.01 * ~0.5 * 20,000 ~= 100, exactly what ONE new
        # species was worth, and the score term added more on top. The agent optimised what
        # it was actually paid for: it survived (+3.3 sigma vs random) and caught nothing
        # (+1.0 sigma, indistinguishable). Height now pays ~10 per episode against 300 per
        # species, and score is off entirely -- stage 2 already showed score tracks bumper
        # luck more than skill.
        super().__init__(score_weight=score_weight, height_weight=height_weight, **kwargs)
        self.new_species_bonus = new_species_bonus
        self.evolution_bonus = evolution_bonus
        self.catch_progress_bonus = catch_progress_bonus

    def reset(self, raw, gw) -> None:
        super().reset(raw, gw)
        self._prev_progress = raw["catch_tiles_flipped"] + raw["mon_hits"]

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        reward = super().step(raw, gw, ball_lost=ball_lost)

        caught = raw["dex_caught"]
        if caught > self._prev_caught:
            reward += self.new_species_bonus * (caught - self._prev_caught)
            self._prev_caught = caught

        evolutions = gw.evolution_success_count
        if evolutions > self._prev_evolutions:
            reward += self.evolution_bonus * (evolutions - self._prev_evolutions)
            self._prev_evolutions = evolutions

        # Pay for progress *inside* catch mode. NOTE: the original justification for never
        # paying for mode ENTRY was wrong -- "a random policy is in a special mode ~32-58%
        # of frames" is a duration statistic, one ~5,000-frame attempt inside a ~17,000-frame
        # game. Entries happen ~1.0 times per game for every policy including random, and
        # that is the actual bottleneck. See SaucerReward.
        progress = raw["catch_tiles_flipped"] + raw["mon_hits"]
        if progress > self._prev_progress:
            reward += self.catch_progress_bonus * (progress - self._prev_progress)
        self._prev_progress = progress

        return reward


class SaucerReward(DexReward):
    """Stage 4: pay for aiming at the one spot that starts a catch attempt.

    Measured (see pinball_env.ADDR_CATCH_READY): catch mode starts when the ball comes to
    rest in the saucer at (124, 120) while 0xD532 == 128, which it is at the start of every
    episode. A random policy gets 1.55 saucer visits per episode, 0.88 of them while ready,
    and ~80% of those become a catch with no further help. So the entire objective reduces to
    one aiming problem, and for the first time there is a signal available on every frame:
    distance from the ball to a fixed target.

    The distance term is potential-based -- gamma * phi(s') - phi(s) with phi = -dist -- so it
    is policy-invariant: it cannot be farmed by hovering near the saucer without entering it,
    which a raw -distance-per-frame term would pay for indefinitely. The height term from the
    earlier stages stays tiny and only to keep the ball alive long enough to aim.
    """

    def __init__(
        self,
        saucer_weight: float = 2.0,
        saucer_shaping: str = "potential",  # "potential" | "raw"
        catch_mode_bonus: float = 100.0,
        gamma: float = 0.999,
        catch_progress_bonus: float = 0.0,
        height_weight: float = 0.001,
        **kwargs,
    ):
        super().__init__(
            catch_progress_bonus=catch_progress_bonus, height_weight=height_weight, **kwargs
        )
        if saucer_shaping not in ("potential", "raw"):
            raise ValueError(f"saucer_shaping must be 'potential' or 'raw', got {saucer_shaping!r}")
        self.saucer_weight = saucer_weight
        self.saucer_shaping = saucer_shaping
        self.catch_mode_bonus = catch_mode_bonus
        self.gamma = gamma

    @staticmethod
    def _potential(raw) -> float:
        """-normalised distance to the saucer, ungated on purpose.

        Gating this on catch_ready / in-play / not-in-mode was tried and is worse than
        useless: every time a gate flips, phi jumps to 0, and since phi is otherwise negative
        that hands out up to +2 reward for *losing the ball* -- twice the ball-lost penalty.
        Ungated, the only discontinuities are real teleports (the ball being served), which
        are bounded and rare. The policy sees catch_ready in the observation, so it can learn
        for itself when the shot is worth taking.
        """
        return -min(raw["saucer_dist"], 180.0) / 180.0

    def reset(self, raw, gw) -> None:
        super().reset(raw, gw)
        self._prev_potential = self._potential(raw)
        self._prev_entries = raw.get("catch_entries", 0.0)

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        reward = super().step(raw, gw, ball_lost=ball_lost)

        potential = self._potential(raw)
        if self.saucer_shaping == "raw":
            # Proximity paid per frame, exactly like the ball-height term that is the only
            # shaping this project has ever gotten to work. saucer-01 showed the
            # potential-based form moves catch entries a little and saucer visits not at all.
            #
            # Gated on stage 0, and that gate is the point. current_stage indexes two
            # different screens: measured over 121,752 frames, the ball is at (124,120) in
            # 1,437 frames of stage 0 and *zero* frames of stage 1, and all measured saucer
            # visits and catch-mode entries are stage 0 -- but stage 1 is 69% of all frames.
            # Ungated (saucer-01 and saucer-02, both null on visits), this term spent most of
            # its budget paying for proximity to a coordinate on the wrong screen. Gating a
            # raw term is safe; gating the potential-based form is not, because every gate
            # flip becomes a reward spike.
            if raw["current_stage"] == 0:
                reward += self.saucer_weight * (1.0 + potential)
        else:
            reward += self.saucer_weight * (self.gamma * potential - self._prev_potential)
        self._prev_potential = potential

        entries = raw.get("catch_entries", 0.0)
        if entries > self._prev_entries:
            reward += self.catch_mode_bonus * (entries - self._prev_entries)
            self._prev_entries = entries

        return reward


class CatchReward(Reward):
    """Sub-task stage: the episode begins inside a catch attempt and ends when it resolves.

    Exists because dex-01 and dex-02 both came back statistically indistinguishable from
    random. A catch spans several thousand frames (enter mode, flip ~6 catch tiles, hit the
    target ~6 times, then 3-4 more) while gamma=0.999 only reaches ~1,000 frames back, so
    the payout never reaches the actions that earned it. Here the whole episode IS the
    catch, so the horizon covers it and every episode carries signal.
    """

    def __init__(
        self,
        catch_bonus: float = 100.0,
        progress_bonus: float = 5.0,
        ball_lost_penalty: float = 10.0,
        height_weight: float = 0.002,
        time_penalty: float = 0.002,
    ):
        self.catch_bonus = catch_bonus
        self.progress_bonus = progress_bonus
        self.ball_lost_penalty = ball_lost_penalty
        self.height_weight = height_weight
        self.time_penalty = time_penalty

    def reset(self, raw, gw) -> None:
        super().reset(raw, gw)
        self._prev_progress = raw["catch_tiles_flipped"] + raw["mon_hits"]

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        height = 1.0 - min(raw["ball_y"], PLAYFIELD_HEIGHT) / PLAYFIELD_HEIGHT
        reward = self.height_weight * height - self.time_penalty

        progress = raw["catch_tiles_flipped"] + raw["mon_hits"]
        if progress > self._prev_progress:
            reward += self.progress_bonus * (progress - self._prev_progress)
        self._prev_progress = progress

        caught = raw["dex_caught"]
        if caught > self._prev_caught:
            reward += self.catch_bonus * (caught - self._prev_caught)
            self._prev_caught = caught

        if ball_lost:
            reward -= self.ball_lost_penalty
        return reward


class ShotReward(Reward):
    """Sub-task stage: one shot at the saucer, ~400 frames, and the episode ends when it
    resolves.

    The precedent is catch-01, the only run in this project that ever produced a real result
    (+3.2 sigma): it worked because the sub-task was short and its base rate was ~12%. This has
    the same shape -- tools/shotsearch.py measured a 12.7% base rate over 400 frames from real
    shot-opportunity states, and measured that the first 90 frames of action triple the hit
    rate, so the signal is inside the episode by construction rather than thousands of frames
    downstream of it.

    Deliberately narrow: no species bonus, no score, no catch-mode progress. Reaching the
    saucer while ready starts catch mode, and catch mode converts ~80% of the time unaided, so
    the shot is the whole objective and everything else is noise on a 400-frame episode.
    """

    def __init__(
        self,
        visit_bonus: float = 10.0,
        # ~1.0 per episode against 10 for the shot. dex-01 died of shaping worth as much as
        # the objective; keep the ratio at 10:1 and it can only break ties.
        proximity_weight: float = 0.003,
        ball_lost_penalty: float = 1.0,
        time_penalty: float = 0.0,
    ):
        self.visit_bonus = visit_bonus
        self.proximity_weight = proximity_weight
        self.ball_lost_penalty = ball_lost_penalty
        self.time_penalty = time_penalty

    def reset(self, raw, gw) -> None:
        super().reset(raw, gw)
        self._prev_visits = raw.get("saucer_visits", 0.0)

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        reward = -self.time_penalty
        # Gated on stage 0: current_stage indexes two different screens and the saucer only
        # exists on one of them. Ungated, this term spends most of its budget paying for
        # proximity to a coordinate on the wrong screen -- which is what made saucer-01 and
        # saucer-02 null.
        if raw["current_stage"] == 0:
            reward += self.proximity_weight * (1.0 - min(raw["saucer_dist"], 180.0) / 180.0)

        visits = raw.get("saucer_visits", 0.0)
        if visits > self._prev_visits:
            reward += self.visit_bonus * (visits - self._prev_visits)
            self._prev_visits = visits

        if ball_lost:
            reward -= self.ball_lost_penalty
        return reward


_STAGES = {
    "survive": SurviveReward,
    "score": ScoreReward,
    "dex": DexReward,
    "saucer": SaucerReward,
    "catch": CatchReward,
    "shot": ShotReward,
}


def make_reward(stage: str, **kwargs) -> Reward:
    try:
        return _STAGES[stage](**kwargs)
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGES)}") from None
