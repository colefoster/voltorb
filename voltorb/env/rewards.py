"""Curriculum reward stages: survive -> score -> dex.

Each stage is a superset of the one before it, so a policy trained on an earlier stage
transfers its weights forward instead of starting over.
"""
from __future__ import annotations

import math

N_SPECIES = 151
PLAYFIELD_HEIGHT = 172.0  # max observed ball_y; see tools/validate.py


def _dex_caught(pokedex) -> int:
    # One byte per species in the wrapper's Pokedex: 2 == caught, 1 == seen.
    return sum(1 for v in pokedex if v == 2)


class Reward:
    def reset(self, raw, gw) -> None:
        self._prev_score = gw.score
        self._prev_caught = _dex_caught(gw.pokedex)
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
        new_species_bonus: float = 100.0,
        evolution_bonus: float = 20.0,
        catch_progress_bonus: float = 0.5,
        score_weight: float = 0.02,
        **kwargs,
    ):
        super().__init__(score_weight=score_weight, **kwargs)
        self.new_species_bonus = new_species_bonus
        self.evolution_bonus = evolution_bonus
        self.catch_progress_bonus = catch_progress_bonus

    def reset(self, raw, gw) -> None:
        super().reset(raw, gw)
        self._prev_progress = raw["catch_tiles_flipped"] + raw["mon_hits"]

    def step(self, raw, gw, *, ball_lost: bool) -> float:
        reward = super().step(raw, gw, ball_lost=ball_lost)

        caught = _dex_caught(gw.pokedex)
        if caught > self._prev_caught:
            reward += self.new_species_bonus * (caught - self._prev_caught)
            self._prev_caught = caught

        evolutions = gw.evolution_success_count
        if evolutions > self._prev_evolutions:
            reward += self.evolution_bonus * (evolutions - self._prev_evolutions)
            self._prev_evolutions = evolutions

        # Pay for progress *inside* catch mode, not for entering it. Validation showed a
        # random policy is already in a special mode 58% of the time, so an entry bonus is
        # free money the agent would farm instead of finishing a catch.
        progress = raw["catch_tiles_flipped"] + raw["mon_hits"]
        if progress > self._prev_progress:
            reward += self.catch_progress_bonus * (progress - self._prev_progress)
        self._prev_progress = progress

        return reward


_STAGES = {"survive": SurviveReward, "score": ScoreReward, "dex": DexReward}


def make_reward(stage: str, **kwargs) -> Reward:
    try:
        return _STAGES[stage](**kwargs)
    except KeyError:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(_STAGES)}") from None
