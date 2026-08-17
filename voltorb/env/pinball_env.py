"""Voltorb's Pokemon Pinball environment: RAM-only observations, frame-level flipper control.

Derived from NicoleFaye/pokemon-pinball-gym (MIT). Flattened to a single module because
Voltorb replaces its observation, reward, and info layers wholesale; what survives is the
PyBoy plumbing, the launch sequence, and the episode-termination logic. See NOTICE.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pyboy import PyBoy

CARTRIDGE_TITLE = "POKEPINBALLVPH"
N_SPECIES = 151

# Ball position is a 16-bit fixed-point value: high byte = pixel, low byte = subpixel.
# PyBoy 2.7.0's game_wrapper.ball_x / .ball_y return only the LOW byte, i.e. the fractional
# part — verified identical to (raw16 & 0xFF) on 20k consecutive frames. Feeding that to a
# policy is feeding it noise, so read the full word here and keep the wrapper's value out
# of the observation entirely.
ADDR_BALL_X = 0xD4B3
ADDR_BALL_Y = 0xD4B5

# Addresses the PyBoy wrapper reads but does not surface as attributes.
ADDR_POKEMON_TO_CATCH = 0xD579
ADDR_RARE_POKEMON_FLAG = 0xD55B
ADDR_NUM_CATCH_TILES_FLIPPED = 0xD5B6
ADDR_NUM_MON_HITS = 0xD5C0
ADDR_TIMER_SECONDS = 0xD57A
ADDR_TIMER_MINUTES = 0xD57B
ADDR_TIMER_ACTIVE = 0xD57D
ADDR_SPECIAL_MODE_STATE = 0xD54D
ADDR_STAGE_COLLISION_STATE = 0xD4AF
ADDR_POKEDEX = 0xD962  # 151 bytes, one bitfield per species

# Four actions: the flipper state to hold for this frame. Unlike press/release action
# schemes this is stateless — the agent re-declares intent every frame, so "hold to trap"
# is expressible without the policy tracking what it pressed last.
ACTION_NONE, ACTION_LEFT, ACTION_RIGHT, ACTION_BOTH = 0, 1, 2, 3
N_ACTIONS = 4

LEFT_FLIPPER_BUTTON = "left"
RIGHT_FLIPPER_BUTTON = "a"  # doubles as the ball launcher

# Observation layout. Kept as an explicit ordered list so tools/validate.py can check every
# field against the emulator and so a saved policy's inputs stay traceable.
OBS_FIELDS: tuple[str, ...] = (
    "ball_x",
    "ball_y",
    "ball_dx",  # derived from position delta; wrapper velocities may be unsigned
    "ball_dy",
    "ball_vx_raw",
    "ball_vy_raw",
    "ball_type",
    "ball_size",
    "balls_left",
    "multiplier",
    "saver_seconds_left",
    "pikachu_saver_charge",
    "current_stage",
    "current_map",
    "stage_collision_state",
    "special_mode",
    "special_mode_active",
    "special_mode_state",
    "timer_remaining",
    "timer_active",
    "pokemon_to_catch",
    "rare_pokemon_flag",
    "catch_tiles_flipped",
    "mon_hits",
    "dex_caught_frac",
    "target_already_caught",
)
OBS_DIM = len(OBS_FIELDS)

# Divisors chosen to land each field roughly in [-1, 1]; values are clipped, so a wrong
# guess costs resolution rather than blowing up the policy input.
_SCALES: dict[str, float] = {
    "ball_x": 255.0,
    "ball_y": 255.0,
    "ball_dx": 8.0,
    "ball_dy": 8.0,
    "ball_vx_raw": 255.0,
    "ball_vy_raw": 255.0,
    "ball_type": 3.0,
    "ball_size": 4.0,
    "balls_left": 3.0,
    "multiplier": 5.0,
    "saver_seconds_left": 60.0,
    "pikachu_saver_charge": 15.0,
    "current_stage": 16.0,
    "current_map": 16.0,
    "stage_collision_state": 255.0,
    "special_mode": 4.0,
    "special_mode_active": 1.0,
    "special_mode_state": 8.0,
    "timer_remaining": 180.0,
    "timer_active": 1.0,
    "pokemon_to_catch": float(N_SPECIES),
    "rare_pokemon_flag": 1.0,
    "catch_tiles_flipped": 24.0,
    "mon_hits": 8.0,
    "dex_caught_frac": 1.0,
    "target_already_caught": 1.0,
}


@dataclass
class EnvConfig:
    rom_path: str = "roms/pokemon_pinball.gbc"
    headless: bool = True
    render: bool = False  # keep the screen buffer live while headless, for video capture
    frame_skip: int = 1  # frame-level control; see the design spec
    max_frames: int = 60 * 60 * 30  # 30 min of game time, truncation backstop
    launch_grace_frames: int = 120  # frames the agent gets to press A before we do it
    stage: str = "dex"  # curriculum stage: "survive" | "score" | "dex"
    reward_kwargs: dict = field(default_factory=dict)


def _wrapped_delta(now: float, prev: float) -> float:
    """Pixel coordinates span 0..255 and wrap, so a naive subtraction reads a 1px move at
    the seam as a 255px jump. Take the shorter way around the circle instead."""
    return (now - prev + 128.0) % 256.0 - 128.0


def _dex_caught_count(pokedex) -> int:
    """Count caught species from a slice of Pokedex bytes.

    The per-species byte is a bitfield, not an enum: bit 0 = seen, bit 1 = caught. So a
    caught species reads 2 if it was never "seen" first and 3 in the normal case where it
    was. PyBoy's own has_pokemon() tests `== 2` and therefore misses most real catches --
    observed directly: index 12 went 1 -> 3 at the exact frame the catch counter
    incremented, while index 0 went 0 -> 2.
    """
    return sum(1 for v in pokedex if v & 2)


class PinballEnv(gym.Env):
    """One game (3 balls) per episode, from a fresh save so the Pokedex starts empty."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: EnvConfig | dict | None = None, reward_fn=None):
        super().__init__()
        if config is None:
            config = EnvConfig()
        elif isinstance(config, dict):
            config = EnvConfig(**config)
        self.config = config

        from .rewards import make_reward

        self.reward_fn = reward_fn if reward_fn is not None else make_reward(
            config.stage, **config.reward_kwargs
        )

        self.pyboy = PyBoy(
            config.rom_path,
            window="null" if config.headless else "SDL2",
            sound_emulated=False,
        )
        if self.pyboy.cartridge_title != CARTRIDGE_TITLE:
            raise ValueError(
                f"expected {CARTRIDGE_TITLE!r} ROM, got {self.pyboy.cartridge_title!r}"
            )
        self.pyboy.set_emulation_speed(0)
        self.gw = self.pyboy.game_wrapper

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Boot once to the pre-launch state and keep it in memory. Reloading this is far
        # cheaper than rebooting, and gives an identical starting Pokedex every episode.
        self.gw.start_game()

        # Wipe the Pokedex before snapshotting. PyBoy persists cartridge SRAM to
        # <rom>.ram on exit and reloads it on boot, so without this every run inherits the
        # catches of every previous run -- measured at 8 species already caught at reset,
        # silently making "new species this episode" mean the wrong thing and shrinking the
        # objective over time.
        for i in range(N_SPECIES):
            self.pyboy.memory[ADDR_POKEDEX + i] = 0

        self._boot_state = io.BytesIO()
        self.pyboy.save_state(self._boot_state)

        self._held = ACTION_NONE
        self._prev_xy = (0, 0)
        self._frames = 0
        self._launched = False

    # ---- observation -------------------------------------------------------------

    def _dex_bytes(self):
        """Pokedex bytes read live from memory.

        NOT gw.pokedex: that list is refreshed by PyBoy's tick hooks, so immediately after
        a load_state() it still holds the previous episode's catches.
        """
        return self.pyboy.memory[ADDR_POKEDEX : ADDR_POKEDEX + N_SPECIES]

    def _ball_xy(self) -> tuple[float, float]:
        """Ball position in pixels, subpixel precision retained. See ADDR_BALL_X."""
        mem = self.pyboy.memory
        x16 = mem[ADDR_BALL_X] | (mem[ADDR_BALL_X + 1] << 8)
        y16 = mem[ADDR_BALL_Y] | (mem[ADDR_BALL_Y + 1] << 8)
        return x16 / 256.0, y16 / 256.0

    def _raw_state(self) -> dict[str, float]:
        gw, mem = self.gw, self.pyboy.memory
        x, y = self._ball_xy()
        px, py = self._prev_xy
        target = mem[ADDR_POKEMON_TO_CATCH]
        pokedex = self._dex_bytes()
        dex_caught = _dex_caught_count(pokedex)
        return {
            "ball_x": x,
            "ball_y": y,
            "ball_dx": _wrapped_delta(x, px),
            "ball_dy": _wrapped_delta(y, py),
            "ball_vx_raw": gw.ball_x_velocity,
            "ball_vy_raw": gw.ball_y_velocity,
            "ball_type": gw.ball_type,
            "ball_size": gw.ball_size,
            "balls_left": gw.balls_left,
            "multiplier": gw.multiplier,
            "saver_seconds_left": gw.ball_saver_seconds_left,
            "pikachu_saver_charge": gw.pikachu_saver_charge,
            "current_stage": gw.current_stage,
            "current_map": gw.current_map,
            "stage_collision_state": mem[ADDR_STAGE_COLLISION_STATE],
            "special_mode": gw.special_mode,
            "special_mode_active": float(bool(gw.special_mode_active)),
            "special_mode_state": mem[ADDR_SPECIAL_MODE_STATE],
            "timer_remaining": mem[ADDR_TIMER_MINUTES] * 60 + mem[ADDR_TIMER_SECONDS],
            "timer_active": float(mem[ADDR_TIMER_ACTIVE] == 1),
            "pokemon_to_catch": target,
            "rare_pokemon_flag": float(bool(mem[ADDR_RARE_POKEMON_FLAG])),
            "catch_tiles_flipped": mem[ADDR_NUM_CATCH_TILES_FLIPPED],
            "mon_hits": mem[ADDR_NUM_MON_HITS],
            "dex_caught": dex_caught,
            "dex_caught_frac": dex_caught / N_SPECIES,
            "target_already_caught": float(
                0 < target <= N_SPECIES and bool(pokedex[target - 1] & 2)
            ),
        }

    def _observe(self, raw: dict[str, float]) -> np.ndarray:
        vec = np.fromiter(
            (raw[f] / _SCALES[f] for f in OBS_FIELDS), dtype=np.float32, count=OBS_DIM
        )
        return np.clip(vec, -1.0, 1.0, out=vec)

    # ---- gym api ----------------------------------------------------------------

    def _apply_action(self, action: int) -> None:
        if action == self._held:
            return
        want_left = action in (ACTION_LEFT, ACTION_BOTH)
        want_right = action in (ACTION_RIGHT, ACTION_BOTH)
        had_left = self._held in (ACTION_LEFT, ACTION_BOTH)
        had_right = self._held in (ACTION_RIGHT, ACTION_BOTH)
        if want_left != had_left:
            (self.pyboy.button_press if want_left else self.pyboy.button_release)(
                LEFT_FLIPPER_BUTTON
            )
        if want_right != had_right:
            (self.pyboy.button_press if want_right else self.pyboy.button_release)(
                RIGHT_FLIPPER_BUTTON
            )
        self._held = action

    def step(self, action):
        action = int(action)
        self._apply_action(action)

        prev_balls_left = self.gw.balls_left
        prev_lost_during_saver = self.gw.lost_ball_during_saver

        self._prev_xy = self._ball_xy()
        self.pyboy.tick(
            self.config.frame_skip,
            self.config.render or not self.config.headless,
            False,
        )
        self._frames += self.config.frame_skip

        # A also launches the ball, and until it is launched nothing in the game advances:
        # stage stays 1, the ball sits frozen in the plunger, score stays 0. If the policy
        # has not pressed A within the grace window, keep pressing it until the game starts.
        #
        # current_stage is the only reliable in-play signal. Ball position is NOT — the
        # coordinate bytes hold uninitialised garbage before launch (ball_y reads 152), so
        # testing ball_y > 0 latches instantly and disables this fallback entirely. That bug
        # made every greedy-policy recording an identical clip of a ball that never moved.
        if not self._launched:
            if self.gw.current_stage == 0:
                self._launched = True
            elif (
                self._frames >= self.config.launch_grace_frames
                and self._frames % 30 < self.config.frame_skip
            ):
                self.pyboy.button("a", 5)

        raw = self._raw_state()
        ball_lost = (
            self.gw.balls_left < prev_balls_left
            or self.gw.lost_ball_during_saver > prev_lost_during_saver
        )
        reward = self.reward_fn.step(raw, self.gw, ball_lost=ball_lost)

        terminated = bool(self.gw.game_over)
        truncated = self._frames >= self.config.max_frames
        info = {}
        if terminated or truncated:
            info = {
                "score": self.gw.score,
                "frames": self._frames,
                "dex_caught": _dex_caught_count(self._dex_bytes()),
                "caught_in_session": self.gw.pokemon_caught_in_session,
                "evolutions": self.gw.evolution_success_count,
            }
        return self._observe(raw), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._boot_state.seek(0)
        self.pyboy.load_state(self._boot_state)
        self.gw.reset_tracking()
        self.pyboy.button_release(LEFT_FLIPPER_BUTTON)
        self.pyboy.button_release(RIGHT_FLIPPER_BUTTON)
        self._held = ACTION_NONE
        self._prev_xy = self._ball_xy()
        self._frames = 0
        self._launched = False
        raw = self._raw_state()
        self.reward_fn.reset(raw, self.gw)
        return self._observe(raw), {}

    def render(self):
        return np.asarray(self.pyboy.screen.ndarray)

    def close(self):
        self.pyboy.stop()
