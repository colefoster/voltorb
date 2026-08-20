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
SPECIAL_MODE_CATCH = 0  # pyboy SpecialMode.CATCH; EVOLVE=1, STAGE_CHANGE=2
ADDR_STAGE_COLLISION_STATE = 0xD4AF
ADDR_POKEDEX = 0xD962  # 151 bytes, one bitfield per species

# What actually starts a catch attempt, measured over 20 random episodes:
#
#   0xD532 == 128 means catch mode is READY. It is 128 at the first frame of every episode,
#   drops to 0 the instant catch mode starts, and re-arms later in ~40% of episodes. Mean
#   0.56 of in-play frames are ready.
#
#   The trigger is the ball coming to rest in the saucer at ball position (124.0, 120.0) --
#   identical to the pixel in every catch-mode entry observed, on both the red and the blue
#   field. Of 31 saucer visits, all 16 that happened while ready started catch mode and all
#   15 that happened while not ready did not: perfect separation.
#
# So the chain is: hit one fixed spot while ready -> catch mode -> ~80% of attempts end in a
# catch, unaided. A random policy visits the saucer 1.55 times per episode and converts 0.8.
# Nothing else in the chain matters, which is why every reward shaped at the downstream end
# produced a null result.
ADDR_CATCH_READY = 0xD532
CATCH_READY_VALUE = 128

# WHAT ARMS IT, settled from the pret/pokepinball disassembly on 2026-08-19 and verified
# against the ROM. 0xD532 is `wIndicatorStates + 3`, and every site that writes it computes
# the same predicate:
#
#     HandleSecondaryRightAlleyTrigger_RedField / HandleThirdRightAlleyTrigger_RedField:
#         ld a, [wRightAlleyCount] ; cp $3 ; ret z      ; capped at 3
#         inc a ; ld [wRightAlleyCount], a
#         ld a, [wRightAlleyCount] ; cp $2 ; ret c      ; needs >= 2
#         ld a, $80 ; ld [wIndicatorStates + 3], a      ; ARM
#
# and `SetLeftAndRightAlleyArrowIndicatorStates_RedField` recomputes it from the same byte on
# every map move. So **catch-ready is exactly `wRightAlleyCount >= 2`**, and wRightAlleyCount
# is incremented by the ball passing the secondary/third right alley trigger *after* the main
# right alley trigger set wRightAlleyTrigger -- i.e. by shooting the right ramp. It is zeroed
# at stage init and consumed by catch mode.
#
# This is not a timer and it is not luck. The ledger's "the objective is hard-capped at ~1.4
# arms per game for any policy" was two errors stacked: the 0.42 re-arms/episode behind it was
# measured at n=12 and is **0.708 +- 0.075 at n=120**, and the quantity is not policy-invariant
# at all -- it is a count of ramp shots. Verified over 53,205 frames: arm transitions and
# wRightAlleyCount increments occur in lockstep, and the 4.7% of frames where the biconditional
# fails are all bookkeeping lag inside catch mode, which zeroes the count a few frames after it
# clears the indicator.
ADDR_RIGHT_ALLEY_TRIGGER = 0xD544
ADDR_RIGHT_ALLEY_COUNT = 0xD545  # 0..3; >=2 arms the saucer
ADDR_LEFT_ALLEY_TRIGGER = 0xD542
ADDR_LEFT_ALLEY_COUNT = 0xD543
SAUCER_X, SAUCER_Y = 124.0, 120.0

# 0xD586 (48 bytes, "tile illumination" in PyBoy's table) is NOT the upstream lit-target
# state it looks like. Measured: it is all zeros until special_mode_active goes 1, its bit
# count tracks catch_tiles_flipped, and its bits blink several times a second as an
# animation. It is downstream of the bottleneck and unusable as a reward signal.

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
    # dex_caught, not dex_caught_frac: /151 put one catch at 0.0066 against features of
    # order 0.5, an effectively dead input.
    "dex_caught",
    "target_already_caught",
    # The gate and the target. Without these the agent is asked to hit one specific spot
    # while blind both to where it is and to whether hitting it does anything.
    "catch_ready",
    "saucer_dx",
    "saucer_dy",
    "saucer_dist",
    # The arm gate, and the reason the saucer stages were blind. catch_ready is a *derived*
    # bit -- it is `right_alley_count >= 2`. Feeding only the bit tells the agent whether the
    # shot is live but never how close it is to making it live, so "get one more ramp" was an
    # unrepresentable intention. right_alley_trigger is the sequencing latch: the count only
    # increments on the secondary trigger when this is already set.
    "right_alley_count",
    "right_alley_trigger",
    "left_alley_count",
    "left_alley_trigger",
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
    "dex_caught": 8.0,
    "target_already_caught": 1.0,
    "catch_ready": 1.0,
    "saucer_dx": 128.0,
    "saucer_dy": 128.0,
    "saucer_dist": 180.0,
    "right_alley_count": 3.0,
    "right_alley_trigger": 1.0,
    "left_alley_count": 3.0,
    "left_alley_trigger": 1.0,
}


@dataclass
class EnvConfig:
    rom_path: str = "roms/pokemon_pinball.gbc"
    headless: bool = True
    render: bool = False  # keep the screen buffer live while headless, for video capture
    frame_skip: int = 1  # frame-level control; see the design spec
    max_frames: int = 60 * 60 * 30  # 30 min of game time, truncation backstop
    launch_grace_frames: int = 120  # frames the agent gets to press A before we do it
    stage: str = "dex"  # "survive" | "score" | "dex" | "saucer" | "catch" | "shot"
    # "catch" trains the catch sub-task in isolation: every episode starts already inside
    # catch mode, so the agent gets a dense stream of attempts instead of one incidental
    # catch per 20,000-frame game. Full-game stages leave these unused.
    catch_max_frames: int = 5_400  # 90s; the natural mode timer runs 1-2.5 min
    catch_launch_frames: int = 400  # frames to get the ball into play before forcing mode
    # "shot" trains the one thing the objective reduces to: getting the ball into the saucer.
    # Episodes start from a savestate of a real shot opportunity -- ball in play, stage 0,
    # saucer ready -- and last ~400 frames, which is the window tools/shotsearch.py measured
    # the shot to be decided in. Base rate 12.7%, the same shape as the catch sub-task that is
    # the only thing in this project that has ever trained.
    #
    # The states come from actual random play in this same env, so every observation field is
    # a real one. That is deliberate: catch-01 started episodes from a synthesised
    # start_catch_mode(unlimited_time=True), which left timer_active and timer_remaining
    # frozen at values that occur in 0% of real attempts -- two of twenty-six inputs wrong for
    # a whole run.
    # 1,200 frames (20s), chosen by measurement, not by the 400-frame window shotsearch used:
    # random hit rates are 2.5% at 400 frames, 8.5% at 700 and 12.5% +- 2.3% at 1,200. The last
    # matches the ~12% base rate that made the catch sub-task trainable, and mean episode length
    # is 625 frames -- still inside gamma=0.999's ~1,000-frame credit window, and ~30x shorter
    # than a full game.
    shot_max_frames: int = 1_200
    shot_pool_size: int = 48       # savestates held per worker
    shot_pool_gap: int = 400       # frames between snapshots, so the pool is not one rally
    shot_pool_refresh: int = 400   # episodes between pool rebuilds, to keep it from staling
    # Backward start-state curriculum. shot-01 showed the shot is not learned from a cold
    # start even at a 12.5% base rate, while tools/shotsearch.py showed a winning 90-frame
    # prefix reproduces at 0.379 -- open-loop control exists, closed-loop control was not
    # found. That is an exploration failure, so start the agent near the goal and walk it
    # back.
    #
    # States are harvested for free rather than searched for: play randomly with a ring of
    # savestates, and when the ball lands in the saucer, keep the snapshots from N frames
    # before it. Each observed visit yields one start state per level.
    shot_curriculum: bool = False
    shot_levels: tuple = (45, 90, 180, 360, 720)  # frames before an observed saucer visit
    shot_level_states: int = 16    # start states banked per level
    shot_ring_stride: int = 15     # frames between ring snapshots (176 KB each)
    shot_window: int = 50          # episodes of success history driving promotion
    shot_promote: float = 0.70
    shot_demote: float = 0.25
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
        self._shot_pool: list[io.BytesIO] = []
        self._level_pool: dict[int, list[io.BytesIO]] = {}
        self._level = 0
        self._level_history: list[float] = []
        self._episodes_since_pool = 0
        self._catch_entries = 0
        self._catch_idle_frames = self.CATCH_ENTRY_GAP_FRAMES
        self._saucer_visits = 0
        self._saucer_dwell = 0
        self._saucer_counted = False
        # Ramp shots that move the arm gate. Counted as increments of wRightAlleyCount rather
        # than 0->128 edges on catch_ready, because the count is what the game actually gates
        # on and it keeps accruing (to its cap of 3) after the gate is already open.
        self._alley_shots = 0
        self._arms = 0
        self._prev_alley_count = None
        self._prev_ready = None

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
        sdx, sdy = x - SAUCER_X, y - SAUCER_Y
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
            "catch_ready": float(mem[ADDR_CATCH_READY] == CATCH_READY_VALUE),
            "saucer_dx": sdx,
            "saucer_dy": sdy,
            "saucer_dist": (sdx * sdx + sdy * sdy) ** 0.5,
            "right_alley_count": mem[ADDR_RIGHT_ALLEY_COUNT],
            "right_alley_trigger": float(bool(mem[ADDR_RIGHT_ALLEY_TRIGGER])),
            "left_alley_count": mem[ADDR_LEFT_ALLEY_COUNT],
            "left_alley_trigger": float(bool(mem[ADDR_LEFT_ALLEY_TRIGGER])),
        }

    CATCH_ENTRY_GAP_FRAMES = 180
    LAUNCH_TOGGLE_FRAMES = 15  # half-period of the A toggle that produces a launch edge
    SAUCER_RADIUS = 2.0
    SAUCER_DWELL_FRAMES = 5  # the ball is held in the saucer, so a real visit is not a graze

    def _count_events(self, raw: dict[str, float]) -> None:
        """Track the two numbers the objective decomposes into: saucer visits, and how many
        of them became catch attempts.

        Catch-mode entries are debounced because special_mode_active flickers off mid-attempt
        (observed at frame 2,940 with 96s still on the clock), so a naive 0->1 edge counts one
        attempt many times.
        """
        in_catch = raw["special_mode_active"] > 0 and raw["special_mode"] == SPECIAL_MODE_CATCH
        if in_catch:
            if self._catch_idle_frames >= self.CATCH_ENTRY_GAP_FRAMES:
                self._catch_entries += 1
            self._catch_idle_frames = 0
        else:
            self._catch_idle_frames += self.config.frame_skip

        if raw["saucer_dist"] < self.SAUCER_RADIUS and not in_catch:
            self._saucer_dwell += self.config.frame_skip
            if self._saucer_dwell >= self.SAUCER_DWELL_FRAMES and not self._saucer_counted:
                self._saucer_visits += 1
                self._saucer_counted = True
        else:
            self._saucer_dwell = 0
            self._saucer_counted = False

        # Reward stages read the debounced count from here rather than re-deriving an edge:
        # special_mode_active flickers, and a bonus on the raw edge pays repeatedly for one
        # attempt. It is also mode-specific -- an unqualified edge pays for stage-change and
        # evolution modes, which is worth 100 per episode of free reward.
        count = raw["right_alley_count"]
        if self._prev_alley_count is not None and count > self._prev_alley_count:
            self._alley_shots += int(count - self._prev_alley_count)
        self._prev_alley_count = count

        ready = raw["catch_ready"]
        if self._prev_ready is not None and ready > self._prev_ready:
            self._arms += 1
        self._prev_ready = ready

        # For the flipper-cost term. Actions are HELD states, not presses, so a policy can sit
        # with a flipper up forever at no cost -- measured on alley-02: a flipper is up on
        # **99.6%** of frames and the policy switches between left and right every ~2.6 frames.
        # That is not timing a shot, it is holding the flippers and jittering.
        # Counted, not a boolean: a flat per-frame charge prices BOTH the same as one flipper,
        # so ACTION_BOTH strictly dominates ACTION_LEFT/ACTION_RIGHT -- same cost, more ball
        # contact. Measured on alley-long: p(left) = p(right) = 0.000 and the policy uses only
        # NONE and BOTH, which makes an aimed shot unrepresentable.
        raw["flipper_held"] = 2.0 if self._held == ACTION_BOTH else float(
            self._held != ACTION_NONE
        )
        raw["catch_entries"] = float(self._catch_entries)
        raw["saucer_visits"] = float(self._saucer_visits)
        raw["alley_shots"] = float(self._alley_shots)
        raw["arms"] = float(self._arms)

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

    def _launch_action(self, action: int) -> int:
        """Override the policy's action to toggle A while the ball is still in the plunger.

        Must run BEFORE the tick: applying it afterwards means the next step's action releases
        A before the emulator ever advances a frame with it down, so the press is invisible and
        the ball never launches -- which is exactly what happened to every fixed action.
        """
        keep_left = action in (ACTION_LEFT, ACTION_BOTH)
        if (self._frames // self.LAUNCH_TOGGLE_FRAMES) % 2:
            return ACTION_BOTH if keep_left else ACTION_RIGHT
        return ACTION_LEFT if keep_left else ACTION_NONE

    def step(self, action):
        action = int(action)
        if not self._launched and self._frames >= self.config.launch_grace_frames:
            action = self._launch_action(action)
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
        # stage stays 1, the ball sits frozen in the plunger, score stays 0.
        #
        # **Launching needs a press EDGE, not a held button.** Measured: with A held from
        # before the serve, every fixed action -- including ACTION_RIGHT and ACTION_BOTH,
        # which hold A forever -- sits at stage 1 with score 0 for 3,000 frames, while a
        # random policy launches at frame 40. So the fallback toggles A on a 15-frame duty
        # cycle instead of pressing it, and routes through _apply_action so `_held` stays
        # truthful. (Going through the emulator directly is what desynced them: a bare
        # pyboy.button("a", 5) queues a release 5 frames later while _apply_action still
        # believes A is down, which can leave the right flipper
        # released-but-believed-held for the rest of the episode -- a silent two-action
        # space. A bare button_press generates no edge at all when A is already held.)
        #
        # current_stage is the only reliable in-play signal. Ball position is NOT — the
        # coordinate bytes hold uninitialised garbage before launch (ball_y reads 152), so
        # testing ball_y > 0 latches instantly and disables this fallback entirely. That bug
        # made every greedy-policy recording an identical clip of a ball that never moved.
        if not self._launched and self.gw.current_stage == 0:
            self._launched = True

        raw = self._raw_state()
        self._count_events(raw)
        ball_lost = (
            self.gw.balls_left < prev_balls_left
            or self.gw.lost_ball_during_saver > prev_lost_during_saver
        )
        reward = self.reward_fn.step(raw, self.gw, ball_lost=ball_lost)

        if self.config.stage == "catch":
            # End as soon as the attempt resolves, so episodes stay short and every one is
            # a full catch attempt. Mode going inactive means it timed out or completed.
            # Deliberately NOT terminating on special_mode_active going false: it drops out
            # mid-attempt (observed at frame 2,940 with 96s still on the clock), which cut
            # episodes to a 1,320-frame median and gave a 0/30 catch rate.
            terminated = bool(
                self.gw.game_over or ball_lost or raw["dex_caught"] > self._episode_start_dex
            )
            truncated = self._frames >= self.config.catch_max_frames
        elif self.config.stage == "shot":
            # Success ends the episode, so the return is the shot and nothing else. Losing the
            # ball ends it too -- otherwise the agent could farm the proximity term across a
            # ball it has already lost.
            terminated = bool(self._saucer_visits > 0 or ball_lost or self.gw.game_over)
            truncated = self._frames >= self.config.shot_max_frames
        else:
            terminated = bool(self.gw.game_over)
            truncated = self._frames >= self.config.max_frames
        info = {}
        if terminated or truncated:
            if self.config.stage == "shot" and self.config.shot_curriculum:
                self._update_curriculum(self._saucer_visits > 0)
            info = {
                "score": self.gw.score,
                "frames": self._frames,
                "dex_caught": _dex_caught_count(self._dex_bytes()),
                "caught_in_session": self.gw.pokemon_caught_in_session,
                "evolutions": self.gw.evolution_success_count,
                # The decomposition the objective actually has: saucer visits x whether the
                # saucer was ready x ~80% in-mode conversion. Random gets 1.55 visits, 0.88
                # catch-mode entries and ~0.88 dex per episode. Judge on catch_entries;
                # dex/game is its noisy consequence.
                "saucer_visits": self._saucer_visits,
                "catch_entries": self._catch_entries,
                # The arm gate, now that it is known to be earned rather than granted.
                "alley_shots": self._alley_shots,
                "arms": self._arms,
                "shot_level": float(getattr(self, "_level", -1)),
                # Roulette slots are kept only as a diagnostic. They are NOT the catch-mode
                # trigger: measured entries happen with zero slots opened, and slot events
                # land thousands of frames away from mode entry.
                "slots_opened": self.gw.roulette_slots_opened,
                "slots_entered": self.gw.roulette_slots_entered,
            }
        return self._observe(raw), reward, terminated, truncated, info

    def _build_backward_pool(self) -> None:
        """Harvest start states at fixed distances before observed saucer visits.

        No search: a random policy reaches the saucer often enough that keeping a ring of
        savestates and looking backwards from each visit yields one start state per level per
        visit, at the cost of the play itself.
        """
        levels = list(self.config.shot_levels)
        self._level_pool = {lv: [] for lv in levels}
        stride = self.config.shot_ring_stride
        ring: list[tuple[int, io.BytesIO]] = []
        ring_span = max(levels) + stride * 2

        self._boot_state.seek(0)
        self.pyboy.load_state(self._boot_state)
        self.gw.reset_tracking()
        self._held = ACTION_NONE
        self._frames = 0
        self._launched = False
        dwell = 0
        counted = False
        f = 0
        budget = 400_000
        while f < budget and any(
            len(v) < self.config.shot_level_states for v in self._level_pool.values()
        ):
            action = int(self.np_random.integers(0, N_ACTIONS))
            if not self._launched and self._frames >= self.config.launch_grace_frames:
                action = self._launch_action(action)
            self._apply_action(action)
            self.pyboy.tick(1, self.config.render, False)
            self._frames += 1
            f += 1
            if not self._launched and self.gw.current_stage == 0:
                self._launched = True
            if self.gw.game_over:
                self._boot_state.seek(0)
                self.pyboy.load_state(self._boot_state)
                self.gw.reset_tracking()
                self._held = ACTION_NONE
                self._frames = 0
                self._launched = False
                ring.clear()
                dwell, counted = 0, False
                continue

            if f % stride == 0:
                snap = io.BytesIO()
                self.pyboy.save_state(snap)
                ring.append((f, snap))
                while ring and f - ring[0][0] > ring_span:
                    ring.pop(0)

            mem = self.pyboy.memory
            x = (mem[ADDR_BALL_X] | (mem[ADDR_BALL_X + 1] << 8)) / 256.0
            y = (mem[ADDR_BALL_Y] | (mem[ADDR_BALL_Y + 1] << 8)) / 256.0
            at_saucer = ((x - SAUCER_X) ** 2 + (y - SAUCER_Y) ** 2) ** 0.5 < self.SAUCER_RADIUS
            if at_saucer and mem[0xD54B] == 0:
                dwell += 1
                if dwell >= self.SAUCER_DWELL_FRAMES and not counted:
                    counted = True
                    for lv in levels:
                        if len(self._level_pool[lv]) >= self.config.shot_level_states:
                            continue
                        # nearest ring entry to `lv` frames back
                        best = min(ring, key=lambda e: abs((f - e[0]) - lv), default=None)
                        if best is not None and abs((f - best[0]) - lv) <= stride:
                            self._level_pool[lv].append(best[1])
            else:
                dwell = 0
                counted = False

        self._level = 0
        self._level_history = []
        self._episodes_since_pool = 0

    def _load_curriculum_state(self) -> None:
        if not getattr(self, "_level_pool", None):
            self._build_backward_pool()
        levels = list(self.config.shot_levels)
        # Fall back to the nearest stocked level if the harvest came up short for this one.
        order = sorted(range(len(levels)), key=lambda i: abs(i - self._level))
        idx = next((i for i in order if self._level_pool[levels[i]]), None)
        if idx is None:
            self._load_shot_state()
            return
        pool = self._level_pool[levels[idx]]
        snap = pool[int(self.np_random.integers(0, len(pool)))]
        snap.seek(0)
        self.pyboy.load_state(snap)
        self.pyboy.button_release(LEFT_FLIPPER_BUTTON)
        self.pyboy.button_release(RIGHT_FLIPPER_BUTTON)
        self._held = ACTION_NONE
        self._launched = True

    def _update_curriculum(self, hit: bool) -> None:
        """Promote to a start further from the goal once the current one is reliable."""
        self._level_history.append(1.0 if hit else 0.0)
        if len(self._level_history) < self.config.shot_window:
            return
        rate = sum(self._level_history) / len(self._level_history)
        if rate >= self.config.shot_promote and self._level < len(self.config.shot_levels) - 1:
            self._level += 1
            self._level_history = []
        elif rate <= self.config.shot_demote and self._level > 0:
            self._level -= 1
            self._level_history = []
        elif len(self._level_history) >= self.config.shot_window * 2:
            self._level_history = self._level_history[-self.config.shot_window :]

    def _is_shot_opportunity(self) -> bool:
        """Ball in play, on the screen that has the saucer, with the saucer ready."""
        return (
            self.gw.current_stage == 0
            and self.pyboy.memory[ADDR_CATCH_READY] == CATCH_READY_VALUE
            and not self.gw.game_over
        )

    def _build_shot_pool(self) -> None:
        """Collect savestates of real shot opportunities by playing randomly.

        Only ~17% of frames qualify (stage 0 is 31% of frames, ready ~56%), and snapshots are
        spaced by shot_pool_gap so the pool is a spread of situations rather than one rally
        sampled forty-eight times.
        """
        self._shot_pool = []
        self._boot_state.seek(0)
        self.pyboy.load_state(self._boot_state)
        self.gw.reset_tracking()
        self._held = ACTION_NONE
        self._frames = 0
        self._launched = False
        since = self.config.shot_pool_gap
        budget = self.config.shot_pool_size * self.config.shot_pool_gap * 8
        for _ in range(budget):
            if len(self._shot_pool) >= self.config.shot_pool_size:
                break
            action = int(self.np_random.integers(0, N_ACTIONS))
            if not self._launched and self._frames >= self.config.launch_grace_frames:
                action = self._launch_action(action)
            self._apply_action(action)
            self.pyboy.tick(1, self.config.render, False)
            self._frames += 1
            if not self._launched and self.gw.current_stage == 0:
                self._launched = True
            since += 1
            if self.gw.game_over:
                self._boot_state.seek(0)
                self.pyboy.load_state(self._boot_state)
                self.gw.reset_tracking()
                self._held = ACTION_NONE
                self._frames = 0
                self._launched = False
                continue
            if since >= self.config.shot_pool_gap and self._is_shot_opportunity():
                snap = io.BytesIO()
                self.pyboy.save_state(snap)
                self._shot_pool.append(snap)
                since = 0
        self._episodes_since_pool = 0

    def _load_shot_state(self) -> None:
        if not self._shot_pool or self._episodes_since_pool >= self.config.shot_pool_refresh:
            self._build_shot_pool()
        snap = self._shot_pool[int(self.np_random.integers(0, len(self._shot_pool)))]
        snap.seek(0)
        self.pyboy.load_state(snap)
        self.pyboy.button_release(LEFT_FLIPPER_BUTTON)
        self.pyboy.button_release(RIGHT_FLIPPER_BUTTON)
        self._held = ACTION_NONE
        self._launched = True
        self._episodes_since_pool += 1

    def _force_catch_mode(self) -> None:
        """Put the ball in play, then drop straight into a catch attempt for a random
        species. Randomised so the policy learns the mechanic rather than one target."""
        from pyboy.plugins.game_wrapper_pokemon_pinball import Pokemon

        self.pyboy.button("a", 5)
        for _ in range(self.config.catch_launch_frames):
            self.pyboy.tick(1, self.config.render, False)
            if self.gw.current_stage == 0:
                break
        self._launched = True
        species = list(Pokemon)[self.np_random.integers(0, len(list(Pokemon)))]
        # unlimited_time: the natural 120s timer ends the attempt long before a learning
        # policy could finish one, and special_mode_active flickers off mid-attempt anyway.
        # Give the sub-task room; the timer can come back once catches are reliable.
        self.gw.start_catch_mode(pokemon=species, unlimited_time=True)
        self.pyboy.tick(10, self.config.render, False)
        self._catch_mode_seen = bool(self.gw.special_mode_active)

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
        self._catch_mode_seen = False
        if self.config.stage == "shot":
            if self.config.shot_curriculum:
                self._load_curriculum_state()
            else:
                self._load_shot_state()
        self._catch_entries = 0
        self._catch_idle_frames = self.CATCH_ENTRY_GAP_FRAMES
        self._saucer_visits = 0
        self._saucer_dwell = 0
        self._saucer_counted = False
        self._alley_shots = 0
        self._arms = 0
        self._prev_alley_count = None
        self._prev_ready = None
        if self.config.stage == "catch":
            # WARNING: this path passes unlimited_time=True, which freezes timer_active and
            # timer_remaining at values that occur in 0% of real catch attempts -- two of the
            # observation's inputs wrong for the whole of catch-01. It is quarantined to the
            # "catch" stage on purpose. Do not reuse it for a new sub-task without either
            # dropping unlimited_time or masking those two fields.
            self._force_catch_mode()
        raw = self._raw_state()
        self._episode_start_dex = raw["dex_caught"]
        self.reward_fn.reset(raw, self.gw)
        return self._observe(raw), {}

    def render(self):
        return np.asarray(self.pyboy.screen.ndarray)

    def close(self):
        # save=False: pyboy.stop() otherwise writes cartridge SRAM to <rom>.ram, which is
        # what every later PyBoy boots from. Each closed env (every video capture opens and
        # closes one) would mutate the boot state of the next run. Episodes already start
        # from an in-memory savestate with a zeroed Pokedex, so there is nothing to keep.
        self.pyboy.stop(save=False)
