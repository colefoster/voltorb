"""Single-file PPO for Pokemon Pinball. CleanRL-shaped, kept deliberately readable.

Curriculum use — each stage warm-starts from the previous one's checkpoint:

    uv run python -m voltorb.train.ppo --stage survive --total-steps 20_000_000
    uv run python -m voltorb.train.ppo --stage score --init-from runs/<survive-run>/final.pt
    uv run python -m voltorb.train.ppo --stage dex   --init-from runs/<score-run>/final.pt

The `saucer` stage trains from scratch on purpose: it changes the observation, and the
survive/score warm start buys ball-holding that raises frames per episode without aiming at
anything. Deaths mid-run are cheap now -- `--resume-from runs/<name>/latest.pt`.
"""
from __future__ import annotations

import argparse
import functools
import subprocess
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from voltorb.env import OBS_DIM, EnvConfig, PinballEnv

N_ACTIONS = 4
SCREEN_H, SCREEN_W = 144, 160

# Terminal-info keys logged to TensorBoard. catch_entries is the headline: catch mode starts
# when the ball rests in the saucer at (124,120) while the saucer is ready, ~80% of attempts
# then produce a catch unaided, and every policy so far -- random included -- gets ~0.9
# entries per game. Still a rolling window, so tools/eval.py remains the only thing to trust
# for a verdict.
TERMINAL_KEYS = (
    "score",
    "dex_caught",
    "caught_in_session",
    "saucer_visits",
    "catch_entries",
    "slots_opened",
    "slots_entered",
)


def _env_thunk(
    rom: str, stage: str, frame_skip: int, seed: int, reward_kwargs: dict | None = None
) -> PinballEnv:
    # Module-level so it survives pickling into spawned worker processes.
    env = PinballEnv(
        EnvConfig(
            rom_path=rom, frame_skip=frame_skip, stage=stage,
            reward_kwargs=dict(reward_kwargs or {}),
        )
    )
    env.reset(seed=seed)
    return env


class ActorCritic(nn.Module):
    """A 30-float observation does not need depth. Two hidden layers is plenty, and keeps
    the forward pass cheap enough that env stepping stays the bottleneck."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        self.apply(self._init)
        # Small actor gain keeps the initial policy near-uniform, so early exploration is
        # not biased toward whichever flipper the random init happened to favour.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def act(self, obs: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


@torch.no_grad()
def record_video(model: ActorCritic, args, path: Path, device, max_frames: int = 30_000) -> None:
    """One episode piped straight into ffmpeg. Every 4th frame, so a ~6-minute game becomes
    a watchable clip.

    Samples from the policy rather than taking the argmax. The env is fully deterministic
    (savestate reload, no stochastic reset), so a greedy policy replays a bit-identical
    trajectory every time — the first attempt at this produced 20 clips of which only 4 were
    unique and three sampled 20M steps apart were byte-identical. Sampling also shows the
    behaviour policy that is actually being trained, not an argmax the agent never uses.
    """
    env = PinballEnv(
        EnvConfig(rom_path=args.rom, frame_skip=args.frame_skip, stage=args.stage, render=True)
    )
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{SCREEN_W}x{SCREEN_H}", "-r", "30",
            "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    obs, _ = env.reset()
    try:
        for i in range(max_frames):
            logits, _ = model(torch.as_tensor(obs, dtype=torch.float32, device=device))
            action = int(Categorical(logits=logits).sample().item())
            obs, _, terminated, truncated, _ = env.step(action)
            if i % 4 == 0:
                proc.stdin.write(np.ascontiguousarray(env.render(), dtype=np.uint8).tobytes())
            if terminated or truncated:
                break
    finally:
        proc.stdin.close()
        proc.wait()
        env.close()


def save_checkpoint(model, optimizer, update: int, path) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update},
        path,
    )


def load_checkpoint(path, map_location="cpu") -> dict:
    """Read a checkpoint in either format.

    Runs up to dex-03 saved a bare state_dict; newer ones save a dict with the optimizer
    state and update counter so a mid-flight death costs minutes instead of the run.
    """
    blob = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(blob, dict) and "model" in blob:
        return blob
    return {"model": blob, "optimizer": None, "update": 0}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage", default="survive", choices=["survive", "score", "dex", "saucer", "catch", "shot"]
    )
    p.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    p.add_argument("--run-name", default=None)
    p.add_argument("--init-from", default=None, help="checkpoint to warm-start from")
    p.add_argument(
        "--resume-from",
        default=None,
        help="checkpoint to resume: restores optimizer state and update counter too",
    )
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--num-envs", type=int, default=8, help="8 is peak measured efficiency")
    p.add_argument("--num-steps", type=int, default=512, help="rollout length per env")
    p.add_argument("--frame-skip", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.999, help="high: a ball lasts ~4k frames")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    # 0.01 keeps the policy pinned at max entropy: the survive-stage advantage signal is
    # too weak to beat it. Swept 0.01/0.001/0.0 -- 0.001 moves the policy and survives
    # longest. See data/runs.md.
    p.add_argument("--ent-coef", type=float, default=0.001)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--device", default="cpu", help="cpu beats mps for a net this small")
    p.add_argument("--seed", type=int, default=1)
    # saucer stage only. "raw" pays proximity every frame like the ball-height term;
    # "potential" is the policy-invariant form, which saucer-01 showed does not make it aim.
    p.add_argument("--saucer-shaping", default="potential", choices=["potential", "raw"])
    p.add_argument("--saucer-weight", type=float, default=None)
    p.add_argument("--video-every", type=int, default=25, help="updates between videos; 0=off")
    p.add_argument("--save-every", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    run_name = args.run_name or f"{args.stage}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("runs") / run_name
    (run_dir / "videos").mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    writer.add_text("args", "\n".join(f"{k}={v}" for k, v in vars(args).items()))

    reward_kwargs: dict = {}
    if args.stage == "saucer":
        reward_kwargs["saucer_shaping"] = args.saucer_shaping
        if args.saucer_weight is not None:
            reward_kwargs["saucer_weight"] = args.saucer_weight

    envs = gym.vector.AsyncVectorEnv(
        [
            functools.partial(
                _env_thunk, args.rom, args.stage, args.frame_skip, args.seed + i, reward_kwargs
            )
            for i in range(args.num_envs)
        ],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)

    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    first_update = 1
    if args.init_from:
        model.load_state_dict(load_checkpoint(args.init_from, device)["model"])
        print(f"warm-started from {args.init_from}")
    if args.resume_from:
        ckpt = load_checkpoint(args.resume_from, device)
        model.load_state_dict(ckpt["model"])
        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        first_update = int(ckpt["update"]) + 1
        print(f"resumed from {args.resume_from} at update {first_update}")

    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_updates = args.total_steps // batch_size

    obs_buf = torch.zeros((args.num_steps, args.num_envs, OBS_DIM), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    global_step = (first_update - 1) * batch_size
    steps_this_run = 0
    start = time.time()
    # Episodes run 12k-24k frames, so completed-episode stats are rare. Keep a rolling
    # window rather than reporting per-update, which would mostly be empty.
    ep_returns: list[float] = []
    ep_lengths: list[float] = []
    ep_infos: list[dict] = []

    for update in range(first_update, num_updates + 1):
        if args.anneal_lr:
            for g in optimizer.param_groups:
                g["lr"] = args.lr * (1.0 - (update - 1.0) / num_updates)

        for step in range(args.num_steps):
            global_step += args.num_envs
            steps_this_run += args.num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logp, _, value = model.act(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value

            obs_np, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rew_buf[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            done = np.logical_or(terminated, truncated)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)

            if "episode" in infos:
                mask = infos["episode"].get("_r", infos.get("_episode", done))
                mask = np.asarray(mask, dtype=bool)
                ep_returns.extend(np.asarray(infos["episode"]["r"])[mask].tolist())
                ep_lengths.extend(np.asarray(infos["episode"]["l"])[mask].tolist())
            # Our env only emits info on terminal steps, and the vector wrapper files that
            # under final_info -- NOT at the top level. Reading the top level silently
            # logged nothing at all for a whole 50M-step run.
            final = infos.get("final_info")
            if final:
                for key in TERMINAL_KEYS:
                    if key not in final:
                        continue
                    vals = np.asarray(final[key], dtype=np.float64)
                    valid = np.asarray(final.get(f"_{key}", done), dtype=bool)
                    if valid.any():
                        ep_infos.append({key: float(vals[valid].mean())})

        with torch.no_grad():
            _, next_value = model(next_obs)
            advantages = torch.zeros_like(rew_buf)
            last_gae = torch.zeros(args.num_envs, device=device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - done_buf[t + 1]
                    next_values = val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * next_values * next_nonterminal - val_buf[t]
                last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
                advantages[t] = last_gae
            returns = advantages + val_buf

        b_obs = obs_buf.reshape(-1, OBS_DIM)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)

        indices = np.arange(batch_size)
        clipfracs = []
        for _ in range(args.update_epochs):
            np.random.shuffle(indices)
            for start_idx in range(0, batch_size, minibatch_size):
                mb = indices[start_idx : start_idx + minibatch_size]
                new_logp, entropy, new_value = model.evaluate(b_obs[mb], b_act[mb])
                log_ratio = new_logp - b_logp[mb]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                ).mean()

                v_clipped = b_val[mb] + torch.clamp(
                    new_value - b_val[mb], -args.clip_coef, args.clip_coef
                )
                v_loss = 0.5 * torch.max(
                    (new_value - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2
                ).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int(steps_this_run / (time.time() - start))
        writer.add_scalar("charts/steps_per_second", sps, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/reward_per_step", rew_buf.mean().item(), global_step)
        # If advantage_std collapses toward zero the reward is action-independent and no
        # policy gradient exists, no matter how long you train. Watch this before anything.
        writer.add_scalar("diag/advantage_std", b_adv.std().item(), global_step)
        writer.add_scalar("diag/advantage_absmean", b_adv.abs().mean().item(), global_step)
        writer.add_scalar("diag/return_mean", b_ret.mean().item(), global_step)
        var_y = b_ret.var().item()
        writer.add_scalar(
            "diag/explained_variance",
            float("nan") if var_y == 0 else 1.0 - (b_ret - b_val).var().item() / var_y,
            global_step,
        )

        msg = f"update {update}/{num_updates} step {global_step:,} sps {sps:,}"
        if ep_returns:
            window_r = float(np.mean(ep_returns[-20:]))
            window_l = float(np.mean(ep_lengths[-20:]))
            writer.add_scalar("charts/episodic_return", window_r, global_step)
            writer.add_scalar("charts/episodic_length", window_l, global_step)
            msg += f" | ep_return {window_r:.1f} ep_len {window_l:,.0f} (n={len(ep_returns)})"
        for key in TERMINAL_KEYS:
            vals = [d[key] for d in ep_infos[-20 * len(TERMINAL_KEYS) :] if key in d]
            if vals:
                writer.add_scalar(f"game/{key}", float(np.mean(vals)), global_step)
        print(msg, flush=True)

        if args.save_every and update % args.save_every == 0:
            save_checkpoint(model, optimizer, update, run_dir / "latest.pt")
        if args.video_every and update % args.video_every == 0:
            # Guarded because this is the only path in the loop that boots a second PyBoy in
            # the parent process and spawns ffmpeg, and three long runs died mid-flight --
            # two of them within 16 updates of a video, with no crash report and a disproven
            # memory-leak hypothesis. A missing clip must never cost a 30-minute run.
            try:
                record_video(model, args, run_dir / "videos" / f"update{update:05d}.mp4", device)
            except Exception as exc:  # noqa: BLE001 - deliberately broad; video is optional
                print(f"video capture failed at update {update}: {exc!r}", flush=True)
                writer.add_text("video_error", f"update {update}: {exc!r}", global_step)

    save_checkpoint(model, optimizer, num_updates, run_dir / "final.pt")
    envs.close()
    writer.close()
    print(f"done -> {run_dir}")


if __name__ == "__main__":
    main()
