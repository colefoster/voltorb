"""Live view of the current policy in a browser, at http://localhost:9876.

Runs its own env in a background thread, plays with whatever checkpoint is on disk, and
hot-reloads that checkpoint whenever it changes — so an open tab always shows the latest
policy without the training process knowing this exists. Costs one core.

    uv run python -m voltorb.tools.live --checkpoint runs/score-01/latest.pt

Frames go out as MJPEG (multipart/x-mixed-replace), which every browser plays natively with
no player, no HLS segmenting and no WebRTC handshake. A 160x144 Game Boy screen is small
enough that MJPEG's inefficiency does not matter, and the page upscales with nearest-
neighbour in CSS so the wire stays tiny.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.distributions import Categorical

from voltorb.env import EnvConfig, PinballEnv
from voltorb.train.ppo import ActorCritic

BOUNDARY = "voltorbframe"

PAGE = """<!doctype html>
<title>voltorb — live</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0d12; color:#c9d1d9; font:14px ui-monospace,monospace;
         display:flex; gap:24px; align-items:flex-start; padding:24px; flex-wrap:wrap; }
  img { width:480px; height:432px; image-rendering:pixelated;
        border:1px solid #30363d; border-radius:6px; background:#000; }
  table { border-collapse:collapse; }
  th { text-align:left; padding:4px 16px 4px 0; color:#8b949e; font-weight:400; }
  td { text-align:right; font-variant-numeric:tabular-nums; }
  h1 { font-size:14px; margin:0 0 12px; color:#8b949e; font-weight:400; }
</style>
<div><img src="/stream" alt="live"></div>
<div>
  <h1>voltorb — live policy</h1>
  <table id="s"></table>
</div>
<script>
const FIELDS = [["checkpoint","checkpoint"],["reloads","reloads"],["stage","stage"],
  ["episode","episode"],["frame","frame"],["reward","reward (episode)"],
  ["score","score"],["balls_left","balls left"],["dex_caught","dex caught"],
  ["stage_id","game stage"],["fps","fps"]];
async function tick(){
  try {
    const s = await (await fetch("/stats")).json();
    document.getElementById("s").innerHTML = FIELDS
      .filter(([k]) => k in s)
      .map(([k,label]) => `<tr><th>${label}</th><td>${s[k]}</td></tr>`).join("");
  } catch (e) {}
  setTimeout(tick, 500);
}
tick();
</script>
"""


class Player:
    """Steps one env forever, keeping the newest JPEG and a stats snapshot available."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.stats: dict = {"checkpoint": "none", "reloads": 0, "stage": args.stage}
        self._ckpt_mtime = 0.0
        self._reloads = 0
        self.model = ActorCritic()
        self.model.eval()

    def _maybe_reload(self) -> None:
        path = Path(self.args.checkpoint)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime <= self._ckpt_mtime:
            return
        try:
            self.model.load_state_dict(torch.load(path, map_location="cpu"))
        except Exception:
            return  # mid-write; try again on the next poll
        self._ckpt_mtime = mtime
        self._reloads += 1

    @torch.no_grad()
    def run(self) -> None:
        env = PinballEnv(
            EnvConfig(
                rom_path=self.args.rom,
                frame_skip=self.args.frame_skip,
                stage=self.args.stage,
                render=True,
            )
        )
        episode = 0
        last_reload_check = 0.0
        fps_t0, fps_n, fps = time.time(), 0, 0.0
        try:
            while True:
                episode += 1
                obs, _ = env.reset()
                ep_reward, frame = 0.0, 0
                while True:
                    now = time.time()
                    if now - last_reload_check > 3.0:
                        self._maybe_reload()
                        last_reload_check = now

                    logits, _ = self.model(torch.as_tensor(obs, dtype=torch.float32))
                    action = int(Categorical(logits=logits).sample().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    frame += self.args.frame_skip

                    fps_n += 1
                    if now - fps_t0 >= 1.0:
                        fps, fps_n, fps_t0 = fps_n / (now - fps_t0), 0, now

                    if frame % self.args.every == 0:
                        img = Image.fromarray(
                            np.asarray(env.render(), dtype=np.uint8)[:, :, :3]
                        )
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        gw = env.gw
                        snapshot = {
                            "checkpoint": Path(self.args.checkpoint).name
                            if self._reloads
                            else "none (random weights)",
                            "reloads": self._reloads,
                            "stage": self.args.stage,
                            "episode": episode,
                            "frame": f"{frame:,}",
                            "reward": f"{ep_reward:.1f}",
                            "score": f"{gw.score:,}",
                            "balls_left": gw.balls_left,
                            "dex_caught": sum(1 for v in gw.pokedex if v == 2),
                            "stage_id": gw.current_stage,
                            "fps": f"{fps:,.0f}",
                        }
                        with self.lock:
                            self.jpeg = buf.getvalue()
                            self.stats = snapshot

                    if terminated or truncated:
                        break
        finally:
            env.close()


def make_handler(player: Player):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console clean for training output
            pass

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stats":
                with player.lock:
                    body = json.dumps(player.stats).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header(
                    "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last = None
                try:
                    while True:
                        with player.lock:
                            frame = player.jpeg
                        if frame is None or frame is last:
                            time.sleep(0.02)
                            continue
                        last = frame
                        self.wfile.write(
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(frame)}\r\n\r\n".encode()
                        )
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/score-01/latest.pt")
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--stage", default="score", choices=["survive", "score", "dex"])
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--port", type=int, default=9876)
    ap.add_argument(
        "--every", type=int, default=2, help="emit every Nth frame; raise to cut bandwidth"
    )
    args = ap.parse_args()

    player = Player(args)
    threading.Thread(target=player.run, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(player))
    print(f"live view -> http://localhost:{args.port}  (checkpoint {args.checkpoint})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
