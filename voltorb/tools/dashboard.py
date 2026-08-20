"""One page that answers "is it learning?" and "what does it look like?" at the same time.

    uv run python -m voltorb.tools.dashboard          # -> http://localhost:9881

Left: every run in runs/, with the live one marked. Middle: its TensorBoard scalars, drawn
straight from the event files (no tensorboard server), with up to three runs overlaid so a
new run can be read against the one it is trying to beat. Right: a policy actually playing,
streamed as MJPEG, hot-reloading whatever checkpoint you point it at.

The player is lazy -- it only boots PyBoy while a browser is attached to /stream, and shuts
the env down ~20s after the last viewer leaves, so leaving this open during training costs
nothing.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributions import Categorical

from voltorb.env import EnvConfig, PinballEnv
from voltorb.train.ppo import ActorCritic, load_checkpoint

RUNS = Path("runs")
BOUNDARY = "voltorbframe"
STAGES = ["survive", "score", "dex", "saucer", "catch", "shot", "alley"]
MAX_POINTS = 900          # per series sent to the browser
ACTIVE_WINDOW = 120.0     # seconds since last event write for a run to count as live
IDLE_SHUTDOWN = 20.0      # seconds with no /stream viewer before the env is closed


# --------------------------------------------------------------------------- scalars

class Scalars:
    """Reads runs/*/events.out.tfevents.* on demand, and re-reads only what grew."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self._acc: dict[str, EventAccumulator] = {}
        self._checked: dict[str, float] = {}

    @staticmethod
    def run_dirs() -> list[Path]:
        if not RUNS.is_dir():
            return []
        return sorted(
            (d for d in RUNS.iterdir() if d.is_dir() and any(d.glob("events.out.tfevents.*"))),
            key=lambda d: d.name,
        )

    @staticmethod
    def mtime(run: Path) -> float:
        return max((f.stat().st_mtime for f in run.glob("events.out.tfevents.*")), default=0.0)

    def accumulator(self, name: str) -> EventAccumulator | None:
        run = RUNS / name
        if not run.is_dir():
            return None
        with self.lock:
            acc = self._acc.get(name)
            if acc is None:
                # size_guidance 0 == keep every point; these files are a few MB at most.
                acc = EventAccumulator(str(run), size_guidance={"scalars": 0, "tensors": 0})
                self._acc[name] = acc
                self._checked[name] = 0.0
            # Reload() is incremental, but it still walks the file, so rate-limit it.
            if time.time() - self._checked[name] > 2.0:
                acc.Reload()
                self._checked[name] = time.time()
            return acc

    def series(self, name: str) -> dict[str, dict]:
        acc = self.accumulator(name)
        if acc is None:
            return {}
        out: dict[str, dict] = {}
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            stride = max(1, len(events) // MAX_POINTS)
            picked = events[::stride]
            if picked and picked[-1] is not events[-1]:
                picked.append(events[-1])  # never drop the newest value
            out[tag] = {
                "steps": [e.step for e in picked],
                "values": [None if e.value != e.value else round(e.value, 6) for e in picked],
                "last": None if events[-1].value != events[-1].value else events[-1].value,
                "n": len(events),
            }
        return out

    def index(self) -> list[dict]:
        now = time.time()
        rows = []
        for run in self.run_dirs():
            mtime = self.mtime(run)
            steps = 0
            acc = self.accumulator(run.name)
            if acc is not None:
                tags = acc.Tags().get("scalars", [])
                probe = "charts/steps_per_second" if "charts/steps_per_second" in tags else (
                    tags[0] if tags else None
                )
                if probe:
                    ev = acc.Scalars(probe)
                    steps = ev[-1].step if ev else 0
            rows.append(
                {
                    "name": run.name,
                    "steps": steps,
                    "updated": mtime,
                    "age": now - mtime,
                    "active": (now - mtime) < ACTIVE_WINDOW,
                    "videos": sorted(p.name for p in (run / "videos").glob("*.mp4")),
                    "checkpoints": sorted(p.name for p in run.glob("*.pt")),
                }
            )
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return rows


# --------------------------------------------------------------------------- live play

class LivePlayer:
    """Steps one env while at least one browser is watching, and stops when none are."""

    def __init__(self, args) -> None:
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.rom = args.rom
        self.frame_skip = args.frame_skip
        self.every = args.every
        self.viewers = 0
        self.last_viewer = 0.0
        self.want = {"checkpoint": args.checkpoint, "stage": args.stage, "speed": args.speed}
        self.stats: dict = {"status": "idle"}
        self.load_error = ""
        self._loaded = ("", 0.0)
        self._reloads = 0
        self.model = ActorCritic()
        self.model.eval()

    def configure(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                if k in self.want and v is not None:
                    self.want[k] = v

    def viewer_join(self) -> None:
        with self.lock:
            self.viewers += 1
            self.last_viewer = time.time()

    def viewer_leave(self) -> None:
        with self.lock:
            self.viewers = max(0, self.viewers - 1)
            self.last_viewer = time.time()

    def _watching(self) -> bool:
        with self.lock:
            return self.viewers > 0 or (time.time() - self.last_viewer) < IDLE_SHUTDOWN

    def _maybe_reload(self) -> None:
        path = Path(self.want["checkpoint"])
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if (str(path), mtime) == self._loaded:
            return
        try:
            self.model.load_state_dict(load_checkpoint(path)["model"])
        except Exception as exc:  # noqa: BLE001
            # Usually a mid-write file (retried on the next poll), but an obs-layout change
            # makes an old checkpoint permanently unloadable -- say so instead of silently
            # playing random weights.
            self.load_error = f"{path.name}: {exc.__class__.__name__}"
            return
        self.load_error = ""
        self._loaded = (str(path), mtime)
        self._reloads += 1

    def run(self) -> None:
        while True:
            if not self._watching():
                with self.lock:
                    self.stats = {
                        "status": "idle",
                        "note": "no viewer — env not running",
                        "want": dict(self.want),
                    }
                time.sleep(0.5)
                continue
            try:
                self._play_stage()
            except Exception as exc:  # noqa: BLE001 - a dead viewer must not kill the server
                with self.lock:
                    self.stats = {"status": "error", "error": repr(exc)}
                time.sleep(2.0)

    @torch.no_grad()
    def _play_stage(self) -> None:
        stage = self.want["stage"]
        env = PinballEnv(
            EnvConfig(rom_path=self.rom, frame_skip=self.frame_skip, stage=stage, render=True)
        )
        episode = 0
        fps_t0, fps_n, fps = time.time(), 0, 0.0
        last_check = 0.0
        try:
            while self._watching() and self.want["stage"] == stage:
                episode += 1
                obs, _ = env.reset()
                ep_reward, frame = 0.0, 0
                deadline = time.time()
                while True:
                    now = time.time()
                    if now - last_check > 2.0:
                        self._maybe_reload()
                        last_check = now
                        if not self._watching() or self.want["stage"] != stage:
                            return

                    speed = self.want["speed"]
                    if speed > 0:  # uncapped is ~40x realtime and unwatchable
                        target = deadline + self.frame_skip / (60.0 * speed)
                        if target > now:
                            time.sleep(target - now)
                        deadline = max(target, now - 0.25)

                    logits, value = self.model(torch.as_tensor(obs, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=-1)
                    action = int(Categorical(logits=logits).sample().item())
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += reward
                    frame += self.frame_skip

                    fps_n += 1
                    if now - fps_t0 >= 1.0:
                        fps, fps_n, fps_t0 = fps_n / (now - fps_t0), 0, now

                    if frame % self.every == 0:
                        img = Image.fromarray(
                            np.asarray(env.render(), dtype=np.uint8)[:, :, :3]
                        )
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        gw = env.gw
                        snap = {
                            "status": "playing",
                            "checkpoint": self._loaded[0] or "random weights",
                            "load_error": self.load_error,
                            "reloads": self._reloads,
                            "stage": stage,
                            "episode": episode,
                            "frame": frame,
                            "reward": round(ep_reward, 1),
                            "value": round(float(value.item()), 2),
                            "score": int(gw.score),
                            "balls_left": int(gw.balls_left),
                            "dex_caught": sum(1 for v in env._dex_bytes() if v & 2),
                            "stage_id": int(gw.current_stage),
                            "fps": round(fps),
                            "probs": [round(float(p), 3) for p in probs.tolist()],
                            "action": action,
                        }
                        with self.lock:
                            self.jpeg = buf.getvalue()
                            self.stats = snap

                    if terminated or truncated:
                        break
        finally:
            env.close()
            with self.lock:
                self.jpeg = None


# --------------------------------------------------------------------------- http

PAGE_PATH = Path(__file__).with_name("dashboard.html")


def make_handler(scalars: Scalars, player: LivePlayer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep the console clean for training output
            pass

        # -- helpers -------------------------------------------------------
        def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(json.dumps(obj).encode(), "application/json")

        # -- routes --------------------------------------------------------
        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            path, query = url.path, urllib.parse.parse_qs(url.query)

            if path == "/":
                # Read per request so editing the page is a browser refresh, not a restart.
                self._send(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/runs":
                self._json({"runs": scalars.index(), "stages": STAGES, "now": time.time()})
            elif path == "/api/scalars":
                names = [n for n in query.get("runs", [""])[0].split(",") if n]
                self._json({n: scalars.series(n) for n in names})
            elif path == "/api/stats":
                with player.lock:
                    self._json({**player.stats, "want": dict(player.want)})
            elif path.startswith("/api/video/"):
                self._video(path[len("/api/video/"):])
            elif path == "/stream":
                self._stream()
            else:
                self.send_error(404)

        def do_POST(self):  # noqa: N802
            if urllib.parse.urlparse(self.path).path != "/api/live":
                self.send_error(404)
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                self.send_error(400)
                return
            speed = body.get("speed")
            player.configure(
                checkpoint=body.get("checkpoint"),
                stage=body.get("stage"),
                speed=None if speed is None else float(speed),
            )
            self._json({"ok": True, "want": player.want})

        def _video(self, rel: str) -> None:
            # Confined to runs/<run>/videos/<file>.mp4 -- this binds to 0.0.0.0 by default.
            target = (RUNS / urllib.parse.unquote(rel)).resolve()
            root = RUNS.resolve()
            if not (
                target.is_file()
                and target.suffix == ".mp4"
                and root in target.parents
                and target.parent.name == "videos"
            ):
                self.send_error(404)
                return
            self._send(target.read_bytes(), "video/mp4", cache=True)

        def _stream(self) -> None:
            player.viewer_join()
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
                        player.last_viewer = time.time()
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
            finally:
                player.viewer_leave()

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9881)  # 9875/9876/9880 are taken locally
    ap.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    ap.add_argument("--checkpoint", default="runs/score-01/latest.pt")
    ap.add_argument("--stage", default="score", choices=STAGES)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--every", type=int, default=2, help="emit every Nth frame")
    ap.add_argument("--speed", type=float, default=1.0, help="playback multiple of realtime")
    args = ap.parse_args()

    player = LivePlayer(args)
    threading.Thread(target=player.run, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(Scalars(), player))
    print(f"dashboard -> http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
