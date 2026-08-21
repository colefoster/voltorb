# colepc / WSL durable training-run recipe

Verified 2026-08-20: a job launched this way survived a full ssh disconnect,
3.5+ minutes idle, and a fresh reconnect (log kept growing, process alive,
killed cleanly by name).

## Root cause (why the naive approach failed)

Two independent, stacking failure modes — both had to be fixed:

1. **VM idle shutdown.** WSL2's lightweight VM (shared kernel) shuts down
   `vmIdleTimeout` ms (default 60s) after the last `wsl.exe` client detaches.
   When it shuts down, the whole guest — including `/tmp` — is gone on next
   boot. Fixed by setting `vmIdleTimeout=-1` in `C:\Users\cole\.wslconfig`
   (`[wsl2]` section) and running `wsl --shutdown` once to apply it.

2. **systemd-logind killing the session on logout.** Even with the VM kept
   alive, every `wsl -d Ubuntu-24.04 -e bash -lc "..."` invocation is its own
   PAM login session. By default `Linger=no`, so the moment that session's
   last process (the `bash -lc` itself) exits, logind kills the entire
   `user-1000.slice` cgroup — including any `setsid`/`nohup`/`tmux` process
   that was supposedly detached. This is why the log file was **never even
   created**: the background job got killed before it finished writing its
   first line, or was killed and then the VM/tmpfs teardown erased the
   evidence. Fixed with `loginctl enable-linger cole` (works as the
   unprivileged user, no sudo needed).

3. **Distro-instance teardown independent of the VM.** Even with both fixes
   above, if literally no `wsl.exe` process stays attached to the
   `Ubuntu-24.04` distro instance, WSL tears down that instance's userspace
   (wiping `/tmp`, restarting systemd) while the shared kernel/VM keeps
   running uninterrupted (`uptime` never resets) — confirmed by an isolated
   test with linger enabled but no anchor process: job died anyway. An
   **anchor process** — one long-lived `wsl.exe` invocation kept alive from
   the Windows side, outside any ssh session — is required to keep the
   instance attached.

Quoting note: `ssh colepc '...'` lands in `cmd.exe`, which mangles **nested**
double quotes (e.g. `wsl -e bash -lc "tmux new -ds x \"foo > bar\""`) —
command silently no-ops or truncates. Fix used throughout: write scripts to
a local file, then `cat file | ssh colepc 'wsl -d Ubuntu-24.04 -e bash -c "cat > /path"'`
to deploy by stdin, then invoke by path with **zero nested quotes**. Same
trick applies to PowerShell one-liners through `cmd.exe`.

Also: don't put persistent scripts in `/tmp` — if any of the above fixes lapse,
`/tmp` is the first thing wiped. Use `$HOME` (e.g. `/home/cole/bin/`).

## One-time setup (already done on colepc, documented for reference)

```
# 1. VM never idles out
ssh colepc 'powershell -NoProfile -Command "Add-Content -Path C:\Users\cole\.wslconfig -Value \"vmIdleTimeout=-1\""'
ssh colepc 'wsl --shutdown'

# 2. user processes survive logout
ssh colepc 'wsl -d Ubuntu-24.04 -e bash -lc "loginctl enable-linger cole"'

# 3. persistent anchor: a wsl.exe process kept alive by Task Scheduler,
#    outside any ssh session, so the distro instance never gets torn down.
#    Runs at every logon, no execution time limit, kept minimal (sleep infinity).
```
The anchor task is named `wsl-anchor` in Task Scheduler. Check/recreate it with:
```
ssh colepc 'schtasks /Query /TN "wsl-anchor" /V /FO LIST | findstr /I Status'
# if missing, recreate (via a .ps1 file to dodge cmd.exe quoting — see mkanchor.ps1 pattern):
#   action:  wsl.exe -d Ubuntu-24.04 -e sleep infinity
#   trigger: At log on of cole
#   settings: ExecutionTimeLimit = 0 (unlimited)
ssh colepc 'schtasks /Run /TN "wsl-anchor"'
```

## Launcher script (deploy once, reuse for every job)

Deploy `~/bin/tmux_launch.sh` inside WSL (survives reboots, avoid `/tmp`):

```bash
cat > /tmp/tmux_launch.sh << 'WRAP'
#!/bin/bash
set -euo pipefail
NAME="$1"; SCRIPT="$2"; LOG="/tmp/${NAME}.log"
tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new -ds "$NAME" "$SCRIPT > $LOG 2>&1"
sleep 1
tmux has-session -t "$NAME" && echo "LAUNCHED $NAME -> $LOG"
WRAP
cat /tmp/tmux_launch.sh | ssh colepc 'wsl -d Ubuntu-24.04 -e bash -c "mkdir -p /home/cole/bin && cat > /home/cole/bin/tmux_launch.sh && chmod +x /home/cole/bin/tmux_launch.sh"'
```

## Launch a named job

Write the actual training command to a small wrapper script first (avoids
all quoting issues — this is the same "file, not inline" pattern), deploy it
to `/home/cole/`, then:

```bash
# from a local file containing your training command, e.g. runs/alley-foo.sh:
#   #!/bin/bash
#   cd ~/voltorb && stdbuf -oL .venv/bin/python train.py --run-name alley-foo ...
cat runs/alley-foo.sh | ssh colepc 'wsl -d Ubuntu-24.04 -e bash -c "cat > /home/cole/alley-foo.sh && chmod +x /home/cole/alley-foo.sh"'
ssh colepc 'wsl -d Ubuntu-24.04 -e bash -lc "/home/cole/bin/tmux_launch.sh alley-foo /home/cole/alley-foo.sh"'
```

Use `stdbuf -oL` (or `python -u`) in the wrapped command — otherwise Python's
stdout buffering means `tail`/`cat` on the log will show nothing for a long
time even though the process is alive and working.

## Check if a job is alive

```bash
ssh colepc 'wsl -d Ubuntu-24.04 -e bash -lc "tmux ls; ps aux | grep alley-foo | grep -v grep"'
```
`tmux ls` lists all running sessions by name. If it says
`no server running on /tmp/tmux-1000/default`, everything died — check the
three fixes above (`.wslconfig`, linger, anchor task) haven't lapsed (e.g.
after a Windows reboot the anchor re-triggers at logon automatically; linger
and vmIdleTimeout are config, not processes, and persist across reboots).

## Tail a job's log

```bash
ssh colepc 'wsl -d Ubuntu-24.04 -e bash -lc "tail -n 50 /tmp/alley-foo.log"'
```

## Kill a job cleanly

```bash
ssh colepc 'wsl -d Ubuntu-24.04 -e bash -lc "tmux kill-session -t alley-foo"'
```
This kills the whole process tree under that tmux session/pane in one shot.

## Gotchas recap

- Nested double quotes through `cmd.exe -> wsl.exe -> bash -lc` silently
  break — always write scripts to files and invoke by path.
- `$HOME`/`~` expand fine inside `bash -lc "..."` (it's still a real bash
  login shell), the danger is only the outer quoting layers, not variable
  expansion.
- Don't rely on `/tmp` for anything that must survive more than a few
  minutes unless all three fixes above are confirmed in place — it's the
  first thing wiped when the distro instance gets torn down.
- Python's stdout buffering will make logs look stalled; use `stdbuf -oL`
  or `PYTHONUNBUFFERED=1`.
