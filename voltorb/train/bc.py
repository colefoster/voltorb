"""Advantage-weighted behaviour cloning, to give PPO a policy worth starting from.

Fits the actor to the search-generated data from tools/collect_demos.py, weighting each sample
by how much its rollout beat the success rate of the state it started in. The critic is left at
its initialisation -- PPO refits it in the first few updates anyway, and there are no returns in
the demo data worth regressing on.

    uv run python -m voltorb.tools.collect_demos --out data/demos.npz
    uv run python -m voltorb.train.bc --demos data/demos.npz --out runs/bc-01.pt
    uv run python -m voltorb.train.ppo --stage shot --init-from runs/bc-01.pt

The output is a normal checkpoint, so --init-from and tools/eval.py both read it.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from voltorb.train.ppo import ActorCritic, save_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/demos.npz")
    ap.add_argument("--out", default="runs/bc-01.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    d = np.load(args.demos)
    obs = torch.as_tensor(d["obs"], dtype=torch.float32)
    act = torch.as_tensor(d["act"], dtype=torch.long)
    w = torch.as_tensor(d["weight"], dtype=torch.float32)
    w = w / w.mean()

    n = len(act)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(n * args.val_frac)
    val, train = perm[:n_val], perm[n_val:]
    print(f"{n:,} samples ({len(train):,} train / {len(val):,} val), obs dim {obs.shape[1]}")

    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss(reduction="none")

    # A weighted fit that beats this is learning something state-dependent; one that does not
    # has only recovered the action frequencies, which is what uniform-random demos would give.
    base = torch.log(torch.tensor(4.0)).item()
    print(f"uniform-policy reference loss {base:.4f}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = train[torch.randperm(len(train))]
        total = 0.0
        for i in range(0, len(idx), args.batch_size):
            mb = idx[i : i + args.batch_size]
            logits, _ = model(obs[mb])
            loss = (ce(logits, act[mb]) * w[mb]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total += loss.item() * len(mb)
        model.eval()
        with torch.no_grad():
            logits, _ = model(obs[val])
            v_loss = (ce(logits, act[val]) * w[val]).mean().item()
            acc = (logits.argmax(-1) == act[val]).float().mean().item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train {total / len(idx):.4f}  val {v_loss:.4f}  "
                  f"val acc {acc:.3f} (chance 0.250)")

    save_checkpoint(model, opt, 0, args.out)
    print(f"\nsaved -> {args.out}")
    print("val loss below the uniform reference means the demos carry state-dependent signal;")
    print("at or above it, the search data is not separable and BC will not help PPO.")


if __name__ == "__main__":
    main()
