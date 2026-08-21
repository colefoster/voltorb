"""Does the saucer shot have exploitable structure NEAR CONTACT, or at the macro level?

`bc-mpc` concluded "winning is a property of the joint sequence, no per-frame conditional
carries information". That measurement tested the *first* action of a 1,200-frame rollout --
roughly 1,150 frames upstream of the event. It never looked at the frames immediately before
the ball reaches the saucer, which is where a pinball shot is actually decided.

This collects fresh rollouts (the archived demo npz files have no episode boundaries, no
win/loss labels and no time-to-event) and stores, per rollout, the full per-frame action
sequence plus the frame at which the env's own debounced saucer counter fired.

Start states come from the env's own `stage="shot"` pool builder -- real shot opportunities
harvested from random play. `catch-01` is on record as invalidated by synthesised states, so
nothing here invents one.

    uv run python -m voltorb.tools.saucer_granularity collect --workers 8 \
        --states-per-worker 4 --rollouts 400
    uv run python -m voltorb.tools.saucer_granularity analyze
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np

HORIZON = 1_200
N_ACT = 4
SAUCER_DIST_IDX = 29  # OBS_FIELDS.index("saucer_dist")
SAUCER_DIST_SCALE = 180.0


# ---- collection -----------------------------------------------------------------

def _worker(args_tuple) -> str:
    (worker, seed, states_per_worker, rollouts, rom, outdir, pool_skip, record_pos) = args_tuple
    from voltorb.env import EnvConfig, PinballEnv

    env = PinballEnv(EnvConfig(rom_path=rom, stage="shot", headless=True))
    env.reset(seed=seed)  # builds the shot pool via the env's own machinery
    pool = list(env._shot_pool)
    # Spread the picks across the pool: consecutive snapshots are >=400 frames apart but
    # picking a contiguous block still biases toward one stretch of play.
    picks = list(range(0, len(pool), max(1, pool_skip)))[:states_per_worker]

    rng = np.random.default_rng(seed * 7919 + 13)
    acts = np.full((len(picks) * rollouts, HORIZON), -1, dtype=np.int8)
    meta = np.zeros((len(picks) * rollouts, 6), dtype=np.int32)   # state,len,hit,hitframe,closest,lost
    mind = np.zeros(len(picks) * rollouts, dtype=np.float32)
    # Ball position at the moment each action is applied, so the near-contact structure can be
    # asked the only question that matters for a policy: is it visible in the observation?
    pos = (np.zeros((len(picks) * rollouts, HORIZON, 2), dtype=np.uint8)
           if record_pos else None)
    row = 0
    t0 = time.time()
    steps = 0
    for si, pi in enumerate(picks):
        snap = pool[pi]
        for r in range(rollouts):
            env._shot_pool = [snap]
            env._episodes_since_pool = 0
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            best = 1e9
            best_i = 0
            hit = 0
            lost = 0
            n = 0
            for i in range(HORIZON):
                a = int(rng.integers(0, N_ACT))
                if pos is not None:
                    bx, by = env._ball_xy()
                    pos[row, i, 0] = int(np.clip(bx, 0, 255))
                    pos[row, i, 1] = int(np.clip(by, 0, 255))
                obs, _, term, trunc, _ = env.step(a)
                acts[row, i] = a
                n = i + 1
                d = float(obs[SAUCER_DIST_IDX]) * SAUCER_DIST_SCALE
                if d < best:
                    best, best_i = d, i
                if term:
                    hit = int(env._saucer_visits > 0)
                    lost = int(not hit)
                    break
                if trunc:
                    break
            steps += n
            meta[row] = (worker * 1000 + pi, n, hit, n - 1 if hit else -1, best_i, lost)
            mind[row] = best
            row += 1
        print(f"[w{worker}] state {si+1}/{len(picks)} done  "
              f"{steps} steps  {steps/(time.time()-t0):.0f} steps/s", flush=True)
    env.close()
    path = os.path.join(outdir, f"saucer_gran_w{worker}.npz")
    extra = {"pos": pos[:row]} if pos is not None else {}
    np.savez_compressed(path, actions=acts[:row], meta=meta[:row], min_dist=mind[:row], **extra)
    return path


def collect(a) -> None:
    import multiprocessing as mp

    os.makedirs(a.outdir, exist_ok=True)
    jobs = [
        (w, a.seed + 101 * w, a.states_per_worker, a.rollouts, a.rom, a.outdir, a.pool_skip,
         a.record_pos)
        for w in range(a.workers)
    ]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for p in pool.imap_unordered(_worker, jobs):
            print("wrote", p, flush=True)


# ---- analysis -------------------------------------------------------------------

def _load(outdir):
    A, M, D = [], [], []
    for p in sorted(glob.glob(os.path.join(outdir, "saucer_gran_w*.npz"))):
        z = np.load(p)
        A.append(z["actions"]); M.append(z["meta"]); D.append(z["min_dist"])
    return np.concatenate(A), np.concatenate(M), np.concatenate(D)


def _entropy(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _win_window(actions, anchor, n):
    """The n actions ending at (and including) frame `anchor`. None if it runs off the front."""
    if anchor - n + 1 < 0:
        return None
    return actions[anchor - n + 1 : anchor + 1]


def analyze(a) -> None:
    acts, meta, mind = _load(a.outdir)
    rng = np.random.default_rng(a.seed)
    hit_all = meta[:, 2].astype(bool)
    # Drop degenerate start states: the pool builder can snapshot a frame where the ball is
    # already at the saucer, which wins 100% of the time under any action and would dominate
    # the pooled winner counts. Explicit, not silent -- the drop is printed.
    drop = []
    for s in np.unique(meta[:, 0]):
        m = meta[:, 0] == s
        if hit_all[m].mean() > 0.9:
            drop.append(int(s))
    if drop:
        keep_rows = ~np.isin(meta[:, 0], drop)
        print(f"EXCLUDED {len(drop)} degenerate start state(s) {drop} "
              f"(win rate > 0.9: the ball starts in/at the saucer), "
              f"{int((~keep_rows).sum())} rollouts dropped")
        acts, meta, mind = acts[keep_rows], meta[keep_rows], mind[keep_rows]
    states = np.unique(meta[:, 0])
    hit = meta[:, 2].astype(bool)
    print(f"rollouts {len(meta)}  states {len(states)}  winners {hit.sum()} "
          f"({hit.mean():.3f})  losers {(~hit).sum()}  ball-lost {(meta[:,5]==1).sum()}")
    print("mean rollout length", meta[:, 1].mean())
    print("\nper-state win rates:")
    for s in states:
        m = meta[:, 0] == s
        print(f"  state {s:6d}: {hit[m].sum():4d}/{m.sum():4d} = {hit[m].mean():.3f}"
              f"   min_dist median {np.median(mind[m]):.1f}")

    windows = [int(v) for v in str(a.windows).split(",")]
    MC = a.mc

    # ---- TEST 1: winners' action distribution in the last N frames before the visit,
    # against the EXACT null (the generating policy is iid uniform over 4 actions).
    print("\n=== TEST 1  near-contact action structure, winners vs exact iid-uniform null ===")
    print("statistic: entropy of pooled action counts in the last N frames before the visit")
    print(f"{'N':>4} {'scope':>10} {'seqs':>6} {'samples':>8} {'H_obs':>8} {'H_null':>8} "
          f"{'sd':>7} {'z':>7} {'chi2':>8} {'p_chi2':>8}")
    from math import erfc, sqrt
    for n in windows:
        rows = []
        for i in np.nonzero(hit)[0]:
            w = _win_window(acts[i], int(meta[i, 3]), n)
            if w is not None:
                rows.append(w)
        if not rows:
            continue
        W = np.stack(rows)
        _report_uniformity("pooled", n, W, MC, rng)
        # per-state, the load-bearing version
        zs = []
        for s in states:
            idx = np.nonzero(hit & (meta[:, 0] == s))[0]
            rr = [_win_window(acts[i], int(meta[i, 3]), n) for i in idx]
            rr = [x for x in rr if x is not None]
            if len(rr) < a.min_winners:
                continue
            z = _report_uniformity(f"state{s}", n, np.stack(rr), MC, rng, quiet=True)
            zs.append(z)
        if zs:
            zs = np.array(zs)
            comb = zs.mean() * np.sqrt(len(zs))
            print(f"{n:4d} {'per-state':>10} {len(zs):6d} {'':>8} {'':>8} {'':>8} {'':>7} "
                  f"{comb:+7.2f}  <- Stouffer over {len(zs)} states, "
                  f"mean z {zs.mean():+.2f}, sd {zs.std(ddof=1) if len(zs)>1 else 0:.2f}, "
                  f"max |z| {np.abs(zs).max():.2f}")

    # ---- TEST 2: mutual information between the action at offset -k and winning, with
    # losers aligned on their closest approach to the saucer.
    print("\n=== TEST 2  MI(action at offset -k ; win), losers aligned on closest approach ===")
    print("null: shuffle the win label within each state (exact under H0), 2000 draws")
    for n in windows:
        _mi_test(acts, meta, hit, states, n, a, rng, align="closest")
    print("\n  control: losers aligned on a random frame instead")
    for n in windows:
        _mi_test(acts, meta, hit, states, n, a, rng, align="random")

    # ---- TEST 1b: the same scan with NO loser alignment at all. Winners only, exact null.
    # The generating policy is iid uniform, so P(a at offset -k | win) = 0.25 under H0 for
    # every offset -- there is no reference group to mis-align and no artefact to induce.
    print("\n=== TEST 1b  P(action | win) per offset before the visit, exact binomial null ===")
    _winner_offset_scan(acts, meta, hit, states, a)

    # ---- TEST 2b: the directly interpretable version -- how much does the action at a
    # single offset before the event move the win rate, pooled with within-state centring?
    print("\n=== TEST 2b  win-rate shift by the action at offset -k (state-stratified) ===")
    print("delta = P(win | a at -k) - P(win) pooled over states after removing each state's")
    print("base rate; z is against the binomial se. 90 offsets x 4 actions = 360 tests, so")
    print("the largest |z| expected under the null is ~3.2.")
    _offset_scan(acts, meta, states, a, rng)

    # ---- TEST 3: macro separation. Does win rate differ by WHICH 90-frame block was played?
    print("\n=== TEST 3  macro separation: win rate by 90-frame action-block category ===")
    _macro_test(acts, meta, hit, states, a, rng, block="prefix")
    _macro_test(acts, meta, hit, states, a, rng, block="preevent")

    # ---- TEST 4: can a supervised model predict the win from the last-N actions?
    print("\n=== TEST 4  out-of-sample prediction of the win from the last-N action window ===")
    print("logistic regression on one-hot(action x offset), 5-fold stratified CV, per state.")
    print("delta = base-rate log-loss - model log-loss, in nats/rollout. >0 means the actions")
    print("carry usable information. Null from label permutations (same CV).")
    for n in [int(v) for v in str(a.windows).split(",")]:
        _predict_test(acts, meta, hit, states, n, a, rng)


def _report_uniformity(scope, n, W, MC, rng, quiet=False):
    from scipy import stats
    counts = np.bincount(W.reshape(-1), minlength=N_ACT).astype(float)
    tot = int(counts.sum())
    h = _entropy(counts)
    # Exact null: the generating policy IS iid uniform over 4 actions, so the null for the
    # same number of drawn actions is a multinomial -- no estimated reference needed.
    draws = rng.multinomial(tot, np.full(N_ACT, 1.0 / N_ACT), size=MC) / tot
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(draws > 0, draws * np.log(draws), 0.0)
    null = -lg.sum(1)
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    z = (h - mu) / sd if sd else 0.0
    exp = tot / N_ACT
    chi2 = float(((counts - exp) ** 2 / exp).sum())
    p = float(stats.chi2.sf(chi2, N_ACT - 1))
    if not quiet:
        frac = np.round(counts / tot, 4)
        print(f"{n:4d} {scope:>10} {len(W):6d} {tot:8d} {h:8.4f} {mu:8.4f} {sd:7.4f} "
              f"{z:+7.2f} {chi2:8.2f} {p:8.3f}   p(a)={frac}")
    return z


def _anchor(meta_row, align="closest", rng=None):
    if meta_row[2]:
        return int(meta_row[3])
    if align == "closest":
        return int(meta_row[4])
    return int(rng.integers(0, max(int(meta_row[1]), 1)))


def _onehot(W):
    """(n, L) int actions -> (L*N_ACT, n) float indicator, for batched count matmuls."""
    n, L = W.shape
    O = np.zeros((L, N_ACT, n), dtype=np.float32)
    idx = np.arange(n)
    for k in range(L):
        O[k, W[:, k], idx] = 1.0
    return O.reshape(L * N_ACT, n)


def _mi_batch(O, L, n, Y):
    """sum_k I(a_k ; y) in nats for each column of Y (n, B) 0/1 labels. Plug-in estimate."""
    cw = (O @ Y).reshape(L, N_ACT, -1)              # winner counts per (offset, action, draw)
    ca = O.sum(1).reshape(L, N_ACT, 1)              # marginal action counts per offset
    c1 = cw
    c0 = ca - cw
    nw = Y.sum(0)[None, None, :]
    py1 = nw / n
    py0 = 1.0 - py1
    pa = ca / n
    out = np.zeros(Y.shape[1])
    for c, py in ((c1, py1), (c0, py0)):
        pj = c / n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = pj * np.log(pj / (py * pa))
        out += np.nansum(np.where(pj > 0, t, 0.0), axis=(0, 1))
    return out


def _mi_test(acts, meta, hit, states, n, a, rng, align):
    """Per-frame mutual information between the action at each offset before the event and
    the win label. This is exactly the quantity the bc-mpc claim is about ("no per-frame
    conditional carries information"), measured near contact instead of at frame 0."""
    zs, kept, obs_sum, null_sum, ns = [], 0, 0.0, 0.0, []
    for s in states:
        idx = np.nonzero(meta[:, 0] == s)[0]
        rowsel, wins = [], []
        for i in idx:
            anc = _anchor(meta[i], align=align, rng=rng)
            if anc - n + 1 < 0:
                continue
            rowsel.append(acts[i, anc - n + 1: anc + 1])
            wins.append(bool(meta[i, 2]))
        wins = np.array(wins)
        if len(wins) == 0 or wins.sum() < a.min_winners or (~wins).sum() < a.min_winners:
            continue
        W = np.stack(rowsel)
        nn = len(W)
        O = _onehot(W)
        y = wins.astype(np.float32)[:, None]
        obs = float(_mi_batch(O, n, nn, y)[0])
        Yp = np.stack([rng.permutation(wins) for _ in range(a.perm)], axis=1).astype(np.float32)
        null = _mi_batch(O, n, nn, Yp)
        mu, sd = null.mean(), null.std(ddof=1)
        zs.append((obs - mu) / sd if sd else 0.0)
        obs_sum += obs
        null_sum += mu
        ns.append(nn)
        kept += 1
    if not kept:
        print(f"  N={n}: no state had enough of both classes")
        return
    zs = np.array(zs)
    comb = zs.mean() * np.sqrt(len(zs))
    print(f"  N={n:3d} [{align}] states {kept}  mean rollouts/state {np.mean(ns):.0f}  "
          f"sum-MI obs {obs_sum:.4f} vs perm-null {null_sum:.4f} nats  "
          f"per-state z: mean {zs.mean():+.2f} sd {zs.std(ddof=1):.2f} "
          f"max {zs.max():+.2f} min {zs.min():+.2f}  Stouffer {comb:+.2f}")


def _features(W):
    """Cheap, interpretable summary of a 90-frame action block."""
    n, L = W.shape
    f = [ (W == k).mean(1) for k in range(N_ACT) ]
    switches = (W[:, 1:] != W[:, :-1]).mean(1)
    up = (W > 0).mean(1)
    # coarse temporal shape: mean flipper-up in each third
    thirds = [ (W[:, i*L//3:(i+1)*L//3] > 0).mean(1) for i in range(3) ]
    return np.stack(f + [switches, up] + thirds, axis=1)


def _macro_test(acts, meta, hit, states, a, rng, block):
    from scipy import stats
    n = 90
    print(f"\n  -- {block} blocks, k-means into {a.clusters} categories, per state --")
    rows = []
    for s in states:
        idx = np.nonzero(meta[:, 0] == s)[0]
        sel, wins = [], []
        for i in idx:
            if block == "prefix":
                if meta[i, 1] < n:
                    continue
                sel.append(acts[i, :n])
            else:
                anc = _anchor(meta[i], align="closest", rng=rng)
                if anc - n + 1 < 0:
                    continue
                sel.append(acts[i, anc - n + 1: anc + 1])
            wins.append(bool(meta[i, 2]))
        if len(sel) < 50:
            continue
        W = np.stack(sel); wins = np.array(wins)
        if wins.sum() < a.min_winners or (~wins).sum() < a.min_winners:
            continue
        X = _features(W)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        from sklearn.cluster import KMeans
        lab = KMeans(n_clusters=a.clusters, n_init=4, random_state=0).fit_predict(X)
        tab = np.stack([np.bincount(lab[~wins], minlength=a.clusters),
                        np.bincount(lab[wins], minlength=a.clusters)]).astype(float)
        keep = tab.sum(0) >= 5
        chi2, p, dof, _ = stats.chi2_contingency(tab[:, keep], correction=False)
        # Permutation null on the same statistic: shuffle the win label within the state,
        # which holds the cluster sizes and the win count fixed.
        C = int(keep.sum())
        L = np.zeros((a.clusters, len(lab)), dtype=np.float32)
        L[lab, np.arange(len(lab))] = 1.0
        L = L[keep]
        Yp = np.stack([rng.permutation(wins) for _ in range(a.perm)], axis=1).astype(np.float32)
        cw = L @ Yp                       # (C, perm) winners per cluster
        csz = L.sum(1)[:, None]
        nw = Yp.sum(0)[None, :]
        ntot = len(lab)
        exp1 = csz * nw / ntot
        exp0 = csz * (ntot - nw) / ntot
        null = (((cw - exp1) ** 2 / exp1) + (((csz - cw) - exp0) ** 2 / exp0)).sum(0)
        pperm = float((null >= chi2).mean())
        rates = tab[1] / np.maximum(tab.sum(0), 1)
        rows.append((s, len(W), wins.mean(), chi2, dof, p, pperm, rates[keep]))
        print(f"    state {s:6d}: n {len(W):4d}  win {wins.mean():.3f}  "
              f"chi2 {chi2:7.2f} df {dof}  p {p:.3f}  p_perm {pperm:.3f}  "
              f"cluster win rates {np.round(rates[keep],3)}")
    if rows:
        ps = np.array([r[6] for r in rows])
        # Fisher combination over states
        chi = -2 * np.log(np.clip(ps, 1e-12, 1)).sum()
        pc = float(stats.chi2.sf(chi, 2 * len(ps)))
        nsig = int((ps < 0.05).sum())
        print(f"    combined over {len(ps)} states (Fisher on permutation p): "
              f"chi2 {chi:.2f}, p = {pc:.4g};  states with p_perm<0.05: {nsig} "
              f"(expected {0.05*len(ps):.1f})")


def _winner_offset_scan(acts, meta, hit, states, a, kmax=90):
    from scipy import stats
    idx = [i for i in np.nonzero(hit)[0] if int(meta[i, 3]) - kmax + 1 >= 0]
    W = np.stack([acts[i, int(meta[i, 3]) - kmax + 1: int(meta[i, 3]) + 1] for i in idx])
    n = len(W)
    cnt = np.stack([(W == act).sum(0) for act in range(N_ACT)], axis=1).astype(float)  # (kmax, 4)
    p0 = 1.0 / N_ACT
    z = (cnt - n * p0) / np.sqrt(n * p0 * (1 - p0))
    print(f"  winners with a full {kmax}-frame window: {n}")
    print(f"  {'offset':>7} " + " ".join(f"{'a'+str(i)+' p/z':>13}" for i in range(N_ACT)))
    for k in (1, 2, 3, 5, 10, 20, 40, 90):
        i = kmax - k
        cells = " ".join(f"{cnt[i,act]/n:.4f}/{z[i,act]:+5.2f}" for act in range(N_ACT))
        print(f"  {-k:7d} {cells}")
    az = np.abs(z)
    j = np.unravel_index(az.argmax(), z.shape)
    print(f"  max |z| over {kmax*N_ACT} cells: {az.max():.2f} at offset -{kmax - j[0]}, "
          f"action {j[1]};  |z|>3: {(az>3).sum()} (expected {kmax*N_ACT*0.0027:.1f});  "
          f"|z|>2: {(az>2).sum()} (expected {kmax*N_ACT*0.0455:.1f})")
    print(f"  largest deviation from 0.25 anywhere: "
          f"{np.abs(cnt/n - p0).max():.4f}  (se {np.sqrt(p0*(1-p0)/n):.4f})")
    # per-state version: same statistic inside each state
    mx, sig = [], 0
    for st in states:
        ii = [i for i in np.nonzero(hit & (meta[:, 0] == st))[0]
              if int(meta[i, 3]) - kmax + 1 >= 0]
        if len(ii) < a.min_winners:
            continue
        Ws = np.stack([acts[i, int(meta[i, 3]) - kmax + 1: int(meta[i, 3]) + 1] for i in ii])
        ns = len(Ws)
        c = np.stack([(Ws == act).sum(0) for act in range(N_ACT)], axis=1).astype(float)
        zz = np.abs((c - ns * p0) / np.sqrt(ns * p0 * (1 - p0)))
        mx.append(zz.max())
        sig += int((zz > 4).sum())
    print(f"  per-state: {len(mx)} states, max |z| mean {np.mean(mx):.2f} "
          f"(expected ~3.5 as the max of 360 correlated normals), overall max {np.max(mx):.2f}; "
          f"cells with |z|>4 summed over states: {sig} "
          f"(expected {len(mx)*360*6.3e-5:.2f})")


def _offset_scan(acts, meta, states, a, rng, kmax=90):
    per_state = []
    for s in states:
        idx = np.nonzero(meta[:, 0] == s)[0]
        rows, wins = [], []
        for i in idx:
            anc = _anchor(meta[i], align="closest", rng=rng)
            if anc - kmax + 1 < 0:
                continue
            rows.append(acts[i, anc - kmax + 1: anc + 1])
            wins.append(bool(meta[i, 2]))
        wins = np.array(wins)
        if len(wins) == 0 or wins.sum() < a.min_winners:
            continue
        per_state.append((np.stack(rows), wins))
    num = np.zeros((kmax, N_ACT))
    den = np.zeros((kmax, N_ACT))
    var = np.zeros((kmax, N_ACT))
    for W, wins in per_state:
        pbar = wins.mean()
        for act in range(N_ACT):
            m = (W == act)                       # (n, kmax)
            n_sa = m.sum(0)                      # per offset
            w_sa = (m & wins[:, None]).sum(0)
            num[:, act] += w_sa - n_sa * pbar
            den[:, act] += n_sa
            var[:, act] += n_sa * pbar * (1 - pbar)
    delta = num / np.maximum(den, 1)
    z = num / np.sqrt(np.maximum(var, 1e-9))
    print(f"  states used {len(per_state)}, rollouts {sum(len(w) for _, w in per_state)}")
    print(f"  {'offset':>7} " + " ".join(f"{'a'+str(i)+' d/z':>14}" for i in range(N_ACT)))
    for k in (1, 2, 3, 5, 10, 20, 40, 90):
        i = kmax - k
        cells = " ".join(f"{delta[i,act]:+.4f}/{z[i,act]:+5.2f}" for act in range(N_ACT))
        print(f"  {-k:7d} {cells}")
    flat = np.abs(z).ravel()
    print(f"  max |z| over {kmax*N_ACT} (offset,action) cells: {flat.max():.2f} "
          f"at offset -{kmax - np.unravel_index(np.abs(z).argmax(), z.shape)[0]}, "
          f"action {np.unravel_index(np.abs(z).argmax(), z.shape)[1]};  "
          f"cells with |z|>3: {(flat>3).sum()} (expected {360*0.0027:.1f}); "
          f"|z|>2: {(flat>2).sum()} (expected {360*0.0455:.1f})")
    print(f"  largest |delta| anywhere: {np.abs(delta).max():.4f} win-rate points")


def _cv_delta(X, y, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ll_model, ll_base, n_tot = 0.0, 0.0, 0
    for tr, te in skf.split(X, y):
        base = y[tr].mean()
        base = min(max(base, 1e-6), 1 - 1e-6)
        clf = LogisticRegression(max_iter=300, C=0.1)
        clf.fit(X[tr], y[tr])
        p = np.clip(clf.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        yt = y[te]
        ll_model += -(yt * np.log(p) + (1 - yt) * np.log(1 - p)).sum()
        ll_base += -(yt * np.log(base) + (1 - yt) * np.log(1 - base)).sum()
        n_tot += len(te)
    return (ll_base - ll_model) / n_tot


def _predict_test(acts, meta, hit, states, n, a, rng):
    deltas, zs, ns = [], [], []
    for s in states:
        idx = np.nonzero(meta[:, 0] == s)[0]
        rows, wins = [], []
        for i in idx:
            anc = _anchor(meta[i], align="closest", rng=rng)
            if anc - n + 1 < 0:
                continue
            rows.append(acts[i, anc - n + 1: anc + 1])
            wins.append(bool(meta[i, 2]))
        wins = np.array(wins)
        if len(wins) == 0 or wins.sum() < a.min_winners or (~wins).sum() < a.min_winners:
            continue
        W = np.stack(rows)
        X = np.zeros((len(W), n * N_ACT), dtype=np.float32)
        for k in range(n):
            X[np.arange(len(W)), k * N_ACT + W[:, k].astype(np.int64)] = 1.0
        y = wins.astype(int)
        obs = _cv_delta(X, y, seed=0)
        null = np.array([_cv_delta(X, rng.permutation(y), seed=0) for _ in range(a.pred_perm)])
        z = (obs - null.mean()) / (null.std(ddof=1) if null.std(ddof=1) > 0 else 1)
        deltas.append(obs); zs.append(z); ns.append(len(W))
    if not deltas:
        print(f"  N={n}: no usable state")
        return
    d = np.array(deltas); zs = np.array(zs)
    print(f"  N={n:3d} states {len(d)} (mean n {np.mean(ns):.0f})  delta nats/rollout: "
          f"mean {d.mean():+.5f} sd {d.std(ddof=1):.5f} best {d.max():+.5f}  "
          f"per-state z: mean {zs.mean():+.2f} max {zs.max():+.2f}  "
          f"Stouffer {zs.mean()*np.sqrt(len(zs)):+.2f}  states with z>2: {(zs>2).sum()}")




# ---- the load-bearing follow-ups ------------------------------------------------
#
# TEST 1b found a band of offsets, 53-66 frames before the visit, where P(action | win)
# departs from uniform. These are the checks that decide whether that is real and whether a
# policy could act on it:
#   band      -- the offset profile, split-half and held-out-state replication, per-state
#                contrast, and a synthetic iid control that validates the machinery
#   hazard    -- needs a collection made with --record-pos. Conditions the effect on where
#                the ball actually is, and runs the exploitable forward version: does the
#                action change P(visit in 45-70 frames) within a ball-position bucket?

def _load_pos(outdir):
    A, M, P = [], [], []
    for p in sorted(glob.glob(os.path.join(outdir, "saucer_gran_w*.npz"))):
        z = np.load(p)
        A.append(z["actions"]); M.append(z["meta"]); P.append(z["pos"])
    return np.concatenate(A), np.concatenate(M), np.concatenate(P)


def _drop_degenerate(acts, meta, *rest):
    hit = meta[:, 2].astype(bool)
    drop = [int(s) for s in np.unique(meta[:, 0])
            if hit[meta[:, 0] == s].mean() > 0.9]
    if not drop:
        return (acts, meta) + rest
    k = ~np.isin(meta[:, 0], drop)
    print(f"EXCLUDED degenerate start state(s) {drop} (win rate > 0.9), "
          f"{int((~k).sum())} rollouts")
    return (acts[k], meta[k]) + tuple(r[k] for r in rest)


def _winner_windows(acts, meta, K=90):
    hit = meta[:, 2].astype(bool)
    idx = np.array([i for i in np.nonzero(hit)[0] if int(meta[i, 3]) - K + 1 >= 0])
    W = np.stack([acts[i, int(meta[i, 3]) - K + 1: int(meta[i, 3]) + 1] for i in idx])
    return idx, W


def _prof(W, p0=0.25):
    n = len(W)
    c = np.stack([(W == a).sum(0) for a in range(N_ACT)], axis=1).astype(float)
    return c / n, (c - n * p0) / np.sqrt(n * p0 * (1 - p0))


def band(a) -> None:
    acts, meta = _drop_degenerate(*_load(a.outdir)[:2])
    K, p0 = 90, 0.25
    idx, W = _winner_windows(acts, meta, K)
    st = meta[idx, 0]
    states = np.unique(st)
    f, z = _prof(W)
    print(f"pooled winners with a full {K}-frame window: {len(W)}")
    print(" off      a0     a1     a2     a3   |    z0    z1    z2    z3")
    for k in range(70, 45, -1):
        i = K - k
        print(f"{-k:4d}  " + " ".join(f"{f[i,x]:.4f}" for x in range(4)) + " | "
              + " ".join(f"{z[i,x]:+5.2f}" for x in range(4)))

    rng = np.random.default_rng(1)
    print("\nreplication:")
    for t in range(3):
        perm = rng.permutation(len(W))
        z1 = _prof(W[perm[: len(W) // 2]])[1]
        z2 = _prof(W[perm[len(W) // 2:]])[1]
        print(f"  random split-half {t}: corr(z, z') = "
              f"{np.corrcoef(z1.ravel(), z2.ravel())[0,1]:+.3f}")
    gA, gB = states[: len(states) // 2], states[len(states) // 2:]
    fA, zA = _prof(W[np.isin(st, gA)])
    fB, zB = _prof(W[np.isin(st, gB)])
    print(f"  state-group split ({len(gA)} vs {len(gB)} states): corr = "
          f"{np.corrcoef(zA.ravel(), zB.ravel())[0,1]:+.3f}")
    kbest = int(np.abs(zA).max(1).argmax())
    dirA = zA[kbest] / np.linalg.norm(zA[kbest])
    nB = int(np.isin(st, gB).sum())
    devB = (fB[kbest] - p0) * np.sqrt(nB / (p0 * (1 - p0)))
    print(f"  discovered on group A: offset -{K-kbest}, p(a)={np.round(fA[kbest],4)}")
    print(f"  HELD-OUT on group B ({nB} winners, one pre-specified test): "
          f"p(a)={np.round(fB[kbest],4)}, projection z = {float(devB @ dirA):+.2f}")
    cnt = []
    for _ in range(20):
        cnt.append((np.abs(_prof(rng.integers(0, N_ACT, size=W.shape))[1]) > 3).sum())
    print(f"  synthetic iid control ({K*N_ACT} cells, 20 draws): cells |z|>3 mean "
          f"{np.mean(cnt):.2f}, max {max(cnt)}; observed {int((np.abs(z)>3).sum())}")

    bandoff = [K - k for k in (53, 54, 55, 56)]
    zs, ds = [], []
    for s in states:
        Ws = W[st == s]
        tot = len(Ws) * len(bandoff)
        d = (sum((Ws[:, i] == 3).sum() for i in bandoff)
             - sum((Ws[:, i] == 0).sum() for i in bandoff)) / tot
        ds.append(d)
        zs.append(d / np.sqrt(2 * p0 / tot))
    ds, zs = np.array(ds), np.array(zs)
    print(f"\nper-state (primary): contrast p(both) - p(none) over offsets -53..-56")
    print(f"  {len(ds)} states, mean {ds.mean():+.4f}, positive in {(ds>0).sum()}/{len(ds)}, "
          f"per-state z mean {zs.mean():+.2f} sd {zs.std(ddof=1):.2f}, "
          f"Stouffer {zs.mean()*np.sqrt(len(zs)):+.2f}")
    base = meta[:, 2].mean()
    print(f"\nimplied one-frame effect at offset -54 (base win rate {base:.4f}):")
    for x, lab in enumerate(("none", "left", "right", "both")):
        print(f"  {lab:5s}: p(a|win) {f[K-54,x]:.4f} -> P(win|a) {base*f[K-54,x]/p0:.4f}")


def hazard(a) -> None:
    """Needs a --record-pos collection. Two questions: is the band effect visible in the
    ball state (so a per-frame policy could express it), and what does the exploitable
    forward version buy?"""
    from scipy import stats

    acts, meta, pos = _drop_degenerate(*_load_pos(a.outdir))
    K, p0 = 90, 0.25
    idx, _ = _winner_windows(acts, meta, K)
    print("conditioning the band on where the ball is:")
    for off in (54, 55, 58, 61):
        y = np.array([pos[i, int(meta[i, 3]) - off, 1] for i in idx])
        for lab, sel in (("all", idx), (f"ball y>={a.ysplit}", idx[y >= a.ysplit]),
                         (f"ball y<{a.ysplit}", idx[y < a.ysplit])):
            act = np.array([acts[i, int(meta[i, 3]) - off] for i in sel]).astype(int)
            n = len(act)
            c = np.bincount(act, minlength=N_ACT).astype(float)
            p = c / n
            zz = (c - n * p0) / np.sqrt(n * p0 * (1 - p0))
            kl = float((p[p > 0] * np.log(p[p > 0] / p0)).sum())
            h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
            print(f"  offset -{off} {lab:16s} n={n:5d} p(a|win)={np.round(p,4)} "
                  f"z={np.round(zz,2)} H={h:.4f} KL={kl:.4f}")
        print()

    LO, HI, nb = a.lo, a.hi, a.buckets
    xb = np.linspace(0, 160, nb + 1)
    yb = np.linspace(0, 160, nb + 1)

    def build(mask):
        t = np.zeros((nb, nb, N_ACT)); s = np.zeros((nb, nb, N_ACT))
        for i in np.nonzero(mask)[0]:
            n = int(meta[i, 1]); h = int(meta[i, 3])
            act = acts[i, :n].astype(int)
            x = np.clip(np.digitize(pos[i, :n, 0], xb) - 1, 0, nb - 1)
            y = np.clip(np.digitize(pos[i, :n, 1], yb) - 1, 0, nb - 1)
            lab = np.zeros(n, bool)
            if h >= 0:
                lab[max(h - HI, 0): max(h - LO + 1, 0)] = True
            valid = np.arange(n) <= (h if h >= 0 else n - 1 - HI)
            np.add.at(t, (x[valid], y[valid], act[valid]), 1)
            np.add.at(s, (x[valid], y[valid], act[valid]), lab[valid])
        return t, s

    tot, succ = build(np.ones(len(meta), bool))
    T, S = tot.sum(2), succ.sum(2)
    print(f"forward hazard P(visit in {LO}-{HI} frames): frames {int(T.sum()):,}, "
          f"events {int(S.sum()):,}, base {S.sum()/T.sum():.5f}")
    chi = 0.0; dof = 0; rows = []
    for i in range(nb):
        for j in range(nb):
            if T[i, j] < 2000 or S[i, j] < 20:
                continue
            ph = S[i, j] / T[i, j]
            e = tot[i, j] * ph
            c = float(((succ[i, j] - e) ** 2 / np.maximum(e, 1e-9)).sum()
                      + ((succ[i, j] - e) ** 2 / np.maximum(tot[i, j] - e, 1e-9)).sum())
            chi += c; dof += N_ACT - 1
            rows.append((c, i, j, int(T[i, j]), ph, succ[i, j] / np.maximum(tot[i, j], 1)))
    print(f"  within-bucket action effect: chi2 {chi:.1f} on {dof} df, "
          f"p = {stats.chi2.sf(chi, dof):.3g} ({len(rows)} of {nb*nb} buckets)")
    rows.sort(reverse=True)
    for c, i, j, n, ph, r in rows[:6]:
        print(f"    x[{xb[i]:.0f},{xb[i+1]:.0f}) y[{yb[j]:.0f},{yb[j+1]:.0f}) n={n:7d} "
              f"base {ph:.4f} by action {np.round(r,4)} chi2 {c:.1f} "
              f"spread {r.max()/max(r.min(),1e-9):.2f}x")
    rng = np.random.default_rng(0)
    half = rng.random(len(meta)) < 0.5
    t1, s1 = build(half); t2, s2 = build(~half)
    pick = (s1 / np.maximum(t1, 1)).argmax(2)
    ok = t1.sum(2) >= 2000
    num = den = bn = bd = 0.0
    for i in range(nb):
        for j in range(nb):
            if not ok[i, j]:
                continue
            k = pick[i, j]
            num += s2[i, j, k]; den += t2[i, j, k]
            bn += s2[i, j].sum(); bd += t2[i, j].sum()
    print(f"  held-out greedy per-bucket action rule: hazard {num/den:.5f} vs base "
          f"{bn/bd:.5f} = {num/den/(bn/bd):.3f}x on {int(den):,} frames")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--rom", default="roms/pokemon_pinball.gbc")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--states-per-worker", type=int, default=4)
    c.add_argument("--rollouts", type=int, default=400)
    c.add_argument("--pool-skip", type=int, default=6)
    c.add_argument("--seed", type=int, default=20260819)
    c.add_argument("--outdir", default="data/saucer_gran")
    c.add_argument("--record-pos", action="store_true",
                   help="also store the ball (x, y) at every applied action")
    an = sub.add_parser("analyze")
    an.add_argument("--outdir", default="data/saucer_gran")
    an.add_argument("--windows", default="5,10,20,40,90")
    an.add_argument("--mc", type=int, default=2000)
    an.add_argument("--perm", type=int, default=2000)
    an.add_argument("--clusters", type=int, default=8)
    an.add_argument("--min-winners", type=int, default=15)
    an.add_argument("--pred-perm", type=int, default=20)
    an.add_argument("--seed", type=int, default=5)
    b = sub.add_parser("band")
    b.add_argument("--outdir", default="data/saucer_gran")
    h = sub.add_parser("hazard")
    h.add_argument("--outdir", default="data/saucer_gran_pos")
    h.add_argument("--ysplit", type=int, default=110)
    h.add_argument("--lo", type=int, default=45)
    h.add_argument("--hi", type=int, default=70)
    h.add_argument("--buckets", type=int, default=8)
    a = ap.parse_args()
    {"collect": collect, "analyze": analyze, "band": band, "hazard": hazard}[a.cmd](a)


if __name__ == "__main__":
    main()
