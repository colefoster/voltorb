# Run ledger

One line of hypothesis per run. Newest first.

| run | stage | steps | hypothesis | result |
|---|---|---|---|---|
| `survive-01` | survive | 50M | Frame-level 4-action control can learn ball retention from 26 RAM floats alone. Random-policy baseline is 16,891 frames/episode; success = episode length climbing clearly above that. | running — at 3.2M steps `ep_len` oscillates 17.3k–18.5k, no clear trend yet |

## Notes

- Baseline to beat: **16,891 frames/episode** (random policy, measured by `tools/validate.py`).
- Watch `charts/episodic_length`, not `episodic_return` — in the survive stage the return is
  just `0.01 x length` minus a negligible ball-loss penalty, so they carry the same signal.
- Suspect if survive plateaus: `gamma=0.999` gives an effective horizon of ~1,000 frames,
  but a ball lasts ~4,000. The agent may be unable to see a drain coming. Try `0.9997`.
- Also suspect: the `-1` ball-loss penalty is ~0.6% of a typical `+169` episode return, so
  survival is being learned almost entirely from the per-frame alive bonus.
