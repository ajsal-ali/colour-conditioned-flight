# Training logs and weights

`log.jsonl` carries one JSON object per rollout: timesteps, stage, success rate,
mean gates cleared, average ground velocity, heading error, the terminal-reason
histogram, and the PPO diagnostics (policy-gradient loss, value loss, entropy,
approximate KL, self-imitation loss, expert fraction). `curves.png` is the same
data plotted.

## `course_21M` — the current result

20.8M steps on the full three-bar course, attention memory with four tokens,
warm-started from behaviour cloning, the SeVAE encoder frozen throughout.

| | |
|---|---|
| Success | 0.905 ± 0.027 |
| Completed / collision / wrong side | 90% / 6% / 4% |
| Mean lap | 12.5 s |
| Average ground velocity | 0.63 m/s |

Averaged over the last 200 rollouts, about 20,600 episodes, on the training
height split.

What makes this run readable where earlier ones were not:

* **Entropy sits on its floor.** A one-sided entropy bonus applies only below a
  target and is switched off above it, so it can prevent a collapse and can
  never pay for spread the task did not ask for. The curve dips to -2.85 early
  and then holds -2.00 for the rest of the run.
* **The trust region holds.** Approximate KL stays at 0.023 against a 0.03
  target, so the early stop never fires and every epoch runs to completion.
* **The encoder is frozen.** PPO carries no reconstruction, segmentation or
  auxiliary term, so a trainable encoder has nothing anchoring it and gets
  repurposed for short-horizon control. Frozen, the latent stays the one the
  SeVAE was trained to produce.
* **No numerical skips.** `skipped` is 0 for all 1625 rollouts.

The scripted-pilot mix decays to zero by 1.5M steps, so the last 19M are the
policy flying unaided.

## Weights

| File | Size | What it is |
|---|---|---|
| [`../ckpt/sevae.pt`](../ckpt/sevae.pt) | 26 MB | Semantic VAE: RGB-D encoder plus RGB / depth / segmentation decoder heads |
| [`../ckpt/memory_attention_m4.pt`](../ckpt/memory_attention_m4.pt) | 6 MB | Temporal-attention memory, 4 recurrent tokens, with its auxiliary reconstruction head |
| `course_21M/ckpt_best.pt` | 30 MB | Full actor-critic, best smoothed success, 15.5M steps |

The policy checkpoint carries the encoder and memory inside it, so
`--memory-type attention --mem-tokens 4` is required to load it:

```bash
python -m mavrl.evaluate --model results/course_21M/ckpt_best.pt \
    --memory-type attention --mem-tokens 4
```

## `course_35M` — an earlier run, kept for contrast

35.7M steps, success 0.29. Policy entropy falls to -4.4 by 10M steps and then
climbs without bound to +16.4, which for a 4-D diagonal Gaussian is a standard
deviation of roughly 14 against an action space clipped to +/-1.

The entropy bonus was paying for spread the environment cannot see: past the
clip, extra deviation changes nothing about behaviour, so no countervailing
gradient exists and the bonus pushes without limit. Sampled actions over the
later half of that run are effectively bang-bang, which is why success plateaus
near 0.25 while episode return keeps drifting up. Bounding `log_std` and making
the entropy bonus one-sided is what `course_21M` does differently.

`pretraining/` holds the SeVAE, memory and behaviour-cloning loss histories.
