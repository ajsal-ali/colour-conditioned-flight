# Training logs and weights

`log.jsonl` carries one JSON object per rollout: timesteps, stage, success rate,
mean gates cleared, average ground velocity, heading error, the terminal-reason
histogram, and the PPO diagnostics (policy-gradient loss, value loss, entropy,
approximate KL, self-imitation loss, expert fraction). `curves.png` is the same
data plotted.

`course_35M` is the longest run to date: 35.7M steps, stage 3 throughout,
attention memory with four tokens, warm-started from behaviour cloning with a
decaying scripted-pilot mix. Success 0.29, mean gates cleared 0.67. Entropy
diverges, as described below.

`pretraining/` holds the SeVAE, memory and behaviour-cloning loss histories.

## Weights

| File | Size | What it is |
|---|---|---|
| [`../ckpt/sevae.pt`](../ckpt/sevae.pt) | 26 MB | Semantic VAE: RGB-D encoder plus RGB / depth / segmentation decoder heads |
| [`../ckpt/memory_attention_m4.pt`](../ckpt/memory_attention_m4.pt) | 6 MB | Temporal-attention memory, 4 recurrent tokens, with its auxiliary reconstruction head |
| `course_35M/ckpt_8320000.pt` | 37 MB | Full actor-critic at 8.3M steps |
| `course_35M/ckpt_9600000.pt` | 37 MB | Full actor-critic at 9.6M steps |

Both policy checkpoints carry the encoder and memory inside them, so
`--memory-type attention --mem-tokens 4` is required to load either:

```bash
python -m mavrl.evaluate --model results/course_35M/ckpt_9600000.pt \
    --memory-type attention --mem-tokens 4
```

## The entropy divergence

Policy entropy falls to -4.4 by 10M steps and then climbs without bound to
+16.4. For a 4-D diagonal Gaussian that is a standard deviation of roughly 14,
against an action space the trainer clips to +/-1.

The entropy bonus is paying for standard deviation the environment cannot see:
past the clip, extra spread changes nothing about behaviour, so no
countervailing gradient exists. Sampled actions over the later half of the run
are effectively bang-bang, which is why success plateaus near 0.25 while episode
return keeps drifting up.

The fix is a squashed (tanh) Gaussian with the log-determinant correction, or a
bounded `log_std`. Numbers from this run should be read as a floor, not as the
method's ceiling.
