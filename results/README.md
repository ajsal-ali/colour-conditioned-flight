# Training logs

Every run's `log.jsonl` carries one JSON object per rollout: timesteps, stage,
success rate, mean gates cleared, average ground velocity, heading error, the
terminal-reason histogram, and the PPO diagnostics (policy-gradient loss, value
loss, entropy, approximate KL, self-imitation loss, expert fraction).
`curves.png` is the same data plotted.

| Run | Steps | Outcome |
|---|---|---|
| `course_35M` | 35.7M | Best to date. Stage 3 throughout, attention memory with 4 tokens, BC warm start, decaying pilot mix. Success 0.29, gates 0.67. Entropy diverges — see below. |
| `course_5.7M_stage3` | 5.7M | Stage 3, no pilot mix. Never left 0% success; terminal reasons are almost entirely `collision`. |
| `course_5.5M_stage1` | 5.5M | Stalled at stage 1 with `wrong_side` on 100% of episodes — the policy settled on one side of the bar regardless of colour. |

`pretraining/` holds the SeVAE, memory and behaviour-cloning loss histories.

## The entropy divergence

In `course_35M`, policy entropy falls to -4.4 by 10M steps and then climbs
without bound to +16.4. For a 4-D diagonal Gaussian that is a standard deviation
of roughly 14, against an action space the trainer clips to +/-1.

The entropy bonus is paying for standard deviation the environment cannot see:
past the clip the extra spread changes nothing about behaviour, so there is no
countervailing gradient. Sampled actions over the later half of the run are
effectively bang-bang, which is why success plateaus near 0.25 while episode
return keeps drifting up.

The fix is a squashed (tanh) Gaussian with the log-determinant correction, or a
bounded `log_std`. Numbers from this run should be read as a floor, not as the
method's ceiling.

No model weights are committed. They are 26-38 MB each and are reproducible
from `mavrl/README.md`.
