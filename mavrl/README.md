# `mavrl/` - how to run

Design / results: [root README](../README.md). This file is install + commands only.

## 1. Environment

```bash
# from the repo root
conda create -n rl_mujoco python=3.11 -y
conda activate rl_mujoco

pip install -e ".[mavrl]"        # torch, sb3, imageio, tqdm, matplotlib, ...
pip install -e ".[mavrl-gui]"    # optional: live windows (pygame)
```

GL backend (set before any `mujoco` import):

```bash
export MUJOCO_GL=egl      # headless / cluster / Docker
export MUJOCO_GL=glfw     # local machine with a display
```

```bash
python -m mavrl.glcheck   # must say GPU, not llvmpipe
```

---

## 2. Pipeline (in order)

Trained SeVAE, memory and policy weights ship with the repo -- see
[`results/README.md`](../results/README.md). The pipeline below is how they
were produced.

```bash
conda activate rl_mujoco
cd /path/to/colour-conditioned-flight

# 1) scripted-pilot dataset
python -m mavrl.collect --episodes 2000 --out data --split all --workers 6

# 2) semantic VAE
python -m mavrl.train_sevae --data data --out ckpt/sevae.pt

# 3) memory (attention + 4 tokens)
python -m mavrl.train_memory --data data --memory-type attention --mem-tokens 4

# 4) warm-start - pick ONE:
#    (a) behaviour cloning on the collected demos
python -m mavrl.bc --data data --memory-type attention --mem-tokens 4
#    (b) skip BC and use pilot-assisted PPO instead (see --expert-frac below)

# 5) PPO on the course
python -m mavrl.train_course --curriculum --init ckpt/bc_init.pt \
    --memory-type attention --mem-tokens 4 \
    --n-envs 60 --render-workers 15 --envs-per-batch 8 \
    --expert-frac 0.25 --expert-decay 1000000 \
    --out runs/course

# 6) evaluate
python -m mavrl.evaluate --model runs/course/final.pt \
    --memory-type attention --mem-tokens 4
```

**Pilot / demo alternatives to BC**

| Option | Command | Notes |
|---|---|---|
| Behaviour cloning | `python -m mavrl.bc ...` → `--init ckpt/bc_init.pt` | Offline imitation of scripted (or teleop) demos |
| Pilot-assisted PPO | `--expert-frac 0.25 --expert-decay 1e6` (no BC needed) | A decaying fraction of envs is flown by the scripted pilot; transitions enter the PPO buffer (self-imitation) |
| Manual demos | `python -m mavrl.teleop --out data_manual` then `merge_data` | Mix human flights into the dataset before SeVAE / BC |

You can use BC **and** `--expert-frac` together (warm-start + online pilot mix).

Optional human data:

```bash
python -m mavrl.teleop --episodes 30 --out data_manual
python -m mavrl.merge_data --out data_all data data_manual
# then point stages 2-4 at data_all
```

---

## 3. `train_course` parameters

```bash
python -m mavrl.train_course \
    --curriculum \
    --init ckpt/bc_init.pt \
    --memory-type attention --mem-tokens 4 \
    --n-envs 60 \
    --render-workers 15 \
    --envs-per-batch 8 \
    --n-steps 128 \
    --lr 3e-4 \
    --ent-coef 0.005 \
    --expert-frac 0.25 \
    --expert-decay 1000000 \
    --sil-coef 0.1 \
    --critic-warmup 30000 \
    --sensor-noise 1.0 \
    --split train \
    --timesteps 500000000 \
    --out runs/course
```

| Flag | Default | What it does |
|---|---|---|
| `--n-envs` | 8 | Parallel environments |
| `--render-workers` | `ceil(n_envs/4)` | Shared GL contexts (VRAM for rendering) |
| `--envs-per-batch` | `n_envs//2` | PPO minibatch size in envs (VRAM for backward) |
| `--n-steps` | 128 | Rollout length per env |
| `--memory-type` | `lstm` | Use `attention` for the main stack |
| `--mem-tokens` | config | **4** with attention |
| `--curriculum` | off | Stage advance + layout redraw |
| `--stage N` | unset | Pin stage (`3` = full course); keep `--curriculum` so heights still redraw |
| `--init` | none | Warm-start from `bc_init.pt` or a PPO ckpt |
| `--expert-frac` | 0.25 | Fraction of envs flown by the scripted pilot at step 0 |
| `--expert-decay` | 1e6 | Timesteps over which expert frac → 0 |
| `--sil-coef` | 0.1 | Weight on the self-imitation (pilot) loss |
| `--ent-coef` | 0.005 | Entropy bonus |
| `--lr` | 3e-4 | Adam learning rate |
| `--critic-warmup` | 30000 | Freeze actor this many steps after `--init` |
| `--sensor-noise` | 1.0 | Depth / camera noise scale (`0` disables) |
| `--split` | `train` | `train` holds out two bar heights; `eval` / `all` |
| `--timesteps` | 5e8 | Total env steps |
| `--out` | `runs2/course` | `log.jsonl`, `curves.png`, `ckpt_*.pt`, `final.pt` |
| `--no-shared-render` | off | One GL context per env (debug / control) |
| `--no-subproc` | off | Single-process vec env |

---

## 4. Preview rollouts

Renders episodes from a checkpoint to image strips; works headless under EGL.

```bash
MUJOCO_GL=egl python -m mavrl.preview \
    --model runs/course/final.pt \
    --memory-type attention --mem-tokens 4 \
    --episodes 4 --out preview
```

---

## 5. GL troubleshooting

| Symptom | Fix |
|---|---|
| `eglQueryString` on `None` | `apt install libegl1` or `bash scripts/gl_no_root.sh` then `. ~/.mavrl_gl_env` |
| `GL_RENDERER = llvmpipe` | NVIDIA ICD: `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` |
| Docker trains GPU / renders CPU | `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` |

```bash
docker build -t mavrl -f docker/Dockerfile .
docker run --rm --gpus all mavrl    # runs glcheck
```
