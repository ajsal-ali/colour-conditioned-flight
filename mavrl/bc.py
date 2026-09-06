#!/usr/bin/env python3
"""Sequence-aware behaviour cloning onto the recurrent policy.

Unlike rl/bc_pretrain.py, which shuffles frames i.i.d., this trains over whole
windows so the recurrent state means something. It also warm-starts from the
already-trained SeVAE and memory checkpoints rather than from scratch.

Only the actor is trained. The critic stays random, which is exactly why
train_course applies a warm-start freeze on the actor for the first stretch of
PPO -- otherwise a random value function destroys the cloned policy before it
learns anything.

Four things here exist because their absence cost a full training run:

**A validation set, and a rollout eval.** The first version logged training MSE
and nothing else. MSE fell to 0.016 and looked healthy; the checkpoint flew into
the first bar, and 448k steps of PPO were spent discovering that. Training loss
cannot see covariate shift, which is the entire failure mode of BC on an
integrating action space, so `--eval-episodes` actually flies the thing.

**Successful episodes only.** The collector deliberately keeps flying after a
missed gate so the SeVAE sees failure states. Cloning that teaches the policy to
reproduce the crashes at equal weight. `--all-episodes` opts back out.

**The same latent sampling as deployment.** `act()` draws z from the SeVAE
posterior; training against the posterior mean instead hands the policy head a
clean input it never sees again after BC. Both paths use `sample=True` here.

**Two phases.** The heads train against a frozen memory, then the memory is
unfrozen at a much lower LR. Doing both at once lets first-epoch gradients from
a randomly-initialised head wreck a memory model that took a whole stage to
train.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mavrl.config import AUX_SEGMENTS
from mavrl.dataset import SequenceDataset
from mavrl.memory import AuxReconstructionHead
from mavrl.policy import MavrlActorCritic


def load_pretrained(policy: MavrlActorCritic, sevae_path, memory_path,
                    device) -> None:
    if sevae_path and Path(sevae_path).exists():
        policy.sevae.load_state_dict(
            torch.load(sevae_path, map_location=device)["model"])
        print("loaded SeVAE   ", sevae_path)
    if memory_path and Path(memory_path).exists():
        ckpt = torch.load(memory_path, map_location=device)
        policy.memory.load_state_dict(ckpt["memory"])
        # The aux head's width is set by the lambda configuration stage 3 ran
        # with, which the policy has no way to know. Rebuild it to match rather
        # than dropping the checkpoint on a shape mismatch.
        segments = tuple(ckpt.get("aux_segments", AUX_SEGMENTS))
        if segments != policy.aux.segments:
            policy.aux = AuxReconstructionHead(segments=segments).to(device)
        policy.aux.load_state_dict(ckpt["aux"])
        print("loaded memory  ", memory_path, f"(aux={list(segments)})")


def batch_tensors(batch, device):
    img = torch.as_tensor(batch["image"], device=device)
    img = img.permute(0, 1, 4, 2, 3).contiguous()
    return (img,
            torch.as_tensor(batch["proprio"], device=device),
            torch.as_tensor(batch["action"], device=device))


def run_epoch(policy, ds, batch_size, rng, device, opt=None):
    """One pass over `ds`. `opt=None` means evaluate: no grad, no update."""
    training = opt is not None
    policy.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in ds.iter_batches(batch_size, rng):
            img, proprio, act = batch_tensors(batch, device)
            state = policy.initial_state(img.shape[0], device)
            # sample=True to match act(): the head has to learn on the
            # stochastic latents it will actually be given, not on the mean.
            dist, _, _, _ = policy.sequence(img, proprio, state, sample=True)
            loss = F.mse_loss(dist.mean, act)
            if training:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for q in policy.parameters() if q.requires_grad], 1.0)
                opt.step()
            total += loss.item()
            n += 1
    return total / max(1, n), n


@torch.no_grad()
def rollout_success(policy, device, episodes: int, n_stations: int,
                    split: str, seed: int) -> dict:
    """Fly the cloned policy. The only measurement that catches covariate shift.

    Imported lazily so this module still loads on a box without mujoco.
    """
    from mavrl import config as C
    from mavrl.course_aviary import CourseAviary
    from mavrl.course_world import sample_layout
    from mavrl.policy import obs_to_tensors

    rng = np.random.default_rng(seed)
    was_training = policy.training
    policy.eval()
    env = CourseAviary(layout=sample_layout(rng, n_stations, split), seed=seed)
    ok, gates = 0, []
    try:
        for _ in range(episodes):
            env.set_layout(sample_layout(rng, n_stations, split))
            obs, _ = env.reset()
            state = policy.initial_state(1, device)
            info: dict = {}
            for _ in range(int(C.t_max(n_stations) * C.POLICY_FREQ) + 5):
                img, prop = obs_to_tensors(obs, device)
                action, _, _, state = policy.act(img, prop, state,
                                                 deterministic=True)
                obs, _, term, trunc, info = env.step(
                    action.squeeze(0).cpu().numpy())
                if term or trunc:
                    break
            ok += int(info.get("is_success", False))
            gates.append(info.get("gates_cleared", 0)
                         / max(1, info.get("n_gates", 1)))
    finally:
        env.close()
        policy.train(was_training)
    return {"success": ok / max(1, episodes),
            "gates": float(np.mean(gates)) if gates else 0.0,
            "episodes": episodes}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--sevae", type=Path, default=Path("ckpt/sevae.pt"))
    p.add_argument("--memory", type=Path, default=None)
    p.add_argument("--out", type=Path, default=Path("ckpt/bc_init.pt"))
    p.add_argument("--memory-type", default="lstm",
                   choices=("lstm", "attention"))
    p.add_argument("--mem-tokens", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40,
                   help="phase 1: heads only, memory frozen")
    p.add_argument("--finetune-epochs", type=int, default=10,
                   help="phase 2: memory unfrozen at --finetune-lr")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-5,
                   help="two orders below --lr: the memory model is already "
                        "trained, and this is a nudge rather than a retrain")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=0,
                   help="must match between the train and val views, or the "
                        "two halves overlap")
    p.add_argument("--all-episodes", action="store_true",
                   help="clone failed runs too (default: completed only)")
    p.add_argument("--eval-episodes", type=int, default=20,
                   help="rollout eval size; 0 disables it (needs mujoco)")
    p.add_argument("--eval-every", type=int, default=5, help="epochs")
    p.add_argument("--eval-stations", type=int, default=3)
    p.add_argument("--eval-split", default="train",
                   choices=("train", "eval", "all"))
    p.add_argument("--freeze-encoder", action="store_true", default=True)
    p.add_argument("--train-encoder", dest="freeze_encoder",
                   action="store_false")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    memory_path = args.memory or Path(
        f"ckpt/memory_{args.memory_type}"
        f"{'' if args.mem_tokens is None else f'_m{args.mem_tokens}'}.pt")

    policy = MavrlActorCritic(memory_type=args.memory_type,
                              mem_tokens=args.mem_tokens,
                              freeze_encoder=args.freeze_encoder).to(device)
    load_pretrained(policy, args.sevae, memory_path, device)

    common = dict(seq_len=args.seq_len, keys=("image", "proprio", "action"),
                  only_success=not args.all_episodes,
                  val_frac=args.val_frac, split_seed=args.split_seed)
    train_ds = SequenceDataset(args.data, split="train", **common)
    val_ds = SequenceDataset(args.data, split="val", **common)
    print(f"train {len(train_ds)} windows / {len(train_ds.episodes)} episodes | "
          f"val {len(val_ds)} windows / {len(val_ds.episodes)} episodes | "
          f"memory={args.memory_type} "
          f"encoder={'frozen' if args.freeze_encoder else 'trainable'}")
    if train_ds.n_dropped:
        print(f"dropped {train_ds.n_dropped} incomplete episode(s) "
              f"-- --all-episodes keeps them")

    # Variance of the labels themselves. A policy that has learned nothing but
    # the dataset mean scores about this, so it is the floor that makes a val
    # MSE mean something. rl/bc_pretrain.py:70 uses the same test.
    acts = np.concatenate([b["action"].reshape(-1, b["action"].shape[-1])
                           for b in val_ds.iter_batches(args.batch_size, rng)])
    action_var = float(acts.var(0).mean())
    print(f"val action variance {action_var:.4f} "
          f"-- mean-collapse scores about this; aim well under half")

    def set_memory_trainable(on: bool) -> None:
        for prm in policy.memory.parameters():
            prm.requires_grad = on

    phases = [(1, args.epochs, args.lr, False),
              (2, args.finetune_epochs, args.finetune_lr, True)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hist_path = args.out.parent / "bc_history.json"
    history: list = []
    best = math.inf
    epoch = 0

    for phase, n_epochs, lr, memory_on in phases:
        if n_epochs <= 0:
            continue
        set_memory_trainable(memory_on)
        trainable = [q for q in policy.parameters() if q.requires_grad]
        opt = torch.optim.Adam(trainable, lr=lr)
        print(f"\n-- phase {phase}: {n_epochs} epochs, lr={lr:g}, "
              f"memory={'trainable' if memory_on else 'frozen'}, "
              f"{sum(q.numel() for q in trainable):,} trainable params")

        for _ in range(n_epochs):
            tr_mse, n_batches = run_epoch(policy, train_ds, args.batch_size,
                                          rng, device, opt)
            if n_batches == 0:
                raise SystemExit(
                    f"epoch {epoch} ran zero batches: no shard holds "
                    f"{args.batch_size} windows of {args.seq_len} frames. "
                    f"Lower --batch-size or --seq-len.")
            val_mse, _ = run_epoch(policy, val_ds, args.batch_size, rng, device)
            row = {"epoch": epoch, "phase": phase, "lr": lr,
                   "mse": tr_mse, "val_mse": val_mse}

            if (args.eval_episodes and args.eval_every
                    and (epoch + 1) % args.eval_every == 0):
                row["rollout"] = rollout_success(
                    policy, device, args.eval_episodes, args.eval_stations,
                    args.eval_split, args.seed + epoch)

            # Selection is on val, never on train. The whole point is that the
            # training curve kept improving while the policy got worse.
            if val_mse < best:
                best = val_mse
                row["best"] = True
                torch.save({"policy": policy.state_dict(),
                            "memory_type": args.memory_type,
                            "mem_tokens": args.mem_tokens,
                            "aux_segments": list(policy.aux.segments),
                            "val_mse": val_mse, "epoch": epoch}, args.out)

            history.append(row)
            msg = (f"epoch {epoch:3d} [p{phase}]  mse={tr_mse:.5f}  "
                   f"val={val_mse:.5f}{'  *' if row.get('best') else ''}")
            if "rollout" in row:
                msg += (f"   rollout success={row['rollout']['success']:.2f} "
                        f"gates={row['rollout']['gates']:.2f}")
            print(msg, flush=True)
            hist_path.write_text(json.dumps(history, indent=2))
            epoch += 1

    if best > 0.5 * action_var:
        print(f"\nWARNING: best val mse {best:.4f} vs action variance "
              f"{action_var:.4f} -- the policy may have collapsed to "
              f"predicting the mean.")
    print(f"\nsaved {args.out} (best val_mse {best:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
