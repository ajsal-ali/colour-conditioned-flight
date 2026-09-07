# Vision-Only Quadrotor Flight Through a Colour-Conditioned Obstacle Course

A memory-augmented reinforcement learning stack for flying a Crazyflie-class
quadrotor through a multi-station indoor obstacle course from onboard RGB-D
alone. The observation carries no gate poses, waypoints, or other privileged
geometry: the drone has to read the course from the camera and retain what it
saw once that geometry leaves the field of view.

Demo (third-person view; top-right inset is the onboard RGB the encoder
receives):

![Course demo](media/course_flight.gif)

The stack follows [MAVRL](https://github.com/tudelft/mavrl) (Yu et al., IEEE
RA-L 2025) in spirit (latent encoder, memory, then policy), but the task and
several design choices differ as described below. For installation and how to
run every script, see [`mavrl/README.md`](mavrl/README.md).

## Environment

The arena is a corridor. The drone spawns behind an entry wall, flies through
zero to three bar stations, passes fixed obstacles (tubes, boxes, a turbine)
that score nothing but must not be hit, and leaves through an exit wall.

![Course overview](media/course_overview.png)

The scored gates are colour-conditioned. At the entry and exit walls the red
opening is the legal path and the blue opening is a decoy of similar size. At
each bar station a horizontal bar between two posts is either red or blue: red
means pass above the bar, blue means pass below it.

![Gate types](media/course_gates.png)

Bar heights are redrawn every episode from a discrete set per colour. Two of
those heights are held out of policy training so evaluation can ask whether the
agent learned the colour rule or memorised the altitudes it saw.

Memory is required because the onboard camera is fixed and forward-facing.
After a descent under a low blue bar, the next high red bar often sits outside
the vertical field of view; the final approach is flown on information from
earlier frames, not on what is currently visible.

Each observation is a 128x128 RGB-D image plus a 16-D proprioceptive vector
(body-frame gravity direction, gyro, body velocity, velocity-command state, and
previous action). The action is body-frame acceleration (3) plus a yaw rate
(1). The policy runs at 10 Hz over a 60 Hz PID loop on 240 Hz physics.

## Method

### Semantic VAE

The semantically-enhanced VAE is Kulkarni et al.'s idea (arXiv:2307.11522):
attaching a segmentation objective to a compression autoencoder so that
semantically important but visually small structure -- in their case thin
obstacles a plain reconstruction loss would smooth away -- survives into the
latent. The same failure applies here for a different reason. A bar occupies
1-4% of the frame, and its colour, not its geometry, decides which side is
legal, so a purely unsupervised reconstruction loss has little reason to
preserve the one property the task depends on.

A six-convolution encoder maps each RGB-D frame to a 64-D latent. Three decoder
heads reconstruct RGB, depth, and a semantic segmentation map. The
reconstruction loss is proximity-weighted so near geometry (what the drone can
hit) dominates capacity. The segmentation head is what keeps red and blue
separable in the latent.

![SeVAE reconstructions](media/sevae_samples.png)

The figure shows noisy RGB as the encoder receives it, clean RGB, the
reconstruction, then depth and segmentation targets against predictions.

### Temporal attention with four memory tokens

The deployed memory is windowed temporal attention with four recurrent memory
tokens (RMT-style; cf. Bulatov et al., 2022). MAVRL uses an LSTM here; the
choice of attention is motivated by the shape of this task.

What the drone has to recall is not a summary of the last few seconds but one
specific earlier view: the frame in which the upcoming bar's colour was legible,
before the approach pushed it out of the vertical field of view. An LSTM folds
every frame into a single hidden vector that is overwritten at each step, so
retrieving one particular past observation competes with everything else the
state is holding. Attention over a window of latents leaves those frames
individually addressable, which is closer to the operation the task actually
needs.

A bare window is hard-capped at its length, though, which would make it strictly
weaker than a recurrent state rather than a different trade-off. The four memory
tokens are what close that gap: their output at step t is their input at t+1, so
memory extends past the window while attention keeps sharp access inside it.

During memory training an auxiliary head must reconstruct the image from about
two seconds earlier as well as the current one. The attention window is shorter
than that offset, so the past frame is not sitting in the buffer and must be
carried in the tokens -- otherwise the objective could be satisfied by copying
rather than remembering, and only the LSTM would face a real memory task.

Both backbones are implemented behind `--memory-type`.

![Memory reconstructions](media/memory_attention_m4_samples.png)

Both reconstructions are decoded from the same latent at time t. The top pair
is the frame from twenty policy steps ago; the bottom pair is the current
frame. When the current view no longer contains the red bar, the past
reconstruction still places it: that content is coming from memory, not from
the input.

### Policy

An actor-critic reads the memory state concatenated with proprioception and
outputs the acceleration and yaw command. Training warm-starts from behaviour
cloning on scripted (or optional teleop) demos, then continues with recurrent
PPO. A decaying fraction of environments can also be flown by the scripted
pilot under a self-imitation learning (SIL) objective, so good expert
transitions reinforce the policy without staying on forever. The course
curriculum grows from entry-to-exit only up to the full three-bar course.

## Relation to MAVRL

MAVRL demonstrated memory-augmented latent flight in unstructured clutter from
depth. This project keeps that overall recipe and changes the parts the colour
course requires.

The task is a structured arena with an explicit colour rule rather than random
obstacles, so the encoder takes RGB-D instead of depth alone and carries the
segmentation head of Kulkarni et al.'s SeVAE, here supervising colour rather
than thin-obstacle geometry, with proximity-weighted reconstruction. Memory is
temporal attention with four recurrent tokens rather than an LSTM. Speed
variation appears through curriculum and progress /
overspeed shaping on a fixed corridor instead of MAVRL's explicit
varying-speed objective. Data for the encoder and warm-start come from a
scripted pilot (and optional teleop), with an optional pilot mix inside PPO,
rather than from online policy rollouts alone. Simulation is the MuJoCo
Crazyflie stack in `multi_drone_mujoco`.

The point of the adaptation is rule-conditioned flight: colour, not geometry
alone, decides the legal path, and bars often leave the camera before the
commitment is finished.

## Code layout

`mavrl/course_world.py`, `course_gates.py`, and `course_aviary.py` define the
arena, gate logic, and Gymnasium environment. `sevae.py`, `memory.py`,
`policy.py`, and `ppo.py` are the encoder, memory backbones, actor-critic, and
recurrent PPO. `collect.py`, `teleop.py`, and `bc.py` build data and warm-start
weights. `train_sevae.py`, `train_memory.py`, and `train_course.py` are the
training stages. `evaluate.py` and `preview.py` evaluate the policy and render the
arena.

Install and run instructions live in [`mavrl/README.md`](mavrl/README.md).

## Training compute

Policy training ran on ParamShakti (IIT Kharagpur HPC) with EGL headless
rendering. Parallel environments share GL contexts across render workers so
VRAM scales with the number of contexts rather than the number of envs.

## Current results

The longest run to date is 35.7M steps on the full three-bar course: 0.29
success, 0.67 of the gates cleared on average. Training is ongoing. Policy
entropy diverges in the later half of that run -- the entropy bonus inflates
`log_std` against a clipped action space -- so the figure is a floor rather than
what the method settles at.

[`results/`](results/) has the per-rollout log, the training curves, and the
SeVAE, memory and policy weights.

## References

- Yu, Ferranti, et al. *MAVRL: Learn to Fly in Cluttered Environments with
  Varying Speed.* IEEE RA-L 2025. https://github.com/tudelft/mavrl
- Kulkarni, Nguyen, Alexis. *Semantically-enhanced Deep Collision Prediction
  for Autonomous Navigation using Aerial Robots.* IROS 2023.
  https://arxiv.org/abs/2307.11522
- Bulatov, Kuratov, Burtsev. *Recurrent Memory Transformer.* NeurIPS 2022.
- Tayal. *MuJoCo-Drones-Gym.* arXiv:2606.08039, 2026.
  https://arxiv.org/abs/2606.08039
- MuJoCo Menagerie, Bitcraze Crazyflie 2.x MJCF model.

## License

MIT
