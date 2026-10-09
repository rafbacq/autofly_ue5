# 2026-10-03 — Training throughput: where the time goes, what changes now, what waits for a measurement

A read-only investigation ran while s01d run 2 trained (no simulator was launched; numbers marked "live" were read
from that run with `nvidia-smi`). Everything here is a code-reading result unless it says measured.

## Where the time goes

- **One instance:** a step takes 74 ms (measured, `docs/gates/m2d_mover_probe.json`). 40 ms of that is clock pacing
  (40 ticks at the 1 ms real-time rate), 20 ms is the backend's collision-grace sleep, and about 12 ms is RPCs,
  rendering and readback. The Python observation work is negligible: about 0.9 ms in total, timed.
- **Four instances:** a worker step takes about 300 ms (13.4 steps/s in total). The GPU reads "96% busy" but draws only
  130–150 W of 450 W, with 4–11% memory bandwidth (live). Its time is sliced among five processes that each block on
  small readbacks, which is why N = 2 → 4 added almost nothing (13.0 → 13.4 steps/s, measured).
- **The training-time evaluator costs twice.** SB3's EvalCallback stops all four workers for each evaluation (about 10
  minutes per 20 s01d episodes). Its simulator, launched at the first evaluation, then renders unthrottled for the rest
  of the run, holding a share of the GPU's time slices.

## Adopted now (no change to what the expert sees)

- **No in-training evaluation.** Train with a huge `--eval-freq`, and score checkpoints with `scripts/eval_watch.py`
  (b819b20): the gate's machinery on the training-time evaluation seeds, one simulator per evaluation, stopped after
  it. Selection already uses the periodic checkpoints (`scripts/select_checkpoint.py`).

## Deferred until measured on a quiet GPU (all training-only; collection and gates keep today's settings)

1. **No RGB during training.** Don't subscribe to the scene camera. The camera-pose check then reads the depth message's
   pose, which comes from the same capture. Estimated +15–35% at N = 4.
2. **No collision-grace sleep.** A collision is published on the game thread before the camera queues its readback,
   and every topic shares one first-in-first-out channel, so the step's frame cannot arrive before its collision.
   Saves 20 ms per step. Adopt only if a late-collision counter stays at 0 over at least 10k steps.
3. **A nearly free main view** for training launches: `-ResX/-ResY` small and show-flags off. These apply to the
   viewport only, not to the capture components.
4. **Concurrent mover moves** (s01d): 12 moves take 3.5 ms concurrently against 15.1 ms sequentially (measured by the
   mover probe). But a refused move would then drop the client connection, so the refusal needs its own handling.
5. **Overlap SAC updates with env stepping**, and **save the replay buffer less often** (12.7 GB every 10k steps).

## A correctness concern found on the way: the image may be one capture behind its timestamp

Unreal 5.7 renders deferred scene captures inside the main view's render, later in the frame than the camera queues
its readback. So the pixels stamped at time T are probably from the previous capture. The stamped pose is T's, so the
camera-pose check cannot see this.

Today that is likely 10–50 sim-ms, a few centimetres at 2 m/s. It is the same in training, gates and collection, so
no result is invalid. But any change to frame pacing changes it, which is why items 1–3 wait for a **lag probe**: fly
straight at a known s01 pillar at 2 m/s and compare each step's centre depth with the stamped kinematics.

A rebuilt plugin could render captures immediately and capture once per step, giving exact images and plausibly 2–4×
throughput. That needs a new simulator binary and the M0/M1 re-verification, so it is the user's call.
