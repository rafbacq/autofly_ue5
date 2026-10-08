"""Closed-loop evaluation (spec §12 M6; plan 6 item 6): any policy flies frozen episodes in the simulator and is scored
as the paper scores its models (success rate, collision rate, path efficiency).

- `policy.py`: the `Policy` interface a policy implements (a VLA plugs in there), and the reference adapters: the SAC
  expert, a privileged straight-line baseline and a hover floor.
- `episodes.py`: the frozen evaluation episode files per condition (assets/eval_episodes/).
- `harness.py`: `run_condition`, which flies a policy on one condition file and writes the record and its summary.
- `metrics.py`: the rates with Wilson intervals, and PER as an interval.
"""
