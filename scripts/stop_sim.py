"""Stop one simulator instance launched by this project (refuses processes it did not launch).

env -u PYTHONPATH .venv/bin/python scripts/stop_sim.py --instance 0
"""

import argparse
import sys

from autofly_ue5.sim.process import stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--grace", type=float, default=30.0)
    args = parser.parse_args()
    print(stop(args.instance, grace_s=args.grace))
    return 0


if __name__ == "__main__":
    sys.exit(main())
