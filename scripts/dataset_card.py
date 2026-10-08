"""Write a merged export's dataset card beside it: dataset_card.md and card.json (autofly_ue5/dataset/card.py).

    env -u PYTHONPATH .venv/bin/python scripts/dataset_card.py --export <out>/<name>

Reads <out>/<name>/merge.json and dataset_statistics.json, which scripts/export_merged.py writes (with the card). The card
is derived from them, so running this again rewrites it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse  # noqa: E402
import json  # noqa: E402

from autofly_ue5.dataset.card import CARD_JSON, CARD_MD, write_card  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", type=Path, required=True, help="<out>/<name>: the directory holding merge.json")
    args = p.parse_args(argv)
    card = write_card(args.export)
    print(json.dumps({"card": str(args.export / CARD_MD), "card_json": str(args.export / CARD_JSON),
                      "episodes": card["totals"]["episodes"]["all"], "records": card["totals"]["records"]["all"],
                      "splits": {s: card["totals"]["episodes"][s] for s in ("train", "val")}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
