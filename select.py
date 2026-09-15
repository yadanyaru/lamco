

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lamco import lamco_select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a LaMCo coreset")
    parser.add_argument("input", type=Path, help="NPZ with image_embeddings, class_prototypes, labels")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--budget", type=int)
    size.add_argument("--keep-fraction", type=float)
    parser.add_argument("--output", type=Path, default=Path("selected_indices.npy"))
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        missing = {"image_embeddings", "class_prototypes", "labels"} - set(data.files)
        if missing:
            raise KeyError(f"missing NPZ arrays: {sorted(missing)}")
        result = lamco_select(
            data["image_embeddings"],
            data["class_prototypes"],
            data["labels"],
            budget=args.budget,
            keep_fraction=args.keep_fraction,
            rho=args.rho,
            alpha=args.alpha,
            gamma=args.gamma,
            temperature=args.temperature,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result.indices)
    summary = {
        "selected": int(result.indices.size),
        "agreement_rate": float(result.agrees_with_label.mean()),
        "per_class": {str(c): item.budget for c, item in result.by_class.items()},
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

