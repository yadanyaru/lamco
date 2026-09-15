"""Small synthetic smoke demo; no model download or GPU is required."""

from __future__ import annotations

import numpy as np

from lamco import lamco_select, normalize_rows


def main() -> None:
    rng = np.random.default_rng(7)
    classes, per_class, dimension = 3, 12, 8
    prototypes = normalize_rows(rng.normal(size=(classes, dimension)))
    labels = np.repeat(np.arange(classes), per_class)
    images = normalize_rows(prototypes[labels] + 0.55 * rng.normal(size=(labels.size, dimension)))
    result = lamco_select(
        images,
        prototypes,
        labels,
        keep_fraction=0.25,
        rho=0.3,
        alpha=1.0,
        gamma=1.0,
        temperature=1.0,
    )
    print("selected indices:", result.indices.tolist())
    print("class counts:", {c: item.budget for c, item in result.by_class.items()})
    print("VLM-label agreement rate:", round(float(result.agrees_with_label.mean()), 3))


if __name__ == "__main__":
    main()

