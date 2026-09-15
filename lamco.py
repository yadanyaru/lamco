

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ClassSelection:


    budget: int
    core: Array
    boundary: Array
    final: Array


@dataclass(frozen=True)
class LaMCoResult:


    indices: Array
    margins: Array
    lcms: Array
    agrees_with_label: Array
    by_class: Dict[int, ClassSelection]


def _as_float_2d(name: str, value: Array) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2-D array")
    return array


def normalize_rows(value: Array, eps: float = 1e-12) -> Array:
    """L2-normalize rows, rejecting zero vectors."""

    array = _as_float_2d("value", value)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("cannot normalize a zero-length row")
    return array / norms


def semantic_margin_scores(
    image_embeddings: Array,
    class_prototypes: Array,
    labels: Array,
    *,
    temperature: float = 1.0,
    alpha: float = 1.0,
) -> tuple[Array, Array, Array]:

    images = normalize_rows(image_embeddings)
    prototypes = normalize_rows(class_prototypes)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must be a length-N vector")
    if images.shape[1] != prototypes.shape[1]:
        raise ValueError("image and prototype dimensions must match")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(labels == np.floor(labels)):
            raise ValueError("labels must contain integer class IDs")
    labels = labels.astype(np.int64, copy=False)
    if np.any(labels < 0) or np.any(labels >= prototypes.shape[0]):
        raise ValueError("labels must be in [0, number_of_prototypes)")
    if prototypes.shape[0] < 2:
        raise ValueError("semantic margins require at least two classes")

    logits = (images @ prototypes.T) / temperature
    row = np.arange(images.shape[0])
    true_logits = logits[row, labels]
    competing = logits.copy()
    competing[row, labels] = -np.inf
    margins = true_logits - competing.max(axis=1)
    predictions = logits.argmax(axis=1)
    agrees = predictions == labels
    # Agreement implies a non-negative margin (ties follow argmax ordering).
    lcms = np.where(agrees, np.exp(np.clip(-alpha * margins, -745.0, 709.0)), 0.0)
    return margins, lcms, agrees


def _k_center(features: Array, global_indices: Array, budget: int) -> Array:

    if budget <= 0 or global_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    budget = min(budget, global_indices.size)
    z = features[global_indices]
    centroid = z.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-12:
        centroid = centroid / centroid_norm
        first = int(np.argmax(z @ centroid))
    else:
        first = 0

    selected = [first]
    nearest_distance = 1.0 - z @ z[first]
    nearest_distance[first] = -np.inf
    while len(selected) < budget:
        nxt = int(np.argmax(nearest_distance))
        selected.append(nxt)
        nearest_distance = np.minimum(nearest_distance, 1.0 - z @ z[nxt])
        nearest_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
    return global_indices[np.asarray(selected, dtype=np.int64)]


def _weighted_facility_location(
    features: Array,
    global_indices: Array,
    weights: Array,
    candidate_mask: Array,
    budget: int,
) -> Array:

    if budget <= 0 or global_indices.size == 0:
        return np.empty(0, dtype=np.int64)
    z = features[global_indices]
    local_weights = weights[global_indices]
    candidates = np.flatnonzero(candidate_mask[global_indices])
    budget = min(budget, candidates.size)
    if budget == 0:
        return np.empty(0, dtype=np.int64)

    similarity = np.clip((1.0 + z @ z.T) / 2.0, 0.0, 1.0)
    coverage = np.zeros(global_indices.size, dtype=np.float64)
    chosen: list[int] = []
    available = np.zeros(global_indices.size, dtype=bool)
    available[candidates] = True

    for _ in range(budget):
        active = np.flatnonzero(available)
        improvements = np.maximum(similarity[:, active] - coverage[:, None], 0.0)
        gains = (local_weights[:, None] * improvements).sum(axis=0)
        # np.argmax chooses the lowest local/global index on exact ties.
        nxt = int(active[int(np.argmax(gains))])
        chosen.append(nxt)
        available[nxt] = False
        coverage = np.maximum(coverage, similarity[:, nxt])
    return global_indices[np.asarray(chosen, dtype=np.int64)]


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def lamco_select(
    image_embeddings: Array,
    class_prototypes: Array,
    labels: Array,
    *,
    budget: int | None = None,
    keep_fraction: float | None = None,
    rho: float = 0.3,
    alpha: float = 1.0,
    gamma: float = 1.0,
    temperature: float = 1.0,
) -> LaMCoResult:

    images = normalize_rows(image_embeddings)
    prototypes = normalize_rows(class_prototypes)
    labels_array = np.asarray(labels)
    n = images.shape[0]
    if (budget is None) == (keep_fraction is None):
        raise ValueError("provide exactly one of budget or keep_fraction")
    if keep_fraction is not None:
        if not 0 < keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        budget = _round_half_up(keep_fraction * n)
    assert budget is not None
    if not isinstance(budget, (int, np.integer)) or not 0 < int(budget) <= n:
        raise ValueError("budget must be an integer in [1, N]")
    budget = int(budget)
    if not 0 <= rho <= 1:
        raise ValueError("rho must be in [0, 1]")
    if gamma < 0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and non-negative")

    margins, lcms, agrees = semantic_margin_scores(
        images, prototypes, labels_array, temperature=temperature, alpha=alpha
    )
    labels_int = labels_array.astype(np.int64, copy=False)
    present_classes = np.unique(labels_int)
    if budget < present_classes.size:
        raise ValueError("budget must be at least the number of represented classes")
    counts = np.asarray([(labels_int == c).sum() for c in present_classes])
    if np.any(counts == 0):
        raise AssertionError("internal class counting error")

    base, remainder = divmod(budget, present_classes.size)
    class_budgets = np.full(present_classes.size, base, dtype=np.int64)
    class_budgets[:remainder] += 1
    if np.any(class_budgets > counts):
        # Even allocation can exceed a small class. Redistribute deterministically.
        overflow = 0
        for pos in range(present_classes.size):
            if class_budgets[pos] > counts[pos]:
                overflow += int(class_budgets[pos] - counts[pos])
                class_budgets[pos] = counts[pos]
        while overflow:
            changed = False
            for pos in range(present_classes.size):
                if class_budgets[pos] < counts[pos]:
                    class_budgets[pos] += 1
                    overflow -= 1
                    changed = True
                    if overflow == 0:
                        break
            if not changed:
                raise ValueError("unable to allocate the requested budget")

    weights = np.power(lcms, gamma) if gamma != 0 else np.ones_like(lcms)
    by_class: Dict[int, ClassSelection] = {}
    final_all: list[int] = []
    for class_id, class_budget in zip(present_classes, class_budgets):
        members = np.flatnonzero(labels_int == class_id)
        boundary_budget = _round_half_up(rho * int(class_budget))
        core_budget = int(class_budget) - boundary_budget
        core = _k_center(images, members, core_budget)
        boundary = _weighted_facility_location(
            images, members, weights, agrees, boundary_budget
        )

        merged: list[int] = []
        seen: set[int] = set()
        for index in np.concatenate((core, boundary)):
            index_int = int(index)
            if index_int not in seen:
                merged.append(index_int)
                seen.add(index_int)

        # Stable descending LCMS, then ascending original index.
        refill_order = members[np.lexsort((members, -lcms[members]))]
        for index in refill_order:
            if len(merged) == class_budget:
                break
            index_int = int(index)
            if index_int not in seen:
                merged.append(index_int)
                seen.add(index_int)
        final = np.asarray(merged, dtype=np.int64)
        if final.size != class_budget:
            raise AssertionError("failed to preserve the per-class budget")
        by_class[int(class_id)] = ClassSelection(
            budget=int(class_budget), core=core, boundary=boundary, final=final
        )
        final_all.extend(merged)

    indices = np.asarray(final_all, dtype=np.int64)
    if indices.size != budget or np.unique(indices).size != budget:
        raise AssertionError("failed to produce an exact, duplicate-free budget")
    return LaMCoResult(
        indices=indices,
        margins=margins,
        lcms=lcms,
        agrees_with_label=agrees,
        by_class=by_class,
    )

