# LaMCo minimal anonymous reproduction

This repository reproduces the smallest self-contained algorithmic core of
**LaMCo: Label-Aware Margin Coreset Selection with Vision-Language Priors**.
It implements selection only and intentionally does not contain paper author,
institution, account, machine, or absolute-path information.

## What is reproduced

Given normalized (or unnormalized) frozen VLM image embeddings `z`, one text
prototype per class `p`, and integer labels `y`, the code implements:

1. Semantic logits and the label-conditioned margin
   `m_i = l[i,y_i] - max_{c != y_i} l[i,c]`.
2. The consistency-gated score
   `LCMS_i = 1[argmax(l_i) = y_i] * exp(-alpha * m_i)`.
3. An even deterministic class budget, split into a cosine k-center core and
   an LCMS-weighted facility-location boundary branch (`rho=0.3` by default).
4. Ordered merge, de-duplication, and within-class LCMS refill to preserve the
   exact global and per-class budgets.

The returned indices refer to the original input row order. The frozen VLM is
used only for selection; train downstream models on the selected images and
their original labels, not on VLM logits.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

Minimal API use:

```python
import numpy as np
from lamco import lamco_select

data = np.load("embeddings.npz")
result = lamco_select(
    data["image_embeddings"],    # [N, D]
    data["class_prototypes"],    # [C, D], row c corresponds to label c
    data["labels"],              # [N], integer IDs in [0, C)
    keep_fraction=0.2,
    rho=0.3,
    alpha=1.0,
    gamma=1.0,
    temperature=1.0,
)
np.save("selected_indices.npy", result.indices)
```

Or use the CLI with an NPZ containing exactly the three arrays above:

```bash
python select.py embeddings.npz --keep-fraction 0.2 \
  --rho 0.3 --alpha 1.0 --gamma 1.0 --temperature 1.0 \
  --output selected_indices.npy
```

On PowerShell, replace the line continuations with backticks or enter the
command on one line.

## Producing VLM inputs

Use a frozen CLIP-family model upstream:

1. Encode every image once in mini-batches, L2-normalize, and cache the result.
2. For each class name, encode several natural-language prompt templates.
3. Normalize each prompt embedding, average embeddings for the class, then
   normalize the resulting class prototype again.
4. Store arrays as `image_embeddings`, `class_prototypes`, and `labels` in NPZ.

The selection module stays independent of a specific CLIP library so it is
small, testable, and usable with cached features from OpenCLIP, Transformers,
or another VLM implementation.

## Reproduction assumptions and paper gaps

The paper specifies `rho=0.3` but does not report numerical values for
`alpha`, `gamma`, the temperature/logit scale, the exact prompt templates, the
k-center first seed, or integer rounding. Therefore the published accuracy
table cannot be exactly regenerated from the PDF alone. This code exposes the
missing hyperparameters and uses transparent deterministic defaults:

- `alpha=1`, `gamma=1`, `temperature=1`; if the encoder provides logit scale
  `s`, pass `temperature=1/s`.
- global size and boundary split use round-half-up;
- class-budget remainders go to ascending class IDs;
- k-center starts at the sample nearest the normalized class centroid;
- exact facility-location greedy is used instead of lazy greedy (same greedy
  objective/result, simpler code), with ascending index tie-breaking;
- the facility-location universe is the current class and candidates must pass
  the label-consistency gate; if too few pass, final LCMS refill may include
  zero-score class members solely to preserve the requested budget.

These choices are implementation assumptions, not claims that they were the
authors' hidden settings. For a strict benchmark reproduction, obtain the
missing configuration and prompts from the authors/artifact and pass them to
the exposed arguments.

## Complexity

For class size `n_c`, the minimal exact boundary implementation materializes an
`n_c x n_c` similarity matrix and uses `O(n_c^2)` memory. This is convenient
for CIFAR-scale class partitions. A large or long-tailed dataset should replace
it with chunked/lazy gain evaluation without changing the public API.

## Anonymous-release checklist

- No author or institution names are present.
- No email, username, host name, source-PDF metadata, experiment service ID, or
  absolute local path is included.
- Package metadata uses an anonymous project name and has no author field.
- Generated caches, environments, arrays, and selected indices are ignored.
- Before submission, inspect commit author metadata separately; Git history is
  outside the files in this folder.
