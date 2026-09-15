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
