"""Runner for the *official* confetti-ts package, executed as a subprocess.

Why a subprocess: confetti-ts (https://github.com/serval-uni-lu/confetti) ships a
compiled Rust extension (via maturin/PyO3) and requires Python >= 3.12, while this
repository's own environment targets Python 3.11 (see pyproject.toml / .venv). The
two cannot coexist in one interpreter, so this script is meant to be invoked with a
*separate* Python >= 3.12 interpreter (see the "Vendoring" cell in
cf_confetti_forda_comparison.ipynb for how that environment is created), while the
comparison notebook itself keeps running in this repo's normal Python 3.11 venv.

Data crosses the process boundary only as plain ``.npy`` files (portable between
numpy versions) plus a ``state_dict`` checkpoint (plain tensors, no custom classes),
so there is no cross-interpreter pickling of anything exotic. This script re-builds
the model architecture itself (from ``examples/base/model.py``, which only imports
torch) and loads the weights into it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", required=True, help="Path to the repository root")
    ap.add_argument("--model-state-dict", required=True, help="Path to a SimpleCNN state_dict .pth file")
    ap.add_argument("--output-classes", type=int, required=True)
    ap.add_argument("--input-length", type=int, required=True)
    ap.add_argument("--instances-npy", required=True, help="(N, T, C) channels-last float32 array")
    ap.add_argument("--reference-npy", required=True, help="(M, T, C) channels-last float32 array")
    ap.add_argument("--out-dir", required=True, help="Directory to write per-instance results + summary.json")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=0.51)
    ap.add_argument("--n-partitions", type=int, default=3)
    ap.add_argument("--population-size", type=int, default=50)
    ap.add_argument("--max-generations", type=int, default=50)
    ap.add_argument("--proximity-distance", default="euclidean")
    args = ap.parse_args()

    import torch

    # Python auto-inserts this script's own directory (cfts/cf_confetti/) at the
    # front of sys.path — which also contains this repo's *own* `confetti.py`
    # module. That shadows the real, pip-installed `confetti` (confetti-ts)
    # package we actually want below, so drop it before importing that package.
    script_dir = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(script_dir)]

    sys.path.insert(0, args.repo_root)
    sys.path.insert(0, str(Path(args.repo_root) / "examples"))
    from base.model import SimpleCNN  # only depends on torch — safe to import here

    model = SimpleCNN(output_channels=args.output_classes, input_length=args.input_length)
    state = torch.load(args.model_state_dict, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # confetti's CONFETTI(model_path=...) only accepts a *path*; for .pt/.pth it does
    # `torch.load(path, weights_only=False)` and wraps the result in TorchModelAdapter.
    # Saving+loading happens within this single process/torch version, so this is safe.
    full_model_path = out_dir / "full_model.pt"
    torch.save(model, full_model_path)

    from confetti import CONFETTI

    instances = np.load(args.instances_npy).astype(np.float32)
    reference = np.load(args.reference_npy).astype(np.float32)

    explainer = CONFETTI(model_path=str(full_model_path))

    summary = []
    for i in range(len(instances)):
        record = {"index": int(i), "success": False, "runtime_s": None, "error": None}
        t0 = time.time()
        try:
            results = explainer.generate_counterfactuals(
                instances_to_explain=instances[i : i + 1],
                reference_data=reference,
                reference_weights=None,  # skip the CAM-guided naive stage — see notebook markdown
                alpha=args.alpha,
                theta=args.theta,
                n_partitions=args.n_partitions,
                population_size=args.population_size,
                maximum_number_of_generations=args.max_generations,
                proximity_distance=args.proximity_distance,
                verbose=False,
            )
            record["runtime_s"] = time.time() - t0

            if results is None or len(results) == 0:
                record["error"] = "No NUN found above theta, or optimization produced nothing"
            else:
                cf_set = results[0]
                record["original_label"] = int(cf_set.original_label)
                np.save(out_dir / f"nun_{i}.npy", np.asarray(cf_set.nearest_unlike_neighbour, dtype=np.float32))
                record["n_candidates"] = len(cf_set.all_counterfactuals)
                if cf_set.best is not None:
                    record["success"] = True
                    record["cf_label"] = int(cf_set.best.label)
                    np.save(out_dir / f"cf_{i}.npy", np.asarray(cf_set.best.counterfactual, dtype=np.float32))
                else:
                    record["error"] = "No best_solution selected among candidates"
        except Exception as exc:  # noqa: BLE001 - report to caller, keep going
            record["runtime_s"] = time.time() - t0
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()

        summary.append(record)
        print(
            f"[{i + 1}/{len(instances)}] success={record['success']} "
            f"runtime={record['runtime_s']:.1f}s error={record['error']}",
            flush=True,
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
