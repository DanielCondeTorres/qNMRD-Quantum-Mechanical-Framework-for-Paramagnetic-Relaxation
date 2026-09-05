"""Create a machine-readable, no-fit Zeeman-limit benchmark for qNMRD.

Example
-------
python scripts/benchmark_zeeman_limit.py --output results/data/zeeman_limit.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qnmrd.validation.zeeman_limit import zeeman_limit_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/data/zeeman_limit.json")
    parser.add_argument("--max-error", type=float, default=0.01,
                        help="Fail if any relative error exceeds this value.")
    args = parser.parse_args()

    fields = np.array([0.1, 1.0, 2.0, 10.0])
    payload = {
        "description": "No-fit D=E=0 comparison of qNMRD Redfield and SBM.",
        "parameters": {"r_m": 3.0e-10, "tau_c_s": 50.0e-12, "temp_K": 298.15},
        "systems": {},
    }
    worst_error = 0.0
    for spin in (2.5, 3.5):
        values = zeeman_limit_errors(spin, fields)
        worst_error = max(worst_error, float(values["relative_error"].max()))
        payload["systems"][f"S={spin:g}"] = {
            key: value.tolist() for key, value in values.items()
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")

    print(f"Wrote {args.output}; worst relative error = {worst_error:.3%}")
    if worst_error > args.max_error:
        raise SystemExit(
            f"Zeeman-limit benchmark failed: {worst_error:.3%} > {args.max_error:.3%}"
        )


if __name__ == "__main__":
    main()
