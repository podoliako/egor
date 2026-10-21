"""Minimal scikit-fmm continuity controls, independent of tomography and ray G.

    OPENBLAS_NUM_THREADS=1 venv/bin/python \
        projects/EMTomo/studies/station_seed/continuity_controls.py

No model rounding, solver patching or replacement is performed. Order 1 is a
control, not a proposed new production default. A passing physical continuity
check requires changes in T to vanish with changes in velocity.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform

import numpy as np
import skfmm


def statistics(difference):
    index = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    return {"max_abs_difference_s": float(np.abs(difference).max()),
            "rms_difference_s": float(np.mean(difference ** 2) ** 0.5),
            "max_abs_index": [int(i) for i in index]}


def compare(phi, velocity, perturbed, h):
    result = {"max_velocity_change_m_s": float(np.abs(perturbed - velocity).max()), "orders": {}}
    for order in (1, 2):
        original = np.asarray(skfmm.travel_time(phi, velocity, dx=h, order=order))
        changed = np.asarray(skfmm.travel_time(phi, perturbed, dx=h, order=order))
        result["orders"][str(order)] = statistics(changed - original)
    return result


def main():
    h, shape = 125.0, (48, 32, 32)
    source = np.array((1000.0, 875.0, 0.0))
    axes = [(np.arange(n) + 0.5) * h - s for n, s in zip(shape, source)]
    distance = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
    velocity = np.full(shape, 4000.0)
    perturbed = velocity.copy()
    perturbed[4:8, 4:8, :4] = np.nextafter(4000.0, np.inf)
    # Both media satisfy the corrected homogeneous-sphere convention.
    inside = distance <= 375.0
    velocity[inside] = 4000.0
    perturbed[inside] = 4000.0
    sphere = compare(distance - 375.0, velocity, perturbed, h)
    sphere.update(shape=list(shape), cell_size_m=h, source_m=source.tolist(), radius_m=375.0,
                  changed_slice="[4:8,4:8,:4], excluding sphere", returned_field_dtype="float64",
                  common_R_over_v_added=False)

    phi = np.ones((24, 24))
    phi[7:9, 7:9] = 0.0
    velocity2 = np.full(phi.shape, 4000.0)
    perturbed2 = velocity2.copy()
    perturbed2[4:8, 4:8] = np.nextafter(4000.0, np.inf)
    nodes = compare(phi, velocity2, perturbed2, h)
    nodes.update(shape=list(phi.shape), zero_seed_slice="[7:9,7:9]", changed_slice="[4:8,4:8]",
                 negative_phi_nodes=0, homogeneous_sphere=False)

    result = {"python": platform.python_version(), "numpy": np.__version__, "skfmm": skfmm.__version__,
              "sphere_3d": sphere, "four_zero_nodes_2d": nodes,
              "interpretation": "Order-2 jumps occur without a sphere or inward propagation; order 1 is a control only."}
    path = Path(__file__).with_name("continuity_results.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
