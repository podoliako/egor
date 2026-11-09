"""Compare the restored synthetic truth with a smooth horizontal-average 1D start."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import gaussian_filter1d

import make_realistic_anomalies as truth_model

OUTPUT = Path(__file__).resolve().parents[1] / "figures/realistic_anomalies"
SMOOTHING_SIGMA_KM = 0.4


def main():
    source = OUTPUT / "model_previous.npz"
    with np.load(source, allow_pickle=False) as data:
        truth = data["velocity"].copy()
        cell_m = float(data["cell_size_m"])
        origin_m = data["origin_m"].copy()
    assert truth.shape == (200, 80, 80)
    cell_km = cell_m / 1000
    x, y, z = [(np.arange(n) + .5) * cell_km for n in truth.shape]
    mean_vp = truth.astype(np.float64).mean(axis=(0, 1))
    initial_profile = gaussian_filter1d(mean_vp, SMOOTHING_SIGMA_KM / cell_km, mode="reflect")
    initial = np.broadcast_to(initial_profile[None, None, :], truth.shape).astype(np.float32).copy()
    assert np.isfinite(initial).all() and np.all(initial > 0)
    np.testing.assert_array_equal(initial, np.broadcast_to(initial[0, 0, :], initial.shape))
    np.savez_compressed(OUTPUT / "initial_1d_previous.npz", velocity=initial,
                        cell_size_m=cell_m, origin_m=origin_m,
                        depth_centres_m=z * 1000, vp_profile_m_s=initial_profile)

    # Match the previous figure's exact-plane analytic sampling.
    truth_model.FAULT_THICKNESS_SCALE = 1.0
    truth_model.SURFACE_LAYER_THICKNESS_SCALE = 1.0
    truth_model.INTRUSION_CENTER_KM = (15.0, 3.8, 4.8)
    expected = truth_model.velocity(x[:, None, None], y[None, :, None], z[None, None, :]) * 1000
    np.testing.assert_allclose(truth, expected, rtol=1e-7)
    true_sections = [truth_model.velocity(x[None, :], 4.0, z[:, None]),
                     truth_model.velocity(10.0, y[None, :], z[:, None]),
                     truth_model.velocity(x[None, :], y[:, None], 4.0)]
    initial_sections = [np.broadcast_to(initial_profile[:, None] / 1000, (len(z), len(x))),
                        np.broadcast_to(initial_profile[:, None] / 1000, (len(z), len(y))),
                        np.full((len(y), len(x)), np.interp(4.0, z, initial_profile) / 1000)]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 12.2), layout="constrained")
    low = np.floor(min(truth.min(), initial.min()) / 200) * .2
    high = np.ceil(max(truth.max(), initial.max()) / 200) * .2
    norm = Normalize(low, high)
    for row, (true_section, start_section, extent, label, xlabel, ylabel) in enumerate(zip(
        true_sections, initial_sections,
        [(0, 20, 8, 0), (0, 8, 8, 0), (0, 20, 0, 8)],
        ["$Y=4$ km", "$X=10$ km", "$Z=4$ km"],
        ["$X$ (km)", "$Y$ (km)", "$X$ (km)"],
        ["Depth (km)", "Depth (km)", "$Y$ (km)"],
    )):
        for col, section in enumerate((true_section, start_section)):
            ax = axes[row, col]
            image = ax.imshow(section, extent=extent, origin="lower" if row == 2 else "upper",
                              interpolation="nearest", aspect="equal", cmap="viridis", norm=norm)
            ax.set(title=f"{'True model' if col == 0 else 'Smooth 1D initial model'} · {label}",
                   xlabel=xlabel, ylabel=ylabel)
            ax.set_xticks(np.arange(0, extent[1] + .01, 2 if row == 1 else 4))
            ax.set_yticks(np.arange(0, 8.01, 2))
    fig.suptitle("Restored synthetic truth and smooth 1D starting model\n"
                 "Horizontal mean $V_P(z)$ · Gaussian depth smoothing $\\sigma_z=0.4$ km",
                 fontsize=14)
    fig.colorbar(image, ax=axes, shrink=.85, pad=.025, label="$V_P$ (km/s)")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"realistic_anomalies_previous_with_initial_1d.{suffix}", dpi=220)
    plt.close(fig)
    report = {"truth": source.name, "truth_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "initial_model": "initial_1d_previous.npz", "method": "Arithmetic horizontal mean Vp, then Gaussian smoothing in depth",
              "smoothing_sigma_km": SMOOTHING_SIGMA_KM, "boundary_mode": "reflect",
              "velocity_range_km_s": [float(initial.min() / 1000), float(initial.max() / 1000)],
              "cell_size_m": cell_m, "shape_xyz": list(initial.shape), "velocity_units": "m/s",
              "sections_km": {"y": 4, "x": 10, "z": 4},
              "horizontal_section_initial_sampling": "Linear interpolation of smooth depth profile at exactly Z=4 km",
              "note": "Truth-derived synthetic starting model; not an independently estimated field-data prior"}
    (OUTPUT / "initial_1d_previous_description.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savetxt(OUTPUT / "initial_1d_previous_profile.csv", np.column_stack((z, mean_vp / 1000, initial_profile / 1000)),
               delimiter=",", header="depth_km,horizontal_mean_vp_km_s,initial_vp_km_s", comments="")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
