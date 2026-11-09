"""Compare restored truth, an 800-m layered 1D start, and an 800-m Vp L2 oracle."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import make_realistic_anomalies as truth_model

OUTPUT = Path(__file__).resolve().parents[1] / "figures/realistic_anomalies"
CELL_KM = 0.8


def main():
    with np.load(OUTPUT / "model_previous.npz", allow_pickle=False) as data:
        truth = data["velocity"].astype(np.float64)
        fine_cell_m = float(data["cell_size_m"])
        origin_m = data["origin_m"].copy()
    with np.load(OUTPUT / "initial_1d_previous.npz", allow_pickle=False) as data:
        profile = data["vp_profile_m_s"].copy()
        assert float(data["cell_size_m"]) == fine_cell_m
    factor = round(CELL_KM * 1000 / fine_cell_m)
    assert factor * fine_cell_m == CELL_KM * 1000
    assert all(n % factor == 0 for n in truth.shape)
    nx, ny, nz = [n // factor for n in truth.shape]
    oracle = truth.reshape(nx, factor, ny, factor, nz, factor).mean(axis=(1, 3, 5))
    layered_profile = profile.reshape(nz, factor).mean(axis=1)
    initial = np.broadcast_to(layered_profile[None, None, :], oracle.shape).copy()
    expanded_oracle = oracle
    expanded_initial = initial
    for axis in range(3):
        expanded_oracle = np.repeat(expanded_oracle, factor, axis=axis)
        expanded_initial = np.repeat(expanded_initial, factor, axis=axis)
    residual_mean = (truth - expanded_oracle).reshape(nx, factor, ny, factor, nz, factor).mean(axis=(1, 3, 5))
    np.testing.assert_allclose(residual_mean, 0, atol=1e-10)
    assert oracle.shape == initial.shape == (25, 10, 10)
    np.testing.assert_array_equal(initial, np.broadcast_to(initial[0, 0, :], initial.shape))
    rms = {"initial_1d": float(np.sqrt(np.mean((truth - expanded_initial) ** 2))),
           "oracle": float(np.sqrt(np.mean((truth - expanded_oracle) ** 2)))}
    assert rms["oracle"] <= rms["initial_1d"]
    for name, field in (("initial_1d_previous_800m", initial), ("oracle_previous_800m", oracle)):
        np.savez_compressed(OUTPUT / f"{name}.npz", velocity=field.astype(np.float32),
                            cell_size_m=np.array(800.0), origin_m=origin_m)

    truth_model.FAULT_THICKNESS_SCALE = 1.0
    truth_model.SURFACE_LAYER_THICKNESS_SCALE = 1.0
    truth_model.INTRUSION_CENTER_KM = (15.0, 3.8, 4.8)
    x, y, z = [(np.arange(n) + .5) * fine_cell_m / 1000 for n in truth.shape]
    true_sections = [truth_model.velocity(x[None, :], 4., z[:, None]),
                     truth_model.velocity(10., y[None, :], z[:, None]),
                     truth_model.velocity(x[None, :], y[:, None], 4.)]
    coarse_sections = []
    for field in (initial, oracle):
        coarse_sections.append([field[:, int(np.floor(4 / CELL_KM)), :].T / 1000,
                                field[int(np.floor(10 / CELL_KM)), :, :].T / 1000,
                                field[:, :, int(np.floor(4 / CELL_KM))].T / 1000])
    fields = [true_sections, *coarse_sections]
    low = np.floor(truth.min() / 200) * .2
    high = np.ceil(truth.max() / 200) * .2
    norm = Normalize(low, high)
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 3, figsize=(16.5, 11.5), layout="constrained")
    titles = ["True model", "1D initial · 800 m cells", "3D L2 oracle · 800 m cells"]
    for row, (extent, label, xlabel, ylabel) in enumerate(zip(
        [(0, 20, 8, 0), (0, 8, 8, 0), (0, 20, 0, 8)],
        ["$Y=4$ km", "$X=10$ km", "$Z=4$ km"],
        ["$X$ (km)", "$Y$ (km)", "$X$ (km)"], ["Depth (km)", "Depth (km)", "$Y$ (km)"],
    )):
        for col in range(3):
            ax = axes[row, col]
            image = ax.imshow(fields[col][row], extent=extent, origin="lower" if row == 2 else "upper",
                              aspect="equal", interpolation="nearest", cmap="viridis", norm=norm)
            ax.set(title=f"{titles[col]}\n{label}", xlabel=xlabel, ylabel=ylabel)
            ax.set_xticks(np.arange(0, extent[1] + .01, 2 if row == 1 else 4))
            ax.set_yticks(np.arange(0, 8.01, 2))
    fig.suptitle("Restored synthetic truth · 800-m piecewise-constant models\n"
                 f"Volume RMS: 1D initial {rms['initial_1d']:.1f} m/s · 3D oracle {rms['oracle']:.1f} m/s", fontsize=14)
    fig.colorbar(image, ax=axes, shrink=.85, pad=.02, label="$V_P$ (km/s)")
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"realistic_anomalies_previous_1d_oracle_800m.{suffix}", dpi=220)
    plt.close(fig)
    report = {"truth": "model_previous.npz", "cell_size_m": 800, "shape_xyz": list(oracle.shape),
              "velocity_units": "m/s", "domain_xyz_km": [20, 8, 8],
              "initial_method": "Average the previous smooth 1D Vp profile over each 800-m depth layer; constant within cells",
              "oracle_method": "Arithmetic mean of truth Vp in each 800-m cube; full-volume Vp L2 optimum, not travel-time optimum",
              "volume_rms_m_s": rms, "color_limits_km_s": [low, high],
              "sections_km": {"y": 4, "x": 10, "z": 4},
              "boundary_convention": "Half-open cells: at Y=4 or Z=4 km use cell [4, 4.8) km; X=10 uses [9.6, 10.4) km",
              "rendering": "Native constant cells without smoothing; truth evaluated analytically at exact planes"}
    (OUTPUT / "realistic_anomalies_previous_1d_oracle_800m.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
