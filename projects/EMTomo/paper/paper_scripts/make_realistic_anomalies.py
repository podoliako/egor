"""Generate an illustrative upper-crustal Vp truth and its exact central sections.

Synthetic geology, not a calibrated or observed model. Coordinates are XYZ,
with depth positive down. Run with the repository's venv Python.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "figures/realistic_anomalies"
CELL_KM = 0.1
SEED = 42
SIZE_SCALE = 2.0
AMPLITUDE_SCALE = 1.5
FAULT_THICKNESS_SCALE = 3.0
SURFACE_LAYER_THICKNESS_SCALE = 2.0
INTRUSION_CENTER_KM = (5.0, 3.8, 2.5)
OUTPUT_SUFFIX = ""


def scaled_coordinates(x, y, z, center):
    """Expand a feature about its fixed centre, without changing its orientation."""
    return tuple(c + (q - c) / SIZE_SCALE for q, c in zip((x, y, z), center))


def components(x, y, z):
    """Analytic velocity components in km/s; input coordinates in km."""
    # Compaction/weathering gradient approaches crystalline upper-crustal Vp.
    background = 3.35 + 1.75 * (1 - np.exp(-z / 1.15)) + 0.115 * z
    original_coordinates = (x, y, z)
    # Additional vertical dilation thickens the surface layer without widening it.
    x, y, z = scaled_coordinates(*original_coordinates, (7.0, 4.4, 0.0))
    z = z / SURFACE_LAYER_THICKNESS_SCALE
    # A buried asymmetric basin and a basement high displace the same contact.
    basin_shape = np.exp(-0.5 * (((x - 7.0) / 3.4) ** 2 + ((y - 4.4) / 2.4) ** 2))
    hx, hy, _ = scaled_coordinates(*original_coordinates, (14.8, 3.0, 0.0))
    high_shape = np.exp(-0.5 * (((hx - 14.8) / 2.8) ** 2 + ((hy - 3.0) / 2.2) ** 2))
    contact = 0.55 + 2.1 * basin_shape - 0.35 * high_shape + 0.12 * np.sin(x / 2.2 + y / 2)
    sediment_fraction = 0.5 * (1 - np.tanh((z - contact) / 0.22))
    basin = -0.8 * sediment_fraction * (1 - np.exp(-z / 0.35))
    # Oblique, curved fault corridor, dipping ~65 degrees; tapered at depth.
    x, y, z = scaled_coordinates(*original_coordinates, (9.0, 4.0, 0.0))
    fault_x = 9.0 + 0.44 * z + 0.42 * (y - 4) + 0.22 * np.sin(y * 0.9)
    width = FAULT_THICKNESS_SCALE * (0.25 + 0.04 * z)
    fault = -0.48 * np.exp(-0.5 * ((x - fault_x) / width) ** 2)
    fault = fault * (0.65 + 0.35 * np.exp(-0.5 * ((y - 4.5) / 2.5) ** 2))
    fault = fault * (1 - np.exp(-z / 0.5)) * np.exp(-(z / 7.4) ** 6)
    # Tilted, non-spherical mafic intrusive body in crystalline basement.
    x, y, z = scaled_coordinates(*original_coordinates, INTRUSION_CENTER_KM)
    cx, cy, cz = INTRUSION_CENTER_KM
    u = (x - cx) + 0.45 * (z - cz)
    radius = (u / 2.5) ** 2 + ((y - cy) / 1.8) ** 2 + ((z - cz) / 2.0) ** 2
    intrusion = 0.48 * np.exp(-0.5 * radius ** 2)
    # Altered/fluid-bearing lens next to the fault, elongated along strike.
    x, y, z = scaled_coordinates(*original_coordinates, (10.216, 4.8, 3.8))
    lens_x = x - (10.0 + 0.32 * (y - 4.0) + 0.2 * (z - 4.0))
    lens_radius = (lens_x / 1.35) ** 2 + ((y - 4.8) / 2.0) ** 2 + ((z - 3.8) / 0.85) ** 2
    alteration = -0.36 * np.exp(-0.5 * lens_radius)
    # Fixed-seed band-limited fabric: longer horizontal than vertical scales.
    x, y, z = scaled_coordinates(*original_coordinates, (10.0, 4.0, 0.0))
    texture = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z)))
    rng = np.random.default_rng(SEED)
    for _ in range(18):
        kx, ky = rng.uniform(-1.8, 1.8, 2)
        kz = rng.uniform(1.5, 5.5)
        phase = rng.uniform(0, 2 * np.pi)
        texture += np.sin(kx * x + ky * y + kz * z + phase)
    texture *= 0.035 / np.sqrt(18 / 2)
    texture *= 1 - np.exp(-z / 0.5)
    return {"background": np.broadcast_to(background, texture.shape),
            "basin": AMPLITUDE_SCALE * basin, "fault": AMPLITUDE_SCALE * fault,
            "intrusion": AMPLITUDE_SCALE * intrusion, "alteration": AMPLITUDE_SCALE * alteration,
            "texture": AMPLITUDE_SCALE * texture}


def velocity(x, y, z):
    return sum(components(x, y, z).values())


def main():
    x, y, z = [(np.arange(round(length / CELL_KM)) + 0.5) * CELL_KM for length in (20, 8, 8)]
    vp = velocity(x[:, None, None], y[None, :, None], z[None, None, :])
    assert vp.shape == (200, 80, 80) and np.isfinite(vp).all()
    assert 2.0 < vp.min() < vp.max() < 7.5
    np.savez_compressed(OUTPUT / f"model{OUTPUT_SUFFIX}.npz", velocity=(vp * 1000).astype(np.float32),
                        cell_size_m=np.array(100.0), origin_m=np.zeros(3))

    # Evaluate the analytic field exactly at the requested planes, not at a nearby voxel centre.
    sections = [velocity(x[None, :], 4.0, z[:, None]),
                velocity(10.0, y[None, :], z[:, None]),
                velocity(x[None, :], y[:, None], 4.0)]
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(13.5, 8.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(20, 8))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    note = fig.add_subplot(grid[1, 1])
    note.axis("off")
    norm = Normalize(np.floor(vp.min() * 5) / 5, np.ceil(vp.max() * 5) / 5)
    assert vp.min() >= norm.vmin and vp.max() <= norm.vmax
    for ax, field, extent, title, xlabel, ylabel, horizontal in zip(
        axes, sections, [(0, 20, 8, 0), (0, 8, 8, 0), (0, 20, 0, 8)],
        ["(a) Vertical section · $Y=4$ km", "(b) Vertical section · $X=10$ km", "(c) Horizontal section · $Z=4$ km"],
        ["$X$ (km)", "$Y$ (km)", "$X$ (km)"], ["Depth (km)", "Depth (km)", "$Y$ (km)"],
        [False, False, True],
    ):
        image = ax.imshow(field, origin="lower" if horizontal else "upper", extent=extent,
                          cmap="viridis", norm=norm, aspect="equal", interpolation="nearest")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.set_xticks(np.arange(0, extent[1] + 0.01, 2 if extent[1] == 8 else 4))
        ax.set_yticks(np.arange(0, 8.01, 2))
    # Dashed lines show the intersections of the three central planes.
    axes[0].axvline(10, color="white", lw=0.8, ls="--", alpha=.75)
    axes[0].axhline(4, color="white", lw=0.8, ls="--", alpha=.75)
    axes[1].axvline(4, color="white", lw=0.8, ls="--", alpha=.75)
    axes[1].axhline(4, color="white", lw=0.8, ls="--", alpha=.75)
    axes[2].axvline(10, color="white", lw=0.8, ls="--", alpha=.75)
    axes[2].axhline(4, color="white", lw=0.8, ls="--", alpha=.75)
    note.text(.03, .94, "Synthetic upper-crustal truth\n\n"
              "• Compaction / depth gradient\n• Sedimentary basin\n• Dipping damaged fault\n• Low-Vp alteration lens\n• High-Vp intrusive body\n• Weak correlated fabric\n\n"
              f"Dashed: central-plane intersections\nFault thickness ×{FAULT_THICKNESS_SCALE:g} · surface layer ×{SURFACE_LAYER_THICKNESS_SCALE:g}\n"
              f"High-Vp centre: X={INTRUSION_CENTER_KM[0]:g}, Z={INTRUSION_CENTER_KM[2]:g} km\n"
              "Grid: 100 m · depth positive down\nIllustrative geology, not field data",
              va="top", fontsize=11, linespacing=1.5, transform=note.transAxes)
    fig.colorbar(image, ax=axes + [note], shrink=.9, pad=.025, label="$V_P$ (km/s)")
    fig.suptitle("Realistic-style synthetic velocity model · $20 × 8 × 8$ km", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"realistic_anomalies_sections{OUTPUT_SUFFIX}.{suffix}", dpi=220)
    plt.close(fig)
    report = {"interpretation": "Illustrative synthetic upper crust, not an observed or calibrated site model",
              "domain_xyz_km": [20, 8, 8], "depth_positive_down": True,
              "grid_shape_xyz": list(vp.shape), "cell_size_m": 100, "origin_m": [0, 0, 0],
              "velocity_units_in_npz": "m/s", "velocity_range_km_s": [float(vp.min()), float(vp.max())],
              "sections_km": {"vertical_y": 4, "vertical_x": 10, "horizontal_z": 4},
              "section_sampling": "Analytic field at exact central planes, 100-m in-plane pixel centres",
              "seed": SEED, "color_limits_km_s": [norm.vmin, norm.vmax],
              "linear_size_scale": SIZE_SCALE, "anomaly_amplitude_scale": AMPLITUDE_SCALE,
              "background_unchanged": True,
              "fault_thickness_scale_relative_to_previous": FAULT_THICKNESS_SCALE,
              "surface_layer_thickness_scale_relative_to_previous": SURFACE_LAYER_THICKNESS_SCALE,
              "intrusion_center_xyz_km": list(INTRUSION_CENTER_KM),
              "scaling_note": "Fixed-centre spatial dilation; basin/fault/fabric anchored at surface; features clipped by unchanged domain",
              "intrusion_semiaxes_km": [5.0, 3.6, 4.0],
              "alteration_scales_km": [2.7, 4.0, 1.7],
              "features": {"basin": f"Undulating sediment/basement contact; deepest near (7, 4.4) km; additional thickness factor {SURFACE_LAYER_THICKNESS_SCALE:g}",
                           "fault": f"Oblique curved corridor dipping about 65 degrees; additional thickness factor {FAULT_THICKNESS_SCALE:g}; up to -0.72 km/s",
                           "alteration": "Tilted low-Vp lens centred at (10.216, 4.8, 3.8) km; -0.54 km/s peak",
                           "intrusion": f"Tilted mafic body centred at {INTRUSION_CENTER_KM} km; unchanged size and +0.72 km/s peak",
                           "fabric": "18 deterministic sinusoidal modes; wavelengths doubled; nominal RMS 0.0525 km/s"}}
    (OUTPUT / f"model_description{OUTPUT_SUFFIX}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", action="store_true",
                        help="Rebuild the version before fault/layer thickening and intrusion relocation, with a _previous suffix")
    args = parser.parse_args()
    if args.previous:
        FAULT_THICKNESS_SCALE = 1.0
        SURFACE_LAYER_THICKNESS_SCALE = 1.0
        INTRUSION_CENTER_KM = (15.0, 3.8, 4.8)
        OUTPUT_SUFFIX = "_previous"
    main()
