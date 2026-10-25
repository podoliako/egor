"""Compare native EM cells and tomoDD fields at Y=10 km and Z=5 km."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from make_checkerboard_geometry import map_frame

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/checkerboard_scaled_20261021"
OUTPUT = Path(__file__).resolve().parent / "figures"
STEM = "checkerboard_nx4_nx6_nx8_comparison"
EM_RUNS = {
    4: "run_cbscale21-nx4-hard-k15-cov0-fast_v1.4_cov-0_ncand-15_20261022_092857_a566dffe817b4c7e89a91063cfbcde17",
    6: "run_cbscale21-nx6-hard-k15-cov0-fast_v1.4_cov-0_ncand-15_20261022_084247_c78101a2744e4a4e9e15ee90c4a992d6",
    8: "run_cbscale21-nx8-hard-k15-cov0-fast_v1.4_cov-0_ncand-15_20261022_093634_ed4066420c1f463a943d7a73df856217",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Sample at pixel centres; keep the same exact Y coordinate for every field.
    spacing, section_y, section_z = .025, 10.0, 5.0
    x = (np.arange(round(24 / spacing)) + .5) * spacing
    z = (np.arange(round(12 / spacing)) + .5) * spacing
    zz, xx = np.meshgrid(z, x, indexing="ij")
    yy = zz.copy()
    rows, horizontal_rows, records, hashes = [], [], [], {}
    for nx, run_id in EM_RUNS.items():
        em_dir = ROOT / "runs" / run_id
        dd_dir = STUDY / "outputs" / f"nx{nx}-tomodd"
        meta = json.loads((em_dir / "meta.json").read_text())
        config = json.loads((dd_dir / "config.json").read_text())
        eid = meta["source_experiment"]["id"]
        assert config["source_experiment"] == eid
        assert meta["run_params"]["coverage_damping_power"] == 0
        assert meta["run_params"]["n_cycles"] == 12
        assert config["solver"]["iteration_sets"][0][0] == 3
        assert len(list(em_dir.glob("iter_*/complete.json"))) == 12
        source = STUDY / "snapshot/projects/forward_modeling/experiments/input" / eid
        for name, digest in config["source_forward_metadata"]["input_sha256"].items():
            assert sha(source / name) == digest, source / name
        assert sha(source / "model.npz") == meta["source_experiment"]["model_sha256"]
        with np.load(source / "model.npz") as model:
            truth = model["velocity"] / 1000
            cell = float(model["cell_size_m"]) / 1000
        initial_em = np.load(em_dir / "initial_model.npy", allow_pickle=False) / 1000
        assert np.all(initial_em == 5.0), "Shared initial panel requires homogeneous 5 km/s EM starts"
        em = np.load(em_dir / "final_model.npy", allow_pickle=False) / 1000
        assert em.shape == (24, 12, 12) and np.isfinite(em).all()
        truth_slice = truth[np.floor(xx / cell).astype(int), int(section_y / cell), np.floor(zz / cell).astype(int)]
        em_slice = em[np.floor(xx).astype(int), int(section_y), np.floor(zz).astype(int)]
        truth_horizontal = truth[np.floor(xx / cell).astype(int), np.floor(yy / cell).astype(int), int(section_z / cell)]
        em_horizontal = em[np.floor(xx).astype(int), np.floor(yy).astype(int), int(section_z)]
        grid = config["tomodd_grid"]
        # Stored DD X/Y coordinates are centred at the domain midpoint.
        axes = (np.array(grid["depths_km"]), np.array(grid["y_km"]) + 6,
                np.array(grid["x_km"]) + 12)
        points = np.stack((zz, np.full_like(xx, section_y), xx), axis=-1)
        horizontal_points = np.stack((np.full_like(xx, section_z), yy, xx), axis=-1)
        with np.load(dd_dir / "state.npz", allow_pickle=False) as state:
            assert np.array_equal(state["physical_velocity_km_s"], truth)
            assert np.allclose(state["depths"], axes[0])
            assert np.all(state["initial_vp"] == 5.0), "DD start differs from shared EM start"
            dd_slice = RegularGridInterpolator(axes, state["final_vp"])(points)
            oracle_slice = RegularGridInterpolator(axes, state["oracle_vp"])(points)
            dd_horizontal = RegularGridInterpolator(axes, state["final_vp"])(horizontal_points)
            oracle_horizontal = RegularGridInterpolator(axes, state["oracle_vp"])(horizontal_points)
        fields = (truth_slice, em_slice, dd_slice, oracle_slice)
        assert all(a.shape == zz.shape and np.isfinite(a).all() for a in fields)
        rows.append(fields)
        horizontal_fields = (truth_horizontal, em_horizontal, dd_horizontal, oracle_horizontal)
        assert all(a.shape == yy.shape and np.isfinite(a).all() for a in horizontal_fields)
        horizontal_rows.append(horizontal_fields)
        records.append({"nx": nx, "em_run": str(em_dir), "em_updates": 12,
                        "coverage_damping_power": 0, "dd_run": str(dd_dir), "dd_iterations": 3,
                        "horizontal_slice_ranges_km_s": {name: [float(a.min()), float(a.max())]
                                                          for name, a in zip(("truth", "em", "dd", "dd_oracle"), horizontal_fields)},
                        "slice_ranges_km_s": {name: [float(a.min()), float(a.max())]
                                               for name, a in zip(("truth", "em", "dd", "dd_oracle"), fields)}})
        for path in (source / "model.npz", em_dir / "meta.json", em_dir / "final_model.npy", em_dir / "initial_model.npy",
                     dd_dir / "config.json", dd_dir / "state.npz"):
            hashes[str(path)] = sha(path)

    rows.extend(horizontal_rows)
    colors = ["#7a7a7a", "#ffffff", "#e32400", "#61177c"]
    low, high = 4.0, 6.0
    assert all(low <= float(a.min()) <= float(a.max()) <= high for row in rows for a in row), "Velocity scale would clip displayed data"
    anchors = [low, 4.75, 5.25, high]
    cmap = LinearSegmentedColormap.from_list("checkerboard_velocity", list(zip(
        [(v - low) / (high - low) for v in anchors], colors)), N=1024)
    norm = Normalize(low, high)
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.linewidth": .6,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30",
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(13.8, 12.8))
    grid = fig.add_gridspec(7, 4, height_ratios=(1, 1, 1, .8, 1, 1, 1),
                           left=.072, right=.97, bottom=.05, top=.93, wspace=.09, hspace=.23)
    axes = np.array([[fig.add_subplot(grid[r if r < 3 else r + 1, c])
                      for c in range(4)] for r in range(6)])
    for r, (nx, fields) in enumerate(zip(list(EM_RUNS) * 2, rows)):
        for c, field in enumerate(fields):
            ax = axes[r, c]
            horizontal = r >= 3
            image = ax.imshow(field, extent=(0, 24, 0, 12) if horizontal else (0, 24, 12, 0),
                              origin="lower" if horizontal else "upper", aspect="equal",
                              cmap=cmap, norm=norm, interpolation="nearest", rasterized=True)
            ax.set(xlim=(0, 24), ylim=(0, 12) if horizontal else (12, 0),
                   xticks=np.arange(0, 25, 6), yticks=[0, 4, 8, 12])
            ax.tick_params(labelbottom=r in (2, 5), labelleft=c == 0)
            ax.tick_params(direction="out", pad=3)
            map_frame(ax)
            if r == 0:
                ax.set_title(("True model", "EMTomo", "tomoDD", "tomoDD oracle")[c], pad=10)
            if r in (2, 5):
                ax.set_xlabel("$X$ (km)")
            if c == 0:
                ax.set_ylabel("$Y$ (km)" if horizontal else "Depth (km)")
                ax.text(-.23, .5, f"$n_x = {nx}$", transform=ax.transAxes,
                        rotation=90, ha="center", va="center", fontsize=11)
    fig.canvas.draw()
    fig.text(.521, .978, "Vertical sections · $Y = 10$ km", ha="center", va="top", fontsize=12)
    separator = grid[3, :].get_position(fig)
    fig.text(.521, separator.y0 + .003,
             "Horizontal sections · depth $Z = 5$ km", ha="center", va="center", fontsize=12)
    color_ax = fig.add_axes([.271, separator.y1 - .027, .50, .012])
    colorbar = fig.colorbar(image, cax=color_ax, orientation="horizontal",
                           ticks=np.arange(low, high + .01, .5), format="%.1f")
    color_ax.set_title("$V$ (km/s)", fontsize=9, pad=6)
    colorbar.outline.set_linewidth(.45)
    color_ax.tick_params(labelsize=8, width=.45, length=2)
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    provenance = {"section_y_km": section_y, "section_depth_km": section_z,
                  "row_order": "vertical nx4/nx6/nx8, then horizontal nx4/nx6/nx8",
                  "sampling_spacing_km": spacing,
                  "shared_initial_model": {"velocity_km_s": 5.0,
                                           "verified": "All three EM initial_model.npy and DD initial_vp arrays are identically 5 km/s"},
                  "representation": {"truth": "piecewise constant original model cells",
                                     "em": "piecewise constant 1 km cells, final model after 12 updates",
                                     "dd": "native trilinear Vp interpolation, final model after 3 iterations",
                                     "dd_oracle": "saved oracle_vp with native trilinear interpolation; not an inversion"},
                  "color_anchors_km_s": dict(zip(colors, anchors)),
                  "colorbar": {"orientation": "horizontal", "tick_spacing_km_s": .5},
                  "normalization": "One shared linear velocity scale; nonlinear placement of colormap color anchors; no data clipping.",
                  "rows": records, "sha256": hashes}
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"color_anchors": dict(zip(colors, anchors)), "rows": records}, indent=2))


if __name__ == "__main__":
    main()
