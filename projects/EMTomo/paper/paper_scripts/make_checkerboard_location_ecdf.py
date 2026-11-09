"""Full-catalogue 3D location-error ECDFs for nx4, nx6, and nx8."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from make_checkerboard_comparison import ROOT, STUDY, EM_RUNS, sha
from make_checkerboard_locations import read_points

OUTPUT = Path(__file__).resolve().parents[1] / "figures"
STEM = "checkerboard_location_error_ecdf"
COLORS = {4: "#0072b2", 6: "#d55e00", 8: "#61177c"}


def ecdf(errors, right_edge):
    """Right-continuous ECDF, including exact ties and the full upper tail."""
    values, counts = np.unique(errors, return_counts=True)
    fractions = np.cumsum(counts) / len(errors)
    assert np.all(np.diff(values) > 0) and fractions[-1] == 1
    return np.r_[0., values, right_edge], np.r_[0., fractions, 1.]


def geographic_xyz(events, mapping):
    lat, lon = mapping["center_lat"], mapping["center_lon"]
    return np.column_stack(((events[:, 2] - lon) * 111 * np.cos(np.deg2rad(lat)) + 12,
                            (events[:, 1] - lat) * 111 + 6, events[:, 3]))


def main():
    series, records, hashes, csv_rows = {}, [], {}, []
    common_truth = common_ids = None
    for nx, run_id in EM_RUNS.items():
        em_dir = ROOT / "runs" / run_id
        dd_dir = STUDY / "outputs" / f"nx{nx}-tomodd"
        meta = json.loads((em_dir / "meta.json").read_text())
        config = json.loads((dd_dir / "config.json").read_text())
        eid = meta["source_experiment"]["id"]
        assert config["source_experiment"] == eid
        assert meta["run_params"]["coverage_damping_power"] == 0
        assert meta["run_params"]["candidate_mode"] == "hard"
        assert meta["run_params"]["weights_top_n"] == 1
        assert meta["run_params"]["n_cycles"] == 12
        assert config["solver"]["iteration_sets"][0][0] == 3
        source = STUDY / "snapshot/projects/forward_modeling/experiments/input" / eid
        for name, expected in config["source_forward_metadata"]["input_sha256"].items():
            assert sha(source / name) == expected, source / name
        assert sha(source / "model.npz") == meta["source_experiment"]["model_sha256"]
        rows, truth = read_points(source / "events.csv")
        ids = [r["event_id"] for r in rows]
        assert len(rows) == len(set(ids)) == 1200
        if common_truth is None:
            common_truth, common_ids = truth.copy(), ids
        else:
            assert np.array_equal(truth, common_truth) and ids == common_ids
        completed = sorted(int(p.parent.name.split("_")[1]) for p in em_dir.glob("iter_*/complete.json"))
        assert completed == list(range(12))
        latest = completed[-1]
        cycle = em_dir / f"iter_{latest}"
        assert {p.name for p in cycle.glob("event_*") if p.is_dir()} == {f"event_{i}" for i in range(len(rows))}
        fine_cell_km = meta["grid_info"]["fine_cell_size"] / 1000
        fine_shape = np.array(meta["grid_info"]["fine_shape"])
        locations = []
        weight_digest = hashlib.sha256()
        for i, event_id in enumerate(ids):
            path = cycle / f"event_{i}/weights.npz"
            weight_digest.update(f"{i}\t{event_id}\t{sha(path)}\n".encode())
            with np.load(path, allow_pickle=False) as weights:
                assert weights["positions"].shape == (1, 3)
                assert np.array_equal(weights["weight_shape"], fine_shape)
                assert weights["weight_values"].shape == (1,) and np.isclose(weights["weight_values"][0], 1)
                locations.append((weights["positions"][0] + .5) * fine_cell_km)
        em_xyz = np.array(locations)
        assert np.isfinite(em_xyz).all() and np.all(em_xyz >= 0) and np.all(em_xyz <= [24, 12, 12])
        with np.load(dd_dir / "state.npz", allow_pickle=False) as state:
            assert np.allclose(state["true_event_xyz_km"], truth, atol=1e-10, rtol=0)
            true_ids = state["true_events"][:, 0]
            assert np.array_equal(true_ids, np.arange(1, len(rows) + 1))
            mapping = config["coordinate_mapping"]
            assert np.allclose(geographic_xyz(state["true_events"], mapping), truth, atol=1e-9, rtol=0)
            final_ids = state["final_events"][:, 0]
            assert len(final_ids) == len(set(final_ids)) == len(rows)
            assert set(final_ids) == set(true_ids)
            by_id = {event_id: i for i, event_id in enumerate(final_ids)}
            aligned = state["final_events"][[by_id[event_id] for event_id in true_ids]]
            dd_xyz = geographic_xyz(aligned, mapping)
        assert np.isfinite(dd_xyz).all()
        errors = {"em": np.linalg.norm(em_xyz - truth, axis=1) * 1000,
                  "dd": np.linalg.norm(dd_xyz - truth, axis=1) * 1000}
        for method, values in errors.items():
            assert len(values) == 1200 and np.isfinite(values).all() and np.all(values >= 0)
            series[nx, method] = values
        summaries = {}
        for method, values in errors.items():
            summaries[method] = {"n": len(values), "rms_m": float(np.sqrt(np.mean(values**2))),
                                 "median_m": float(np.median(values)),
                                 "p90_m": float(np.quantile(values, .90)),
                                 "p95_m": float(np.quantile(values, .95)),
                                 "max_m": float(values.max()),
                                 "fraction_within_250m": float(np.mean(values <= 250)),
                                 "fraction_within_500m": float(np.mean(values <= 500))}
        records.append({"nx": nx, "em_run": str(em_dir), "em_location_iteration": latest,
                        "em_weights_row_id_sha256": weight_digest.hexdigest(),
                        "dd_run": str(dd_dir), "dd_iterations": 3, "statistics": summaries})
        csv_rows.extend((nx, event_id, float(errors["em"][i]), float(errors["dd"][i]))
                        for i, event_id in enumerate(ids))
        for path in (em_dir / "meta.json", cycle / "complete.json", source / "events.csv",
                     source / "stations.csv", source / "model.npz", dd_dir / "config.json", dd_dir / "state.npz"):
            hashes[str(path)] = sha(path)

    right_edge = np.ceil(max(a.max() for a in series.values()) / 250) * 250
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10,
        "axes.labelsize": 11, "axes.linewidth": .6,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "text.color": "#262b30", "axes.labelcolor": "#262b30",
        "xtick.color": "#262b30", "ytick.color": "#262b30", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    for nx, color in COLORS.items():
        for method, style in (("em", "-"), ("dd", (0, (4, 2)))):
            x, y = ecdf(series[nx, method], right_edge)
            ax.step(x, y, where="post", color=color, linestyle=style, linewidth=1.35)
    ax.set(xlabel="3D location error (m)", ylabel="Cumulative fraction of events",
           xlim=(0, right_edge), ylim=(0, 1.005))
    ax.xaxis.set_major_locator(MultipleLocator(500 if right_edge <= 3000 else 1000))
    ax.yaxis.set_major_locator(MultipleLocator(.2))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(direction="out", pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dddddd", linewidth=.45)
    handles = [Line2D([], [], color=COLORS[nx], lw=1.35,
                      linestyle="-" if method == "EMTomo" else (0, (4, 2)),
                      label=f"{method}, $n_x={nx}$")
               for method in ("EMTomo", "tomoDD") for nx in COLORS]
    ax.legend(handles=handles, ncol=2, frameon=False, loc="lower right", fontsize=9,
              handlelength=2.6, columnspacing=1.6, labelspacing=.6, borderaxespad=.9)
    for suffix in ("png", "pdf", "svg"):
        path = OUTPUT / f"{STEM}.{suffix}"
        fig.savefig(path, dpi=300)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    csv_path = OUTPUT / f"{STEM}.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("nx", "event_id", "emtomo_error_3d_m", "tomodd_error_3d_m"))
        writer.writerows(csv_rows)
    hashes[str(csv_path)] = sha(csv_path)
    provenance = {
        "error": "Euclidean 3D distance in metres; all 1200 events per run, equal weight, no subset or tail clipping",
        "ecdf": "Right-continuous step function F(x)=count(error <= x)/1200; ties grouped exactly",
        "x_range_m": [0, float(right_edge)], "layout": "Colour encodes nx; solid EMTomo; dashed tomoDD",
        "em_checkpoint_caveat": "iter_11 saved E-step locations precede the final (12th) M-step; no new relocation",
        "em_coverage_damping_power": 0, "quantiles": "NumPy linear interpolation (descriptive summaries)",
        "rows": records, "sha256": hashes,
    }
    (OUTPUT / f"{STEM}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"x_range_m": provenance["x_range_m"], "rows": records}, indent=2))


if __name__ == "__main__":
    main()
