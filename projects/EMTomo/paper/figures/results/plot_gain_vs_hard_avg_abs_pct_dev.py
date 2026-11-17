"""Export viewer velocity-deviation minima, then plot their truth-based diagnostic gain."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from plot_soft_gain_distribution import SERIES, SOURCE

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
RUNS = PAPER.parent / "runs"
CSV_PATH = PAPER / "exp_data/hard_soft_min_avg_abs_pct_dev.csv"
METRIC = "avg_abs_pct_dev"


def minimum_quality(run_id):
    directory = RUNS / run_id
    path = directory / "quality.jsonl"
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indices = [point["iter"] for point in values]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate quality iterations: {run_id}")
    if not values or any(not np.isfinite(p[METRIC]) or p[METRIC] < 0 for p in values):
        raise ValueError(f"Invalid quality metric: {run_id}")
    if any(not (directory / f"iter_{index}/complete.json").exists() for index in indices):
        raise ValueError(f"Unpublished quality iteration: {run_id}")
    best = min(values, key=lambda p: (p[METRIC], p["iter"]))
    return best, len(values), hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with SOURCE.open() as stream:
        source_rows = list(csv.DictReader(stream))
    rows = []
    quality_hashes = {}
    keep = ("pair_id", "section", "section_name", "experiment_id", "pair_status",
            "hard_control_reuse_count", "n_events", "n_stations", "n_candidates",
            "weights_top_n", "weights_min_distance", "soft_temperature", "hard_temperature",
            "soft_temperature_schedule", "hard_temperature_schedule", "lambda_reg",
            "coverage_damping_power", "assumed_relative_sigma", "assumed_absolute_sigma_s",
            "assumed_model_sigma_s", "initial_kind", "requested_updates", "run_version",
            "control_fraction", "study_manifest")
    for source in source_rows:
        row = {key: source[key] for key in keep}
        for mode in ("hard", "soft"):
            run_id = source[f"{mode}_run_id"]
            best, count, checksum = minimum_quality(run_id)
            quality_hashes[run_id] = checksum
            row.update({
                f"{mode}_run_id": run_id,
                f"{mode}_min_avg_abs_pct_dev": float(best[METRIC]),
                f"{mode}_best_iteration_index": best["iter"],
                f"{mode}_best_update_number": best["iter"] + 1,
                f"{mode}_n_quality_iterations": count,
                f"{mode}_has_final_model": source[f"{mode}_has_final_model"],
            })
        h, s = row["hard_min_avg_abs_pct_dev"], row["soft_min_avg_abs_pct_dev"]
        if h <= 0:
            raise ValueError(f"Cannot compute relative gain with nonpositive hard error: {row['pair_id']}")
        row["soft_gain_percent"] = 100 * (h - s) / h
        row["soft_minus_hard_error_percentage_points"] = s - h
        row["metric"] = METRIC
        row["metric_units"] = "percent"
        row["comparison_basis"] = "independent_truth_based_minimum_over_complete_post_update_quality"
        rows.append(row)
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate comparison rows")
    # Publish the new CSV before generating the figure. The residual CSV is untouched.
    temporary = CSV_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(CSV_PATH)
    hard = np.asarray([r["hard_min_avg_abs_pct_dev"] for r in rows])
    gain = np.asarray([r["soft_gain_percent"] for r in rows])
    noise_levels = np.full(len(rows), np.nan)
    noise_metadata_sha256 = {}
    for index, row in enumerate(rows):
        if row["section"] != "4":
            continue
        path = PAPER.parents[1] / "forward_modeling/experiments/output" / row["experiment_id"] / "metadata.json"
        metadata = json.loads(path.read_text())
        actual = metadata["noise"]["config"]
        relative_multiplier = actual["relative_sigma"] / .01
        absolute_multiplier = actual["absolute_sigma_s"] / .05
        if not np.isclose(relative_multiplier, absolute_multiplier):
            raise ValueError(f"Noise components have different multipliers: {row['pair_id']}")
        noise_levels[index] = relative_multiplier
        noise_metadata_sha256[row["experiment_id"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    noise_cmap = LinearSegmentedColormap.from_list(
        "actual_pick_noise", plt.get_cmap("Greens")(np.linspace(.38, .92, 256)),
    )
    noise_norm = Normalize(vmin=.5, vmax=2.)
    wins = int(np.count_nonzero(gain > 0))
    losses = int(np.count_nonzero(gain < 0))
    x_max = float(np.ceil(hard.max() * 1.07 / 2) * 2)
    y_min = float(min(-2, np.floor(gain.min() / 2) * 2))
    y_max = float(max(2, np.ceil(gain.max() / 2) * 2))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.8, 5.7))
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1.2, zorder=2)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([row["section"] == section for row in rows])
            if section == "4":
                noise_scatter = ax.scatter(
                    hard[mask], gain[mask], s=58, c=noise_levels[mask], cmap=noise_cmap,
                    norm=noise_norm, alpha=1., edgecolors="white", linewidths=.65, zorder=3,
                    label=f"{label} (n={int(mask.sum())})",
                )
            else:
                ax.scatter(hard[mask], gain[mask], s=58, c=color, alpha=.87,
                           edgecolors="white", linewidths=.65, zorder=3,
                           label=f"{label} (n={int(mask.sum())})")
        colorbar_axes = ax.inset_axes([.55, .50, .38, .027])
        colorbar = fig.colorbar(noise_scatter, cax=colorbar_axes, orientation="horizontal")
        colorbar.set_ticks([.5, .75, 1., 1.5, 2.], labels=["0.5×", "0.75×", "1×", "1.5×", "2×"])
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.set_title("Actual pick-noise multiplier (darker = noisier)", fontsize=9, pad=8)
        colorbar.outline.set_linewidth(.5)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)
        ax.grid(color="#dddddd", linewidth=.6)
        ax.set_xlabel("Hard minimum average absolute velocity deviation (%)", labelpad=10)
        ax.set_ylabel("Soft gain in minimum velocity deviation (%)", labelpad=10)
        ax.set_title("Soft gain versus hard velocity-model error", loc="left", pad=31)
        ax.text(0, 1.025, r"Viewer metric: avg_abs_pct_dev; gain = $100\,(E_{hard,min}-E_{soft,min})/E_{hard,min}$",
                transform=ax.transAxes, fontsize=10, color="#444444")
        ax.text(.98, .96, "Soft better", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="#555555")
        ax.legend(loc="lower right", frameon=False, fontsize=9.2, labelspacing=.75)
        fig.text(.53, .092,
                 f"N = {len(rows)} comparisons  |  Soft better: {wins}  |  Hard better: {losses}"
                 "  |  Dashed line: no difference",
                 ha="center", fontsize=9.5, color="#444444")
        fig.text(.53, .035,
                 "Truth-based diagnostic: each run's iteration is selected using the true velocity model.\n"
                 "Independent per-run minima; no clipping or jitter. Some hard controls are reused.",
                 ha="center", fontsize=9, color="#555555", linespacing=1.6)
        fig.subplots_adjust(left=.12, right=.98, top=.81, bottom=.24)
        for stem in ("soft_gain_vs_hard_avg_abs_pct_dev", "soft_gain_vs_hard_avg_abs_pct_dev_noise_gradient"):
            for extension in ("png", "pdf"):
                fig.savefig(HERE / f"{stem}.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(
        source=str(CSV_PATH.relative_to(PAPER)), source_sha256=hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
        pairing_source=str(SOURCE.relative_to(PAPER)),
        pairing_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        metric=METRIC, metric_units="percent", oracle_iteration_selection=True,
        includes_initial_model=False, quality_timing="post-update; index k corresponds to update k+1",
        n_comparisons=len(rows), soft_better=wins, hard_better=losses,
        median_soft_gain_percent=float(np.median(gain)), x_limits=[0, x_max], y_limits=[y_min, y_max],
        quality_sha256_by_run=quality_hashes,
        noise_color_scale=dict(series="4", actual_noise_baseline_relative_sigma=.01,
                               actual_noise_baseline_absolute_sigma_s=.05,
                               multiplier_limits=[.5, 2.], darker_is_noisier=True),
        noise_metadata_sha256=noise_metadata_sha256,
        points=[dict(pair_id=r["pair_id"], section=r["section"],
                     hard_min_avg_abs_pct_dev=r["hard_min_avg_abs_pct_dev"],
                     soft_min_avg_abs_pct_dev=r["soft_min_avg_abs_pct_dev"],
                     soft_gain_percent=r["soft_gain_percent"],
                     actual_noise_multiplier=float(level) if np.isfinite(level) else None)
                for r, level in zip(rows, noise_levels)],
    )
    (HERE / "soft_gain_vs_hard_avg_abs_pct_dev.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("n_comparisons", "soft_better", "hard_better", "median_soft_gain_percent")}, indent=2))
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
