"""Evaluate velocity RMSE at unchanged L-curve checkpoints and plot its gain."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np

from plot_soft_gain_distribution import SERIES

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
SOURCE = PAPER / "exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv"


def main():
    with SOURCE.open() as stream:
        all_rows = list(csv.DictReader(stream))
    quality_cache = {}
    rmse_rows = []
    for original in all_rows:
        row = dict(original)
        row['avg_abs_pct_dev_soft_gain_percent'] = row.pop('soft_gain_percent')
        row['metric'] = 'rms_m_s'
        row['soft_gain_percent'] = ''
        for mode in ('hard', 'soft'):
            row[f'{mode}_velocity_rmse_m_s_at_lcurve'] = ''
            if row['pair_status'] != 'completed':
                continue
            run_id = row[f'{mode}_run_id']
            if run_id not in quality_cache:
                path = PAPER.parent / 'runs' / run_id / 'quality.jsonl'
                quality_cache[run_id] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            index = int(row[f'{mode}_quality_iteration_index'])
            if index != int(row[f'{mode}_selected_update_number']) - 1:
                raise ValueError(f'Checkpoint timing mismatch: {run_id}')
            matches = [q for q in quality_cache[run_id] if q['iter'] == index]
            if len(matches) != 1:
                raise ValueError(f'Missing/duplicate checkpoint: {run_id}, {index}')
            value = float(matches[0]['rms_m_s'])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid velocity RMSE: {run_id}')
            np.testing.assert_allclose(float(row[f'{mode}_avg_abs_pct_dev_at_lcurve']),
                                       matches[0]['avg_abs_pct_dev'], rtol=0, atol=1e-12)
            row[f'{mode}_velocity_rmse_m_s_at_lcurve'] = value
        if row['pair_status'] == 'completed':
            h = row['hard_velocity_rmse_m_s_at_lcurve']
            s = row['soft_velocity_rmse_m_s_at_lcurve']
            row['soft_gain_percent'] = 100 * (h - s) / h
        rmse_rows.append(row)
    target = PAPER / 'exp_data/hard_soft_lcurve_velocity_rmse_topn_sweep.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rmse_rows[0]))
        writer.writeheader()
        writer.writerows(rmse_rows)
    temporary.replace(target)
    all_rows = rmse_rows
    rows = [r for r in all_rows if r["included_in_plot"] == "True" and r["pair_status"] == "completed"]
    excluded = [r["pair_id"] for r in all_rows if r["included_in_plot"] != "True"]
    unfinished = [r["pair_id"] for r in all_rows if r["pair_status"] != "completed"]
    if set(excluded) != {"section6_pair05", "section6_pair06"}:
        raise ValueError(f"Unexpected exclusions: {excluded}")
    hard = np.asarray([float(r["hard_velocity_rmse_m_s_at_lcurve"]) for r in rows])
    soft = np.asarray([float(r["soft_velocity_rmse_m_s_at_lcurve"]) for r in rows])
    gain = np.asarray([float(r["soft_gain_percent"]) for r in rows])
    if not np.all(np.isfinite(hard)) or not np.all(np.isfinite(soft)) or np.any(hard <= 0):
        raise ValueError("Invalid selected metrics")
    np.testing.assert_allclose(gain, 100 * (hard - soft) / hard, atol=1e-12)
    top_n = np.asarray([int(r["soft_weights_top_n"]) for r in rows])
    if np.any(top_n < 1):
        raise ValueError("Invalid weights_top_n")
    noise_levels = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        if row["section"] != "4":
            continue
        path = PAPER.parents[1] / "forward_modeling/experiments/output" / row["experiment_id"] / "metadata.json"
        config = json.loads(path.read_text())["noise"]["config"]
        multiplier = config["relative_sigma"] / .01
        if not np.isclose(multiplier, config["absolute_sigma_s"] / .05):
            raise ValueError(f"Noise components have unequal scaling: {row['pair_id']}")
        noise_levels[i] = multiplier
    cmap = LinearSegmentedColormap.from_list("actual_noise", plt.get_cmap("Greens")(np.linspace(.38, .92, 256)))
    norm = Normalize(vmin=.5, vmax=2.)
    x_min = float(max(0, top_n.min() - 10))
    x_max = float(top_n.max() + 10)
    y_min = float(min(-2, np.floor(gain.min() / 2) * 2))
    y_max = float(max(2, np.ceil(gain.max() / 2) * 2))
    wins = int(np.count_nonzero(gain > 0))
    losses = int(np.count_nonzero(gain < 0))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(9.5, 6.5))
        ax.axhline(0, color="#444444", linestyle="--", linewidth=1.2, zorder=2)
        for section, (label, color) in SERIES.items():
            mask = np.asarray([r["section"] == section for r in rows])
            if section == "4":
                noise_scatter = ax.scatter(top_n[mask], gain[mask], c=noise_levels[mask], cmap=cmap,
                                           norm=norm, s=58, edgecolors="white", linewidths=.65, zorder=3,
                                           label=f"{label} (n={int(mask.sum())})")
            else:
                ax.scatter(top_n[mask], gain[mask], c=color, s=58, alpha=.87,
                           edgecolors="white", linewidths=.65, zorder=3,
                           label=f"{label} (n={int(mask.sum())})")
        for i, row in enumerate(rows):
            if "dminsweep" in row["pair_id"]:
                ax.annotate(f"dmin = {row['weights_min_distance']}",
                            (top_n[i], gain[i]), xytext=(12, 5),
                            textcoords="offset points", fontsize=8.5,
                            color=SERIES["5"][1],
                            arrowprops=dict(arrowstyle="-", color=SERIES["5"][1], linewidth=.7))
        bar_axes = ax.inset_axes([.64, .77, .30, .027])
        bar = fig.colorbar(noise_scatter, cax=bar_axes, orientation="horizontal")
        bar.set_ticks([.5, .75, 1., 1.5, 2.], labels=["0.5×", "0.75×", "1×", "1.5×", "2×"])
        bar.ax.tick_params(labelsize=8)
        bar.ax.set_title("Actual noise (darker = noisier)", fontsize=8.5, pad=8)
        bar.outline.set_linewidth(.5)
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(sorted(set(top_n.tolist())))
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_axisbelow(True)
        ax.grid(color="#dddddd", linewidth=.6)
        ax.set_xlabel("Number of retained hypotheses in soft (weights_top_n)", labelpad=10)
        ax.set_ylabel("Soft gain in velocity RMSE (%)", labelpad=10)
        ax.set_title("Soft gain versus top-N: L-curve-selected models", loc="left", pad=31)
        ax.text(0, 1.025, r"Viewer metric: velocity RMSE (m/s); gain = $100\,(E_{hard,L}-E_{soft,L})/E_{hard,L}$",
                transform=ax.transAxes, fontsize=10, color="#444444")
        ax.text(.98, .96, "Soft better", transform=ax.transAxes, ha="right", va="top",
                fontsize=9.5, color="#555555")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2,
                  frameon=False, fontsize=8.8, labelspacing=.8, columnspacing=1.4)
        fig.text(.53, .10,
                 f"N = {len(rows)} comparisons  |  Soft better: {wins}  |  Hard better: {losses}"
                 "  |  Dashed line: no difference",
                 ha="center", fontsize=9.5, color="#444444")
        fig.text(.53, .025,
                 f"Completed comparisons only; {len(unfinished)} pending/running/failed pairs omitted. Shortlist size varies.\n"
                 "Selection does not use truth; error evaluation does. Two homogeneous-start pairs excluded.\n"
                 "No jitter; coincident points may overlap. Some hard controls are reused.",
                 ha="center", fontsize=8.8, color="#555555", linespacing=1.5)
        fig.subplots_adjust(left=.10, right=.98, top=.81, bottom=.36)
        for extension in ("png", "pdf"):
            fig.savefig(HERE / f"soft_rmse_gain_vs_top_n_lcurve_topn_sweep.{extension}", dpi=220, facecolor="white")
        plt.close(fig)
    report = dict(source=str(SOURCE.relative_to(PAPER)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  n_csv_pairs=len(all_rows), n_plotted_pairs=len(rows), excluded_pair_ids=excluded,
                                    unfinished_pair_ids=unfinished,
                  selection_uses_truth=False, evaluation_uses_truth=True, metric='rms_m_s',
                                    evaluation_csv=str(target.relative_to(PAPER)),
                                    evaluation_csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                                    checkpoints_unchanged=True,
                  soft_better=wins, hard_better=losses, x_limits=[x_min, x_max], y_limits=[y_min, y_max],
                  points=[dict(pair_id=r["pair_id"], section=r["section"],
                               n_candidates=int(r["soft_n_candidates"]), weights_top_n=int(count),
                                                              weights_min_distance=int(r["weights_min_distance"]),
                                                              hard_n_candidates=int(r["hard_n_candidates"]),
                                                              hard_weights_top_n=int(r["hard_weights_top_n"]),
                               hard_velocity_rmse_m_s=float(x), soft_velocity_rmse_m_s=float(s),
                               soft_gain_percent=float(y),
                               hard_selected_update=int(r["hard_selected_update_number"]),
                               soft_selected_update=int(r["soft_selected_update_number"]),
                               actual_noise_multiplier=float(level) if np.isfinite(level) else None)
                          for r, count, x, s, y, level in zip(rows, top_n, hard, soft, gain, noise_levels)])
    (HERE / "soft_rmse_gain_vs_top_n_lcurve_topn_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_csv_pairs", "n_plotted_pairs", "excluded_pair_ids", "soft_better", "hard_better")}, indent=2))


if __name__ == "__main__":
    main()
