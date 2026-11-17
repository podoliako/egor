"""Clean discussion plot: soft gain versus weights_top_n for the article-table pairs only.

Pairs are exactly those of the hard/soft tables in Sections 4.2--4.6
(paper/figures/final_comparison_metrics.json -> tables), deduplicated.
Metric: viewer avg_abs_pct_dev, gain = 100 (E_hard - E_soft) / E_hard.
Checkpoints follow the article policy:
  * 4.2--4.4 (CSV sections 2--4): L-curve-selected V_k from the frozen CSV;
  * 4.5--4.6 (CSV sections 5--6): final model V_N, i.e. quality.jsonl[N-1].
No title, subtitle, footer, noise colorbar or call-outs.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
PNG = HERE.parent / "png"
PAPER = HERE.parents[3]
ROOT = PAPER.parent
SOURCE = PAPER / "exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv"
TABLES = PAPER / "figures/final_comparison_metrics.json"
LCURVE_TABLES = PAPER / "figures/hard_soft_lcurve_comparison_metrics.json"
STEM = "soft_gain_vs_top_n_lcurve"
P90_STEM = "soft_p90_gain_vs_top_n"

# Series colours follow plot_soft_gain_distribution.SERIES, except a muted green for series 4.
SERIES = {
    "2": ("Checkerboard: one-sided", "#4477AA"),
    "3": ("Checkerboard: two clusters", "#EE9944"),
    "4": ("Checkerboard: real geometry", "#6BA56F"),  # muted green; noise level not encoded
    "5": ("Fault: real geometry", "#AA3377"),
    "6": ("Complex: real geometry", "#66CCEE"),
}
LCURVE_SECTIONS = {"2", "3", "4"}
FINAL_SECTIONS = {"5", "6"}
DODGE = 1.045  # multiplicative x offset between series sharing one top-N value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_quality(run_id):
    lines = (ROOT / "runs" / run_id / "quality.jsonl").read_text().splitlines()
    quality = [json.loads(line) for line in lines if line.strip()]
    if [q["iter"] for q in quality] != list(range(len(quality))):
        raise ValueError(f"Non-contiguous quality log: {run_id}")
    return len(quality), quality[-1]


def load_points():
    with SOURCE.open() as stream:
        csv_rows = {r["pair_id"]: r for r in csv.DictReader(stream)}
    final = json.loads(TABLES.read_text())
    lcurve = json.loads(LCURVE_TABLES.read_text())
    table_rmse, table_p90 = {}, {}
    for name, table in (("lcurve", lcurve), ("final", final)):
        for rows in table["tables"].values():
            for row in rows:
                table_rmse[(name, row["pair_id"])] = (row["hard"]["velocity_rmse_m_s"],
                                                      row["soft"]["velocity_rmse_m_s"])
                table_p90[(name, row["pair_id"])] = (row["hard"]["hypocenter_p90_m"],
                                                     row["soft"]["hypocenter_p90_m"])
    points, seen = [], set()
    for table_name, rows in final["tables"].items():
        for row in rows:
            pair_id = row["pair_id"]
            if pair_id in seen:
                continue
            seen.add(pair_id)
            c = csv_rows[pair_id]
            if c["pair_status"] != "completed":
                raise ValueError(f"Incomplete table pair: {pair_id}")
            for mode in ("hard", "soft"):
                if c[f"{mode}_run_id"] != row[mode]["run_id"]:
                    raise ValueError(f"Run mismatch for {pair_id} {mode}")
            section = c["section"]
            if section in LCURVE_SECTIONS:
                policy = "lcurve"
                k = {m: int(c[f"{m}_selected_update_number"]) for m in ("hard", "soft")}
                err = {m: float(c[f"{m}_avg_abs_pct_dev_at_lcurve"]) for m in ("hard", "soft")}
            elif section in FINAL_SECTIONS:
                policy = "final"
                k, err = {}, {}
                for m in ("hard", "soft"):
                    n, q = final_quality(row[m]["run_id"])
                    if n != row[m]["final_update_number"]:
                        raise ValueError(f"Final update mismatch for {pair_id} {m}")
                    if not np.isclose(q["rms_m_s"], row[m]["logged_block_reference_rmse_m_s"], rtol=1e-10):
                        raise ValueError(f"Final RMSE mismatch for {pair_id} {m}")
                    k[m], err[m] = n, float(q["avg_abs_pct_dev"])
            else:
                raise ValueError(f"Unknown section {section}")
            gain = 100 * (err["hard"] - err["soft"]) / err["hard"]
            rh, rs = table_rmse[(policy, pair_id)]
            ph, ps = table_p90[(policy, pair_id)]
            points.append(dict(
                pair_id=pair_id, table=table_name, section=section, checkpoint_policy=policy,
                weights_top_n=int(c["soft_weights_top_n"]), n_candidates=int(c["soft_n_candidates"]),
                weights_min_distance=int(c["weights_min_distance"]),
                start=c["initial_kind"] + (":" + c["plot_exclusion_reason"] if c["plot_exclusion_reason"] else ""), hard_update=k["hard"], soft_update=k["soft"],
                hard_avg_abs_pct_dev=err["hard"], soft_avg_abs_pct_dev=err["soft"],
                soft_gain_percent=gain, table_rmse_gain_percent=100 * (rh - rs) / rh,
                hard_p90_m=ph, soft_p90_m=ps, p90_gain_percent=100 * (ph - ps) / ph,
                hard_run_id=row["hard"]["run_id"], soft_run_id=row["soft"]["run_id"]))
    return points


def dodged_x(top_n, section):
    x = top_n.astype(float).copy()
    for value in np.unique(top_n):
        present = [s for s in SERIES if np.any((top_n == value) & (section == s))]
        for k, s in enumerate(present):
            x[(top_n == value) & (section == s)] = value * DODGE ** (k - (len(present) - 1) / 2)
    return x


def plot(x, gain, section, ticks, ylabel, stem):
    step = 1 if np.ptp(gain) <= 8 else 5
    y_lo = float(min(-step, np.floor(gain.min() / step) * step))
    y_hi = float(max(step, np.ceil(gain.max() / step) * step))
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#888888", "xtick.color": "#555555",
                         "ytick.color": "#555555", "axes.labelcolor": "#222222",
                         "pdf.fonttype": 42, "savefig.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(7.0, 2.6))
        ax.axhline(0, color="#555555", linewidth=0.9, zorder=2)
        for s, (label, color) in SERIES.items():
            m = section == s
            ax.scatter(x[m], gain[m], s=34, color=color, alpha=0.9,
                       edgecolors="white", linewidths=0.8, zorder=3,
                       label=f"{label} ({int(m.sum())})")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(ticks[0] / 1.15, ticks[-1] * 1.15)
        ax.set_ylim(y_lo, y_hi)
        ax.set_yticks(np.arange(y_lo, y_hi + 1e-9, step))
        ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.6)
        ax.set_xlabel("Retained hypotheses in soft (top-N, log scale)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
                  fontsize=7.5, handletextpad=0.3, columnspacing=1.2, borderaxespad=0)
        fig.savefig(PNG / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.06)
        fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return y_lo, y_hi


def main():
    points = load_points()
    top_n = np.asarray([p["weights_top_n"] for p in points])
    gain = np.asarray([p["soft_gain_percent"] for p in points])
    section = np.asarray([p["section"] for p in points])
    x = dodged_x(top_n, section)
    for p, px in zip(points, x):
        p["plotted_x"] = float(px)
    ticks = sorted(set(top_n.tolist()) - {128})  # 128 crowds 100/150; its points stay
    y_lo, y_hi = plot(x, gain, section, ticks, "Soft gain in velocity error (%)", STEM)
    p90_gain = np.asarray([p["p90_gain_percent"] for p in points])
    p90_limits = plot(x, p90_gain, section, ticks, "Soft gain in hypocentre P90 (%)", P90_STEM)

    sign_mismatch = [p["pair_id"] for p in points
                     if np.sign(p["soft_gain_percent"]) != np.sign(p["table_rmse_gain_percent"])]
    report = dict(
        pair_source=str(TABLES.relative_to(PAPER)) + " -> tables (article Sections 4.2--4.6)",
        source=str(SOURCE.relative_to(PAPER)), source_sha256=sha(SOURCE),
        tables_sha256=sha(TABLES), lcurve_tables_sha256=sha(LCURVE_TABLES),
        metric="avg_abs_pct_dev (viewer); gain = 100 (E_hard - E_soft) / E_hard",
        checkpoint_policy={"sections_4.2-4.4": "L-curve-selected V_k (frozen CSV)",
                           "sections_4.5-4.6": "final V_N (quality.jsonl[N-1])"},
        n_plotted_pairs=len(points),
        soft_better=int(np.count_nonzero(gain > 0)), hard_better=int(np.count_nonzero(gain < 0)),
        sign_differs_from_table_rmse_gain=sign_mismatch,
        p90_metric="hypocenter P90 from the article tables (largest-weight candidate); gain = 100 (P90_hard - P90_soft) / P90_hard",
        p90_timing={"sections_4.2-4.4": "E-step in the L-curve-selected V_k",
                    "sections_4.5-4.6": "last E-step, in V_(N-1), before the final update"},
        p90_figure=P90_STEM, p90_soft_better=int(np.count_nonzero(p90_gain > 0)),
        p90_hard_better=int(np.count_nonzero(p90_gain < 0)), p90_y_limits=list(p90_limits),
        x_scale="log", x_dodge_factor=DODGE, y_limits=[y_lo, y_hi], points=points)
    (HERE / f"{STEM}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("n_plotted_pairs", "soft_better", "hard_better",
                                             "sign_differs_from_table_rmse_gain",
                                             "p90_soft_better", "p90_hard_better")}))
    for p in points:
        print(f"{p['pair_id']:40s} s{p['section']} {p['checkpoint_policy']:6s} N={p['weights_top_n']:3d} "
              f"k={p['hard_update']}/{p['soft_update']} gain={p['soft_gain_percent']:+.2f} rmse_gain={p['table_rmse_gain_percent']:+.2f} p90_gain={p['p90_gain_percent']:+.2f}")


if __name__ == "__main__":
    main()
