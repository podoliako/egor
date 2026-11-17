"""Write the Section 4.1--4.6 metric tables of results.tex from the metric JSON files.

python3 -B paper/paper_scripts/make_results_tables.py [--check]

Sources: figures/table_supplementary_metrics.json (all hard/soft rows and 4.1 MAPE/RMSE)
and figures/checkerboard_locations_p90.json-equivalent values already published for 4.1 P90.
Each table environment in results.tex is replaced in place, matched by its \\label.
"""
from pathlib import Path
import argparse
import json
import re

PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER / "results.tex"
METRICS = PAPER / "figures/table_supplementary_metrics.json"
# Published 4.1 P90 values (m): EMTomo before the twelfth update, tomoDD after three iterations.
P90_41 = {4: (505.9, 599.2), 6: (373.4, 588.1), 8: (402.5, 609.1)}

HEAD = r"""\begin{table}[!htb]
\centering
\begin{singlespace}
\small
\setlength{\tabcolsep}{3.5pt}
"""
TAIL = r"""\bottomrule
\end{tabular}
\end{singlespace}
\end{table}"""
METRIC_HEAD = (r"\multicolumn{2}{c}{Velocity} & Residual & \multicolumn{2}{c}{P90 (m)} \\",
               r"RMSE (m/s) & MAPE (\%) & RMS (s) & Max.\ wt. & Mean \\")


def metric_cells(m, soft):
    mean = f"{m['p90_weighted_mean_m']:.1f}" if soft else "---"
    return (f"{m['velocity_rmse_m_s']:.1f} & {m['velocity_mape_percent']:.2f} & {m['residual_rms_s']:.4f} & "
            f"{m['p90_map_m']:.1f} & {mean}")


def hs_table(label, caption, comment, lead_names, rows, groups=None):
    """rows: list of (lead cells, metrics, is_soft); groups: indices after which to add space."""
    n_lead = len(lead_names)
    spec = "l" * n_lead + "rrrrr"
    first = n_lead + 1
    lines = [HEAD.rstrip("\n"), f"% {comment}" if comment else None,
             f"\\caption{{{caption}}}", f"\\label{{{label}}}", r"\smallskip", f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
             "& " * n_lead + METRIC_HEAD[0],
             rf"\cmidrule(lr){{{first}-{first + 1}}}\cmidrule(lr){{{first + 3}-{first + 4}}}",
             " & ".join(lead_names) + " & " + METRIC_HEAD[1], r"\midrule"]
    for i, (lead, metrics, soft) in enumerate(rows):
        lines.append(" & ".join(lead) + " & " + metric_cells(metrics, soft) + r" \\")
        if groups and i in groups:
            lines.append(r"\addlinespace")
    lines.append(TAIL)
    return "\n".join(l for l in lines if l is not None)


def table41(rows):
    lines = [HEAD.rstrip("\n"),
             r"% Velocity: figures/checkerboard_velocity_metrics.json (RMSE) and figures/table_supplementary_metrics.json (MAPE).",
             r"\caption{Checkerboard with full coverage: whole-volume velocity errors on a common 31.25-m grid and "
             r"hypocenter P90 over all 1200 events. Initial errors are 250~m/s RMSE and 5.0\% MAPE. Oracle is the "
             r"trilinear oracle for the tomoDD parameterization; the constant-cell oracle error of EMTomo is zero because "
             r"the boundaries align. EMTomo locations precede the twelfth velocity update; tomoDD locations follow three "
             r"joint iterations.}",
             r"\label{tab:checkerboard_metrics}", r"\smallskip", r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{Velocity RMSE (m/s)} & \multicolumn{3}{c}{Velocity MAPE (\%)} & "
             r"\multicolumn{2}{c}{P90 (m)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"$n_x$ & EMTomo & tomoDD & Oracle & EMTomo & tomoDD & Oracle & EMTomo & tomoDD \\", r"\midrule"]
    for r in rows:
        e, d, o = (r["rmse_m_s"][k] for k in ("em", "dd", "dd_oracle"))
        me, md, mo = (r["mape_percent"][k] for k in ("em", "dd", "dd_oracle"))
        pe, pd = P90_41[r["nx"]]
        lines.append(f"{r['nx']} & {e:.1f} & {d:.1f} & {o:.1f} & {me:.2f} & {md:.2f} & {mo:.2f} & {pe:.1f} & {pd:.1f} \\\\")
    lines.append(TAIL)
    return "\n".join(lines)


def build(metrics):
    t = metrics["tables"]
    by = {name: {r["pair_id"]: r for r in rows} for name, rows in t.items()}
    out = {}
    note = ("Residual RMS and P90 refer to the location step in the reported model; the weighted-mean P90 "
            "is given for soft only. Max. wt.: largest-weight candidate; Mean: weighted mean of the "
            "retained candidates; Mode: hard or soft assignment.")

    def sweep(name, pairs_k50, pair_k100):
        rows = [(["Hard", "50", "1"], by[name][pairs_k50[0][1]]["hard"], False)]
        rows += [(["Soft", "50", used], by[name][pid]["soft"], True) for used, pid in pairs_k50]
        rows += [(["Hard", "100", "1"], by[name][pair_k100]["hard"], False),
                 (["Soft", "100", "100"], by[name][pair_k100]["soft"], True)]
        return rows

    out["tab:onesided_comparison"] = hs_table(
        "tab:onesided_comparison",
        "One-sided coverage after three updates for every run. Tested and used count searched candidates and "
        "locations contributing to the update. P90 covers all 1200 events relocated in the selected model. " + note,
        "L-curve metrics and checkpoints: figures/hard_soft_lcurve_comparison_metrics.json, onesided; "
        "MAPE, residuals and weighted-mean P90: figures/table_supplementary_metrics.json.",
        ["Mode", "Tested", "Used"],
        sweep("onesided", [("10", "section2_topnsweep_k50_top10"), ("25", "section2_topnsweep_k50_top25"),
                           ("50", "section2_pair01")], "section2_topnsweep_k100_top100"), groups={3})
    out["tab:two_clusters_comparison"] = hs_table(
        "tab:two_clusters_comparison",
        "Two station clusters after three updates for every run. Metrics and candidate counts follow "
        "Table~\\ref{tab:onesided_comparison}; P90 covers all 600 events.",
        "L-curve metrics and checkpoints: figures/hard_soft_lcurve_comparison_metrics.json, two_clusters; "
        "supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Mode", "Tested", "Used"],
        sweep("two_clusters", [("10", "section3_topnsweep_k50_top10"), ("25", "section3_topnsweep_k50_top25"),
                               ("50", "section3_pair01")], "section3_topnsweep_k100_top100"), groups={3})
    noise = [("0.5", "section4_pair01"), ("0.75", "section4_pair02"), ("1", "section4_pair07"),
             ("1.5", "section4_pair03"), ("2", "section4_pair04")]
    rows = []
    for level, pid in noise:
        rows += [([level, "Hard"], by["real_geometry_noise"][pid]["hard"], False),
                 (["", "Soft"], by["real_geometry_noise"][pid]["soft"], True)]
    out["tab:real_geometry_noise"] = hs_table(
        "tab:real_geometry_noise",
        "Realistic geometry: noise sweep with 50 candidates, evaluated after the selected three updates. Noise "
        "factors scale both 1\\% relative and 0.05~s absolute standard deviations. P90 covers all 3465 events. "
        "Metrics follow Table~\\ref{tab:onesided_comparison}.",
        "L-curve metrics and checkpoints: figures/hard_soft_lcurve_comparison_metrics.json, real_geometry_noise; "
        "supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Noise", "Mode"], rows, groups={1, 3, 5, 7})
    cand = [("30", "section4_topnsweep_k30_top30"), ("50", "section4_pair07"),
            ("100", "section4_topnsweep_k100_top100"), ("150", "section4_topnsweep_k150_top150"),
            ("200", "section4_topnsweep_k200_top200")]
    rows = []
    for tested, pid in cand:
        rows += [([tested, "Hard"], by["real_geometry_candidates"][pid]["hard"], False),
                 (["", "Soft"], by["real_geometry_candidates"][pid]["soft"], True)]
    out["tab:real_geometry_candidates"] = hs_table(
        "tab:real_geometry_candidates",
        "Candidate-count sweep at nominal noise, using the same selection and metrics as "
        "Table~\\ref{tab:real_geometry_noise}. Hard uses one candidate; soft uses all tested candidates. "
        "The 50-candidate pair is reused.",
        "L-curve metrics and checkpoints: figures/hard_soft_lcurve_comparison_metrics.json, "
        "real_geometry_candidates; supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Tested", "Mode"], rows, groups={1, 3, 5, 7})
    ret = [("40", "section5_pair22"), ("50", "section5_topnsweep_k128_top50"),
           ("75", "section5_topnsweep_k128_top75"), ("100", "section5_topnsweep_k128_top100"),
           ("128", "section5_topnsweep_k128_top128")]
    rows = [(["Hard", "1"], by["fault_retention"]["section5_pair22"]["hard"], False)]
    rows += [(["Soft", used], by["fault_retention"][pid]["soft"], True) for used, pid in ret]
    out["tab:fault_retention"] = hs_table(
        "tab:fault_retention",
        "Fault model: retained-count sweep with 128 tested candidates and 250-m separation. Final models after "
        "12 updates in both modes. P90 covers all 3776 events, located in the E-step preceding the twelfth "
        "update; residuals come from the same step. The hard control is reused. Metrics follow "
        "Table~\\ref{tab:onesided_comparison}.",
        "Final-model metrics: figures/final_comparison_metrics.json, fault_retention; "
        "supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Mode", "Used"], rows)
    rows = []
    for sep, dmin in (("500", "dmin4"), ("875", "dmin7")):
        for k, used in enumerate(("50", "128")):
            pair = by["fault_separation"][f"section5_dminsweep_k128_top{used}_{dmin}"]
            rows += [([sep if k == 0 else "", used, "Hard"], pair["hard"], False),
                     (["", "", "Soft"], pair["soft"], True)]
    out["tab:fault_separation"] = hs_table(
        "tab:fault_separation",
        "Fault model: seed-separation sweep with 128 tested candidates. Retained is the soft count; each soft "
        "run has its own matched hard control, which uses one candidate. Final models after 12 updates in both "
        "modes; timing and metrics follow Table~\\ref{tab:fault_retention}.",
        "Final-model metrics: figures/final_comparison_metrics.json, fault_separation; "
        "supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Sep. (m)", "Retained", "Mode"], rows, groups={1, 3, 5})
    het = {r["pair_id"]: r for r in t["heterogeneous"]}
    rows = []
    for start, pid in (("1-D", "section6_pair01"), ("Homogeneous", "section6_pair05")):
        rows += [([start, "Hard"], het[pid]["hard"], False), (["", "Soft"], het[pid]["soft"], True)]
    out["tab:heterogeneous_comparison"] = hs_table(
        "tab:heterogeneous_comparison",
        "Heterogeneous model: final models after nine updates for the 1-D start (control-RMSE stopping) and 20 "
        "for the homogeneous start (update limit). Initial whole-volume RMSE is 504.0 and 1074.9~m/s, "
        "respectively. P90 covers all 3465 events, located without the withheld picks in the E-step preceding "
        "the final update; residuals use the same training picks and step. Metrics follow "
        "Table~\\ref{tab:onesided_comparison}.",
        "Final-model metrics: figures/final_comparison_metrics.json, heterogeneous; cap20 pairs only; "
        "supplementary metrics: figures/table_supplementary_metrics.json.",
        ["Start", "Mode"], rows, groups={1})
    out["tab:checkerboard_metrics"] = table41(metrics["checkerboard_4_1"])
    return out


def replace_tables(text, tables):
    pieces, position, found = [], 0, set()
    for block in re.finditer(r"\\begin\{table\}\[!htb\].*?\\end\{table\}", text, flags=re.S):
        key = None
        for label in re.findall(r"\\label\{([^}]+)\}", block.group(0)):
            if label in tables:
                key = label
            elif label in ("tab:checkerboard_velocity", "tab:checkerboard_locations"):
                key = "tab:checkerboard_metrics"
        if key is None:
            continue
        pieces.append(text[position:block.start()])
        if key not in found:          # the first 4.1 table becomes the merged one; later copies are dropped
            pieces.append(tables[key])
        position = block.end()
        found.add(key)
    text = "".join(pieces) + text[position:]
    text = re.sub(r"\n{3,}", "\n\n", text)
    missing = set(tables) - found
    if missing:
        raise ValueError(f"Tables not found in results.tex: {sorted(missing)}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="print tables without writing")
    args = parser.parse_args()
    tables = build(json.loads(METRICS.read_text()))
    if args.check:
        for key, value in tables.items():
            print(value, "\n")
        return
    RESULTS.write_text(replace_tables(RESULTS.read_text(), tables))
    print(f"Updated {len(tables)} tables in {RESULTS}")


if __name__ == "__main__":
    main()
