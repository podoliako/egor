"""Apply the article checkpoint policy without rerunning any extraction or solve.

4.2--4.4 retain their previous selected models (three updates).
4.5--4.6 use final models. Only numeric table bodies may be updated.
Run: python -B paper/paper_scripts/make_article_comparison_metrics.py --update-tables
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json

from make_final_comparison_metrics import update_tables

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/figures"
OUTPUT = FIGURES / "article_comparison_metrics.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-tables", action="store_true")
    args = parser.parse_args()
    final_path = FIGURES / "final_comparison_metrics.json"
    previous_path = FIGURES / "hard_soft_lcurve_comparison_metrics.json"
    report = copy.deepcopy(json.loads(final_path.read_text()))
    previous = json.loads(previous_path.read_text())
    early_tables = ["onesided", "two_clusters", "real_geometry_noise", "real_geometry_candidates"]
    earlier_records = {r["run_id"]: r for r in previous["records"]}
    for name in early_tables:
        report["tables"][name] = copy.deepcopy(previous["tables"][name])
    records = {}
    for name, pairs in report["tables"].items():
        for pair in pairs:
            for mode in ("hard", "soft"):
                summary = pair[mode]
                run_id = summary["run_id"]
                if run_id not in records:
                    record = copy.deepcopy(summary)
                    if name in early_tables:
                        record.update(copy.deepcopy(earlier_records[run_id]))
                        record["model_update_number"] = record["selected_update_k"]
                        record["checkpoint_kind"] = "selected"
                        assert record["model_update_number"] == 3
                    else:
                        record["model_update_number"] = record["final_update_number"]
                        record["checkpoint_kind"] = "final"
                    path = ROOT / record["model_path"]
                    record["model_sha256"] = sha(path)
                    records[run_id] = record
                pair[mode] = records[run_id]
    report["records"] = list(records.values())
    report["checkpoint_policy"] = "4.2--4.4: prior selected models after 3 updates. 4.5--4.6: final_model.npy after each run ends."
    report["location_timing"] = "4.2--4.4: E-step in the selected V_3 model. 4.5--4.6: last saved E-step in V_(N-1), before the final velocity update, not a new relocation in V_N."
    report["scope"] = "Article tables and comparison figure for 4.2--4.6; only 4.2--4.4 restored to their previous checkpoints."
    report["source_reports_sha256"] = {p.name: sha(p) for p in [previous_path, final_path]}
    report["script_sha256"] = sha(Path(__file__))
    if args.update_tables:
        update_tables(report)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Saved article policy: 4.2--4.4 selected V_3; 4.5--4.6 final models.")


if __name__ == "__main__":
    main()
