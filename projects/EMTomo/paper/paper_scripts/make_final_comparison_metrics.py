"""Final-model metrics for all synthetic hard/soft tables (4.2--4.6).

python -B paper/paper_scripts/make_final_comparison_metrics.py [--update-tables]
Reads saved artifacts only. Velocity is final_model.npy after N updates; location
P90 uses the last saved E-step in V_(N-1), NOT a fresh relocation in V_N.
Existing early-checkpoint reports are retained unchanged as provenance.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import io
import json
import re

import numpy as np

from make_hard_soft_lcurve_metrics import read, rows, xyz, map_error, volume_reference

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper/figures"
OUTPUT = FIGURES / "final_comparison_metrics.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract():
    legacy = read(FIGURES/"hard_soft_lcurve_comparison_metrics.json")
    heterogeneous = read(FIGURES/"heterogeneous_lcurve_comparison_metrics.json")
    table_identities = copy.deepcopy(legacy["tables"])
    table_identities["heterogeneous"] = []
    for pair_id in ("section6_pair01", "section6_pair05"):
        records = {r["mode"]: r for r in heterogeneous["records"]
                   if r["pair_id"] == pair_id and r["role"] == "primary"}
        assert set(records) == {"hard", "soft"}
        table_identities["heterogeneous"].append(dict(pair_id=pair_id, start=records["hard"]["start"], **records))
    tree = ast.parse((ROOT/"velocity_model.py").read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="block_average_slowness")
    namespace = {"np":np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),"velocity_model.py","exec"),namespace)
    average = namespace["block_average_slowness"]
    sources, records, tables = {}, {}, {}

    def source(identifier):
        if identifier not in sources:
            directory = ROOT.parent/"forward_modeling/experiments/input"/identifier
            events = rows(directory/"events.csv")
            stations = rows(directory/"stations.csv")
            with np.load(directory/"model.npz",allow_pickle=False) as a:
                truth, cell = a["velocity"].copy(),float(a["cell_size_m"])
                assert np.all(a["origin_m"]==0)
            sources[identifier] = dict(directory=str(directory),truth=truth,cell=cell,
                ids=[r["event_id"] for r in events],events=xyz(events),stations=xyz(stations),
                input_sha256={name:sha(directory/name) for name in ["model.npz","events.csv","stations.csv"]})
        return sources[identifier]

    def evaluate(run_id,mode):
        if run_id in records:
            assert records[run_id]["mode"]==mode
            return records[run_id]
        run=ROOT/"runs"/run_id
        meta=read(run/"meta.json")
        params,grid=meta["run_params"],meta["grid_info"]
        identifier=meta["source_experiment"]["id"]
        data=source(identifier)
        assert params["candidate_mode"]==mode
        assert meta["source_experiment"]["model_sha256"]==data["input_sha256"]["model.npz"]
        assert np.array_equal(meta["station_locs"],data["stations"])
        quality=[json.loads(line) for line in (run/"quality.jsonl").read_text().splitlines()]
        n=len(quality)
        assert [r["iter"] for r in quality]==list(range(n))
        iterations=sorted(int(p.parent.name[5:]) for p in run.glob("iter_*/complete.json"))
        assert iterations==list(range(n))
        if (run/"stopping.json").exists():
            assert read(run/"stopping.json")["completed_updates"]==n
        else:
            assert params["n_cycles"]==n
        model_path=run/"final_model.npy"
        model=np.load(model_path,allow_pickle=False)
        initial=np.load(run/"initial_model.npy",allow_pickle=False)
        side=float(grid["coarse_cell_size"])
        reference=average(data["truth"],data["cell"],model.shape,side)
        logged_rmse=float(np.sqrt(np.mean((model-reference)**2)))
        np.testing.assert_allclose(logged_rmse,quality[-1]["rms_m_s"],rtol=1e-10,atol=1e-8)
        score=volume_reference(model,side,data["truth"],data["cell"])
        errors,digest,counts=[],hashlib.sha256(),[]
        iteration=run/f"iter_{n-1}"
        assert not (run/f"iter_{n}").exists(),"Unexpected post-final E-step: review location protocol"
        for i,(event_id,truth) in enumerate(zip(data["ids"],data["events"])):
            payload=(iteration/f"event_{i}/weights.npz").read_bytes()
            digest.update(f'{i}\t{event_id}\t{hashlib.sha256(payload).hexdigest()}\n'.encode())
            with np.load(io.BytesIO(payload),allow_pickle=False) as w:
                assert np.array_equal(w["weight_shape"],grid["fine_shape"])
                count=len(w["positions"])
                assert 1<=count<=params["weights_top_n"]
                counts.append(count)
                errors.append(map_error(w["positions"],w["weight_values"],truth,
                                        grid["fine_cell_size"],grid["fine_shape"]))
        record=dict(run_id=run_id,experiment_id=identifier,mode=mode,
            final_update_number=n,quality_iteration_index=n-1,weights_iteration_index=n-1,
            hypocenter_model_updates=n-1,model_path=str(model_path.relative_to(ROOT)),
            weights_directory=str(iteration.relative_to(ROOT)),
            velocity_rmse_m_s=score,initial_velocity_rmse_m_s=volume_reference(initial,side,data["truth"],data["cell"]),
            hypocenter_p90_m=float(np.percentile(errors,90,method="linear")),
            logged_block_reference_rmse_m_s=logged_rmse,
            logged_block_reference_mape_percent=quality[-1]["avg_abs_pct_dev"],
            n_events=len(errors),tested=params["n_candidates"],
            used=1 if mode=="hard" else params["weights_top_n"],
            dmin_m=params["weights_min_distance"]*grid["fine_cell_size"],
            saved_candidate_count_range=[min(counts),max(counts)],
            row_id_weights_sha256=digest.hexdigest(),final_model_sha256=sha(model_path),
            initial_model_sha256=sha(run/"initial_model.npy"),meta_sha256=sha(run/"meta.json"),
            run_params=params,grid_info=grid)
        records[run_id]=record
        print(f'{mode:4s} N={n:2d} RMSE={score:8.3f} P90={record["hypocenter_p90_m"]:9.3f} {run_id}',flush=True)
        return record

    for name,identities in table_identities.items():
        tables[name]=[]
        for row in identities:
            result=dict(pair_id=row["pair_id"])
            if "start" in row: result["start"]=row["start"]
            for mode in ("hard","soft"):
                result[mode]=evaluate(row[mode]["run_id"],mode)
            assert result["hard"]["experiment_id"]==result["soft"]["experiment_id"]
            assert result["hard"]["initial_model_sha256"]==result["soft"]["initial_model_sha256"]
            tables[name].append(result)
    return dict(
        checkpoint_policy="Last completed velocity model of each existing run; no early-checkpoint or truth-based selection.",
        velocity_metric="Whole-volume velocity RMSE against original truth, exact nested common-grid evaluation.",
        location_metric="Equal-event-weight linear P90 of 3-D Euclidean errors; largest saved candidate weight in both modes.",
        location_timing="Last saved E-step iter_(N-1) in V_(N-1), before the final velocity update to V_N. NOT post-final relocation. No new FMM/localization/inversion was run.",
        weights_digest="SHA256(row_index TAB event_id TAB weights_file_SHA256 LF)",
        scope="Synthetic hard/soft tables 4.2--4.6; section 4.1 and real-data results were already final and are unchanged.",
        tables=tables,records=list(records.values()),
        experiments={k:{key:value for key,value in v.items() if key in ["directory","cell","input_sha256"]}
                     for k,v in sources.items()},
        source_mapping_sha256={p.name:sha(p) for p in [FIGURES/"hard_soft_lcurve_comparison_metrics.json",
                                                     FIGURES/"heterogeneous_lcurve_comparison_metrics.json"]},
        script_sha256=sha(Path(__file__)))


def table_lines(report):
    output={}
    def candidate_rows(name):
        result=[];seen=set()
        for row in report["tables"][name]:
            for mode in ["hard","soft"]:
                r=row[mode]
                if r["run_id"] in seen:continue
                seen.add(r["run_id"])
                result.append(f'{mode.capitalize()} & {r["tested"]} & {r["used"]} & {r["velocity_rmse_m_s"]:.1f} & {r["hypocenter_p90_m"]:.1f} \\\\')
        return result
    output["onesided_comparison"]=candidate_rows("onesided")
    output["two_clusters_comparison"]=candidate_rows("two_clusters")
    for key,name,labels in [("real_geometry_noise","real_geometry_noise",["0.5","0.75","1","1.5","2"]),
                            ("real_geometry_candidates","real_geometry_candidates",["30","50","100","150","200"])]:
        output[key]=[]
        for label,row in zip(labels,report["tables"][name]):
            h,s=row["hard"],row["soft"]
            output[key].append(f'{label} & {h["velocity_rmse_m_s"]:.1f} & {s["velocity_rmse_m_s"]:.1f} & {h["hypocenter_p90_m"]:.1f} & {s["hypocenter_p90_m"]:.1f} \\\\')
    output["fault_retention"]=[]
    for mode,r in [("hard",report["tables"]["fault_retention"][0]["hard"]),
                   *[("soft",r["soft"]) for r in report["tables"]["fault_retention"]]]:
        output["fault_retention"].append(f'{mode.capitalize()} & {r["used"]} & {r["velocity_rmse_m_s"]:.1f} & {r["hypocenter_p90_m"]:.1f} \\\\')
    output["fault_separation"]=[]
    for row in report["tables"]["fault_separation"]:
        h,s=row["hard"],row["soft"]
        output["fault_separation"].append(f'{h["dmin_m"]:g} & {s["used"]} & {h["velocity_rmse_m_s"]:.1f} & {s["velocity_rmse_m_s"]:.1f} & {h["hypocenter_p90_m"]:.1f} & {s["hypocenter_p90_m"]:.1f} \\\\')
    output["heterogeneous_comparison"]=[]
    for start,row in zip(["1-D","Homogeneous"],report["tables"]["heterogeneous"]):
        for mode in ["hard","soft"]:
            r=row[mode]
            output["heterogeneous_comparison"].append(f'{start} & {mode.capitalize()} & {r["velocity_rmse_m_s"]:.1f} & {r["hypocenter_p90_m"]:.1f} \\\\')
    return output


def update_tables(report):
    path=ROOT/"paper/results.tex"
    original=path.read_text()
    updated=original
    # Only numeric data rows are replaced; captions, headers, prose and formatting stay byte-for-byte.
    for label,replacements in table_lines(report).items():
        pattern=rf'(\\label\{{tab:{label}\}}.*?\\begin\{{tabular\}}.*?\\midrule\n)(.*?)(\\bottomrule)'
        match=re.search(pattern,updated,re.S)
        assert match,label
        body=match.group(2).splitlines()
        indices=[i for i,line in enumerate(body) if '&' in line and line.rstrip().endswith('\\\\')]
        assert len(indices)==len(replacements),(label,len(indices),len(replacements))
        for index,line in zip(indices,replacements):body[index]=line
        updated=updated[:match.start(2)]+'\n'.join(body)+'\n'+updated[match.end(2):]
    strip=lambda text:re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}','',text,flags=re.S)
    assert strip(original)==strip(updated),"Prose or captions changed"
    path.write_text(updated)
    report["article_prose_and_captions_sha256"]=hashlib.sha256(strip(original).encode()).hexdigest()
    print("Updated seven table bodies only; article prose and captions unchanged.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-tables",action="store_true")
    args=parser.parse_args()
    report=extract()
    if args.update_tables:update_tables(report)
    OUTPUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f'Saved {len(report["records"])} final-run records to {OUTPUT}')


if __name__=="__main__":
    main()
