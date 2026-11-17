"""Section 4.3: initial and last saved location-step error densities.

python -B paper/figures/results/discussion/source/make_section43_location_errors.py
Shared density renderer for 4.3 and 4.5; maximum-weight candidate for both modes. No solves.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.stats import gaussian_kde

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PNG=HERE.parent/"png"
BANDWIDTH_M=150.
COLORS={"hard":"#336fa5","soft":"#e18435"}
CASES={
    "4.3":dict(table="two_clusters",pair_id="section3_pair01",n_events=600,n_stations=22,
               candidates=50,relative_sigma=.01,absolute_sigma=.05,
               subtitle="Two station clusters · 600 events · 1% relative + 0.05 s absolute pick noise · 50 tested candidates",
               stem="section43_hypocenter_error_initial_final"),
    "4.5":dict(table="fault_separation",pair_id="section5_dminsweep_k128_top128_dmin7",n_events=3776,n_stations=26,
               candidates=128,relative_sigma=.02,absolute_sigma=.02,
               subtitle="Vertical fault · 3776 events · 2% relative + 0.02 s absolute pick noise · 128 candidates · min_dist = 7",
               stem="section45_k128_dmin7_hypocenter_error_initial_final"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(section="4.3"):
    case=CASES[section]
    mapping=ROOT/"paper/figures/hard_soft_lcurve_comparison_metrics.json"
    report=json.loads(mapping.read_text())
    pair=next(r for r in report["tables"][case["table"]] if r["pair_id"]==case["pair_id"])
    identifier=pair["frozen_csv_row"]["experiment_id"]
    source=ROOT.parent/"forward_modeling/experiments/input"/identifier
    with (source/"events.csv").open(newline="") as stream:
        events=list(csv.DictReader(stream))
    truth=np.array([[float(row[k+"_m"]) for k in "xyz"] for row in events])
    assert len(events)==case["n_events"]
    records={}
    for mode in ("hard","soft"):
        run=ROOT/"runs"/pair[mode]["run_id"]
        meta=json.loads((run/"meta.json").read_text())
        params=meta["run_params"]
        assert params["n_candidates"]==params["weights_top_n"]==case["candidates"]
        assert params["weights_min_distance"]==7 and params["candidate_mode"]==mode
        assert params["weight_noise_relative_sigma"]==case["relative_sigma"] and params["weight_noise_absolute_sigma_s"]==case["absolute_sigma"]
        assert meta["source_experiment"]["id"]==identifier
        history=[json.loads(line) for line in (run/"quality.jsonl").read_text().splitlines()]
        assert [r["iter"] for r in history]==list(range(12))
        fine=meta["grid_info"]["fine_cell_size"]
        for name,iteration in [("initial",0),("final",11)]:
            assert json.loads((run/f"iter_{iteration}/complete.json").read_text())["iter"]==iteration
            errors=[];estimates=[];digest=hashlib.sha256()
            for i,target in enumerate(truth):
                path=run/f"iter_{iteration}/event_{i}/weights.npz"
                digest.update(f'{i}\t{events[i]["event_id"]}\t{sha(path)}\n'.encode())
                with np.load(path,allow_pickle=False) as weights:
                    positions=weights["positions"];w=weights["weight_values"]
                    assert np.isfinite(positions).all() and np.isfinite(w).all()
                    assert np.all(w>=0) and np.isclose(w.sum(),1)
                    estimate=(positions[np.argmax(w)]+.5)*fine
                    estimates.append(estimate)
                    errors.append(np.linalg.norm(estimate-target))
            errors=np.array(errors)
            records[name+"_"+mode]=dict(mode=mode,stage=name,run_id=pair[mode]["run_id"],
                  location_iteration_index=iteration,velocity_updates_before_location=iteration,
                  estimates_m=np.array(estimates),errors_m=errors,meta_sha256=sha(run/"meta.json"),
                  row_id_weights_sha256=digest.hexdigest(),median_m=float(np.median(errors)),
                  p90_m=float(np.percentile(errors,90,method="linear")),
                  rmse_m=float(np.sqrt(np.mean(errors**2))),max_m=float(errors.max()))
    assert np.array_equal(records["initial_hard"]["estimates_m"],records["initial_soft"]["estimates_m"])
    return records,events,dict(section=section,experiment_id=identifier,pair_id=pair["pair_id"],n_events=len(events),
             n_stations=case["n_stations"],relative_noise_sigma=case["relative_sigma"],absolute_noise_sigma_s=case["absolute_sigma"],
             tested_candidates=case["candidates"],soft_retained_candidates=case["candidates"],min_distance_fine_cells=7,
             source_events_sha256=sha(source/"events.csv"),mapping_sha256=sha(mapping))


def reflected_density(errors,x):
    # Reflect at zero to keep location-error density on nonnegative support.
    samples=np.r_[errors,-errors]
    kde=gaussian_kde(samples,bw_method=BANDWIDTH_M/np.std(samples,ddof=1))
    return 2*kde(x)


def main(section="4.3"):
    assert PNG.is_dir()
    case=CASES[section]
    records,events,metadata=extract(section)
    upper=np.ceil((max(r["max_m"] for r in records.values())+4*BANDWIDTH_M)/500)*500
    x=np.linspace(0,upper,1500)
    densities={name:reflected_density(r["errors_m"],x) for name,r in records.items()}
    for values in densities.values():
        area=float(np.trapezoid(values,x))
        assert abs(area-1)<1e-4,area
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "savefig.facecolor":"white"})
    fig,axes=plt.subplots(1,2,figsize=(12.8,4.9),sharex=True,sharey=True)
    fig.subplots_adjust(left=.08,right=.98,bottom=.24,top=.76,wspace=.12)
    for ax,stage,title in zip(axes,["initial","final"],
                             ["(a) Initial · location step 1","(b) Final · location step 12"]):
        for mode in ["hard","soft"]:
            density=densities[stage+"_"+mode]
            ax.fill_between(x,density,color=COLORS[mode],alpha=.30,lw=0)
            ax.plot(x,density,color=COLORS[mode],lw=1.8,ls="-" if mode=="hard" else "--",
                    label=mode.capitalize())
        ax.set(title=title,xlim=(0,upper),ylim=(0,max(v.max() for v in densities.values())*1.12),
               xlabel="3-D hypocenter error (m)")
        ax.xaxis.set_major_locator(MultipleLocator(500 if upper<=3000 else 1000))
        ax.ticklabel_format(axis="y",style="sci",scilimits=(-3,-3))
        ax.grid(alpha=.2,lw=.5)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right",frameon=False)
        hard,soft=records[stage+"_hard"],records[stage+"_soft"]
        ax.text(.97,.68,f'P90: hard {hard["p90_m"]:.0f} m\n        soft {soft["p90_m"]:.0f} m',
                transform=ax.transAxes,ha="right",va="top",fontsize=9,linespacing=1.5)
        if stage=="initial":
            ax.text(.97,.36,"Initial locations coincide;\nthe two densities overlap exactly.",
                    transform=ax.transAxes,ha="right",va="top",fontsize=9,color="#555555")
    axes[0].set_ylabel("Probability density (m$^{-1}$)")
    fig.suptitle(f"§{section} · Hypocenter errors: hard vs soft",fontsize=15,y=.97)
    fig.text(.5,.87,case["subtitle"],
             ha="center",fontsize=10,color="#444444")
    fig.text(.5,.12,"Maximum-weight candidate in both modes · reflected density estimate with common 150-m bandwidth",
             ha="center",fontsize=9,color="#555555")
    fig.text(.5,.065,"Final panel uses the last saved location step, before the twelfth velocity update; no post-final relocation.",
             ha="center",fontsize=9,color="#555555")
    stem=case["stem"]
    fig.savefig(PNG/(stem+".png"),dpi=220)
    fig.savefig(HERE/(stem+".pdf"),dpi=220)
    plt.close(fig)
    with (HERE/(stem+"_events.csv")).open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(["event_id",*records.keys()])
        for i,event in enumerate(events):
            writer.writerow([event["event_id"],*[r["errors_m"][i] for r in records.values()]])
    report=dict(**metadata,estimator="Largest saved candidate weight in both modes; first argmax on exact ties.",
          coordinate_conversion="(positions+0.5)*fine_cell_size_m",event_order=f"Original events.csv row order; all {len(events)} events equal weight.",
          checkpoints="iter_0 E-step in the initial V0 model and iter_11 E-step in V11 before update 12; not a post-final relocation in V12.",
          density=dict(method="Gaussian KDE with reflection at zero",bandwidth_m=BANDWIDTH_M,
                       common_x_limits_m=[0,float(upper)],normalization="Each density integrates to one on nonnegative errors; numerical area checked."),
          initial_estimates_identical=True,
          records={name:{k:v for k,v in record.items() if k not in ["errors_m","estimates_m"]} for name,record in records.items()},
          script_sha256=sha(Path(__file__)))
    (HERE/(stem+".json")).write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(HERE/(stem+"_densities.npz"),x_m=x,**densities)
    print("Initial hard/soft estimates are identical.")
    for name,r in records.items():print(name,"median",r["median_m"],"P90",r["p90_m"])
    print("Saved",PNG/(stem+".png"))


if __name__=="__main__":
    main()
