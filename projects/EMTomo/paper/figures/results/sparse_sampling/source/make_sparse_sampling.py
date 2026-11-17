"""One saved event: 10,000 densely spaced vs 50 sparse location hypotheses.

Rebuild: python -B paper/figures/results/sparse_sampling/source/make_sparse_sampling.py
Actual saved iter_0 weights are used, with the exact common initial model/picks.
Missing station fields are recomputed once and the event likelihood is cached.
No candidates are optimized, no rays traced, no velocity inversion is rerun.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
sys.path.insert(0,str(ROOT))
from eikonal import station_travel_time_fields
from experiment_data import load_tomography_experiment
from instruments.likelihood import PickNoise, negative_log_likelihood, candidate_weights
from velocity_model import VelocityModel

EXPERIMENT = "onesided_nx6_24x12x12_18st_10ev_noise01_pick005_seed42"
EVENT_INDEX = 0  # First event in saved order, not selected by closeness of the distributions.
BLUE, ORANGE = "#28659a", "#db7627"
SLAB_THICKNESS_FINE_CELLS = 2
SLAB_TOLERANCE_M = 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_runs(event_index=EVENT_INDEX):
    with (ROOT/"studies/article_inventory/runs.csv").open(newline="") as stream:
        inventory = list(csv.DictReader(stream))
    result = {}
    for name,k,d in [("dense",10000,1),("sparse",50,7)]:
        matches = [r for r in inventory if r["source_id"] == EXPERIMENT
                   and r["n_candidates"] == str(k) and r["min_distance"] == str(d)]
        assert len(matches)==1
        run = ROOT/"runs"/matches[0]["run_id"]
        meta=read(run/"meta.json")
        assert meta["source_experiment"]["id"]==EXPERIMENT
        assert meta["run_params"]["n_candidates"]==meta["run_params"]["weights_top_n"]==k
        assert meta["run_params"]["weights_min_distance"]==d
        path=run/f"iter_0/event_{event_index}/weights.npz"
        with np.load(path,allow_pickle=False) as w:
            positions=w["positions"].copy(); weights=w["weight_values"].copy(); cells=w["weight_indices"].copy()
            assert positions.shape==(k,3) and weights.shape==(k,)
            assert np.array_equal(w["weight_shape"],meta["grid_info"]["fine_shape"])
        assert np.isfinite(positions).all() and np.isfinite(weights).all()
        assert np.all(weights>=0) and np.isclose(weights.sum(),1)
        nearest=cKDTree(cells).query(cells,k=2,p=np.inf)[0][:,1]
        assert nearest.min()>=d
        side=meta["grid_info"]["fine_cell_size"]
        coords=(positions+.5)*side
        mean=weights@coords
        centered=coords-mean
        covariance=(centered*weights[:,None]).T@centered
        result[name]=dict(run=run,meta=meta,positions=positions,weights=weights,cells=cells,coords_m=coords,
                          weighted_mean_m=mean,weighted_std_m=np.sqrt(np.diag(covariance)),
                          map_candidate_index=int(np.argmax(weights)),map_position_m=coords[np.argmax(weights)].copy(),
                          covariance_m2=covariance,effective_candidate_count=float(1/np.sum(weights**2)),
                          min_seed_distance_fine_cells=float(nearest.min()),
                          weights_file_sha256=sha(path),meta_sha256=sha(run/"meta.json"))
    dense,sparse=result["dense"],result["sparse"]
    assert sha(dense["run"]/"iter_0/model.npy")==sha(sparse["run"]/"iter_0/model.npy")
    assert np.array_equal(dense["meta"]["station_locs"],sparse["meta"]["station_locs"])
    assert dense["meta"]["grid_info"]==sparse["meta"]["grid_info"]
    for key in ["temperature","weight_noise_relative_sigma","weight_noise_absolute_sigma_s","weight_model_sigma_s"]:
        assert dense["meta"]["run_params"][key]==sparse["meta"]["run_params"][key]
    return result


def get_field(runs,experiment,event_index=EVENT_INDEX,cache_stem="likelihood_field"):
    dense=runs["dense"]
    model_path=dense["run"]/"iter_0/model.npy"
    p=dense["meta"]["run_params"]
    noise=PickNoise(p["weight_noise_relative_sigma"],p["weight_noise_absolute_sigma_s"],p["weight_model_sigma_s"])
    fingerprint=dict(model_sha256=sha(model_path),event_id=experiment.event_ids[event_index],
                     arrivals_sha256=hashlib.sha256(experiment.arrival_times_s[event_index].tobytes()).hexdigest(),
                     stations_sha256=hashlib.sha256(experiment.station_coordinates_m.tobytes()).hexdigest(),
                     code_sha256={name:sha(ROOT/name) for name in ["eikonal.py","instruments/likelihood.py"]},
                     noise=dict(relative=p["weight_noise_relative_sigma"],absolute=p["weight_noise_absolute_sigma_s"],
                                model=p["weight_model_sigma_s"]),
                     grid=dense["meta"]["grid_info"])
    cache=HERE/f"{cache_stem}.npz"
    provenance=HERE/f"{cache_stem}.json"
    if cache.exists() and provenance.exists():
        saved=read(provenance)
        if saved["fingerprint"]==fingerprint and saved["cache_sha256"]==sha(cache):
            with np.load(cache,allow_pickle=False) as data:
                print("Reusing verified likelihood field cache.",flush=True)
                return {name:data[name].copy() for name in data.files},saved
    model=VelocityModel(np.load(model_path,allow_pickle=False),p["cell_size"]).refined(p["subdivision"])
    assert np.all(model.velocity==5000.)
    print("Recomputing 18 station fields in the common fixed model; no inversion.",flush=True)
    fields=station_travel_time_fields(model,experiment.station_coordinates_m,n_workers=4)
    cost=negative_log_likelihood(fields,experiment.arrival_times_s[event_index],noise)
    assert cost.shape==tuple(dense["meta"]["grid_info"]["fine_shape"]) and np.isfinite(cost).all()
    data=dict(cost=cost)
    verification={}
    for name,run in runs.items():
        predicted=np.array([map_coordinates(field,run["positions"].T,order=1,mode="nearest",prefilter=False)
                            for field in fields])
        candidate_cost=negative_log_likelihood(predicted,experiment.arrival_times_s[event_index],noise)
        recomputed=candidate_weights(candidate_cost,p["temperature"])
        np.testing.assert_allclose(recomputed,run["weights"],rtol=2e-5,atol=1e-9)
        data[name+"_candidate_cost"]=candidate_cost
        verification[name]=dict(max_weight_difference=float(np.max(np.abs(recomputed-run["weights"]))),
                                candidate_cost_min=float(candidate_cost.min()),candidate_cost_max=float(candidate_cost.max()))
    np.savez_compressed(cache,**data)
    saved=dict(fingerprint=fingerprint,verification=verification,cache_sha256=sha(cache),
               reconstruction="Current production scikit-fmm station fields and marginalized-origin Gaussian likelihood; saved candidate probabilities reproduced numerically.")
    provenance.write_text(json.dumps(saved,indent=2)+"\n")
    return data,saved


def weighted_cdf(values,weights):
    order=np.argsort(values,kind="stable")
    x=values[order];f=np.cumsum(weights[order]);f[-1]=1.
    return np.r_[x[0],x],np.r_[0,f]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id",help="Saved synthetic event ID, e.g. EVT_0905")
    parser.add_argument("--xlim",nargs=2,type=float,metavar=("MIN_KM","MAX_KM"))
    parser.add_argument("--full-depth",action="store_true")
    parser.add_argument("--filled-truth",action="store_true",help="Filled burgundy star without outline")
    parser.add_argument("--distance-style",action="store_true",
                        help="White-to-blue likelihood, all candidates shaded black-to-gray by plane distance, compact layout")
    args=parser.parse_args(argv)
    if args.xlim is not None and (not np.isfinite(args.xlim).all() or args.xlim[0]>=args.xlim[1]):
        parser.error("X limits must be finite and increasing")
    assert PNG.is_dir()
    experiment=load_tomography_experiment(EXPERIMENT)
    assert len(experiment.event_ids)==10 and len(experiment.station_ids)==18
    event_id=args.event_id or experiment.event_ids[EVENT_INDEX]
    if event_id not in experiment.event_ids:
        parser.error(f"Event {event_id} is not in the saved pilot experiment")
    event_index=experiment.event_ids.index(event_id)
    stem=f"dense_vs_sparse_{event_id}" if args.event_id else "dense_vs_sparse_event"
    data_stem=f"plotted_data_{event_id}" if args.event_id else "plotted_data"
    cache_stem=f"likelihood_field_{event_id}" if args.event_id else "likelihood_field"
    runs=load_runs(event_index)
    field,provenance=get_field(runs,experiment,event_index,cache_stem)
    cost=field["cost"]
    side=runs["dense"]["meta"]["grid_info"]["fine_cell_size"]/1000
    plane_index=int(np.clip(round(runs["dense"]["weighted_mean_m"][1]/(side*1000)-.5),0,cost.shape[1]-1))
    plane_y_m=(plane_index+.5)*side*1000
    half_width_m=SLAB_THICKNESS_FINE_CELLS*side*1000/2
    slice_cost=cost[:,plane_index,:]
    relative=np.exp(-(slice_cost-slice_cost.min()))
    slab_masks={name:np.abs(r["coords_m"][:,1]-plane_y_m)<=half_width_m+SLAB_TOLERANCE_M
                for name,r in runs.items()}
    selected={name:np.ones(len(r["weights"]),dtype=bool) if args.distance_style else slab_masks[name]
              for name,r in runs.items()}
    if not all(mask.any() for mask in selected.values()):
        raise ValueError("The projection must contain candidates from both runs")
    weight_values=np.concatenate([r["weights"][selected[name]]/r["weights"].max()
                                  for name,r in runs.items()])
    weight_norm=Normalize(float(weight_values.min()),1.)
    weight_cmap=LinearSegmentedColormap.from_list("candidate_weight",["#a7ddfa","#0045a0","#00194d"])
    likelihood_cmap=(LinearSegmentedColormap.from_list("white_blue_likelihood",["white","#9fd4ef","#004c99"])
                     if args.distance_style else "Greys_r")
    plane_distances={name:np.abs(r["coords_m"][:,1]-plane_y_m) for name,r in runs.items()}
    distance_scale_m=max(max(float(d.max()) for d in plane_distances.values()),np.finfo(float).tiny)
    point_gray={name:.8*np.clip(distance/distance_scale_m,0,1) for name,distance in plane_distances.items()}
    x=(np.arange(cost.shape[0])+.5)*side
    z=(np.arange(cost.shape[2])+.5)*side
    truth_m=np.asarray(experiment.reference_event_coordinates_m[event_index],dtype=float)
    if truth_m.shape!=(3,) or not np.isfinite(truth_m).all():
        raise ValueError("Expected finite synthetic hypocentre truth for this event")
    truth=truth_m/1000
    coords=np.vstack([*[r["coords_m"] for r in runs.values()],truth_m])/1000
    bounds=np.array([[max(0,np.floor(coords[:,0].min()*2)/2-.5),min(24,np.ceil(coords[:,0].max()*2)/2+.5)],
                     [max(0,np.floor(coords[:,2].min()*2)/2-.5),min(12,np.ceil(coords[:,2].max()*2)/2+.5)]])
    if args.xlim is not None:
        bounds[0]=args.xlim
    if args.full_depth:
        bounds[1]=[0,cost.shape[2]*side]
    mean_distance=float(np.linalg.norm(runs["dense"]["weighted_mean_m"]-runs["sparse"]["weighted_mean_m"]))
    wasserstein={axis:float(wasserstein_distance(runs["dense"]["coords_m"][:,i],runs["sparse"]["coords_m"][:,i],
                                                 runs["dense"]["weights"],runs["sparse"]["weights"]))
                 for i,axis in enumerate("XYZ")}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,
                         "savefig.facecolor":"white"})
    if args.distance_style:
        map_height, map_bottom, map_left, map_gap = 4.7, .65, 1.4, .28
        map_width=map_height*np.diff(bounds[0])[0]/np.diff(bounds[1])[0]
        cdf_left=map_left+2*map_width+map_gap+.72
        cdf_width, cdf_gap = 3., .8
        cdf_height=(map_height-cdf_gap)/2
        fig=plt.figure(figsize=(cdf_left+cdf_width+.2,6.2))
        fw,fh=fig.get_size_inches()
    else:
        fig=plt.figure(figsize=(16.5,6.2))
        grid=fig.add_gridspec(2,3,width_ratios=[1,1,.85],left=.055,right=.985,top=.91,bottom=.20,
                             wspace=.34,hspace=.55)
    top_axes=[]
    for col,name in enumerate(("dense","sparse")):
        r=runs[name]
        if args.distance_style:
            ax=fig.add_axes([(map_left+col*(map_width+map_gap))/fw,map_bottom/fh,map_width/fw,map_height/fh])
        else:
            ax=fig.add_subplot(grid[:,col])
        top_axes.append(ax)
        ax.imshow(relative.T,origin="upper",extent=(0,24,12,0),vmin=0,vmax=1,cmap=likelihood_cmap,
                  interpolation="nearest",rasterized=True)
        ax.contour(x,z,relative.T,levels=[.2,.5,.8,.95],colors="0.55",linewidths=.45,zorder=2)
        mask=selected[name]
        points=r["coords_m"][mask]/1000
        weights=r["weights"][mask]/r["weights"].max()
        if args.distance_style:
            gray=point_gray[name][mask]
            colors=np.column_stack((gray,gray,gray,np.ones(len(gray))))
            # Far, light-gray points first; near-plane black points remain visible on top.
            order=np.argsort(-plane_distances[name][mask],kind="stable")
            ax.scatter(points[order,0],points[order,2],s=10 if name=="dense" else 35,
                       marker=".",c=colors[order],edgecolors="none",linewidths=0,
                       zorder=3,rasterized=True)
        else:
            ax.scatter(points[:,0],points[:,2],s=10 if name=="dense" else 35,
                       marker=".",c=weights,cmap=weight_cmap,norm=weight_norm,
                       edgecolors="none",linewidths=0,alpha=1.,zorder=3,rasterized=True)
        mean=r["weighted_mean_m"]/1000
        map_position=r["map_position_m"]/1000
        ax.scatter(mean[0],mean[2],marker="D",s=85,facecolors="#009e45",
                   edgecolors="white",linewidths=.5,zorder=4,label="Weighted mean")
        ax.scatter(map_position[0],map_position[2],marker="D",s=45,facecolors="#e02020",
                   edgecolors="white",linewidths=.5,zorder=5,label="MAP candidate")
        ax.scatter(truth[0],truth[2],marker="*",s=175,
                   facecolors="#800020" if args.filled_truth else "none",
                   edgecolors="none" if args.filled_truth else "#e87500",
                   linewidths=0 if args.filled_truth else 1.5,zorder=6,
                   label="True hypocentre (XZ projection)")
        ax.legend(loc="lower right",fontsize=7,frameon=True,framealpha=.8)
        ax.set(xlim=bounds[0],ylim=bounds[1,::-1],xlabel="X (km)",ylabel="Depth (km)")
        ax.set_aspect("equal")
        if args.distance_style:
            ax.set_title(f'{name.capitalize()}: {len(r["weights"]):,} candidates\nY plane = {plane_y_m/1000:g} km',fontsize=10,pad=9)
            if col==1:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
        else:
            ax.set_title(f'{name.capitalize()}: {len(r["weights"]):,} candidates\n'
                         f'{mask.sum():,} in Y = {plane_y_m/1000:g} ± {half_width_m/1000:g} km',
                         fontsize=10,pad=9)
    for col,(axis,i) in enumerate([("X",0),("Depth",2)]):
        if args.distance_style:
            bottom=map_bottom+(cdf_height+cdf_gap if col==0 else 0)
            ax=fig.add_axes([cdf_left/fw,bottom/fh,cdf_width/fw,cdf_height/fh])
        else:
            ax=fig.add_subplot(grid[col,2])
        for name,color in [("dense",BLUE),("sparse",ORANGE)]:
            r=runs[name];values=r["coords_m"][:,i]/1000
            xx,ff=weighted_cdf(values,r["weights"])
            ax.step(xx,ff,where="post",color=color,lw=1.5,label=f'{len(r["weights"]):,} candidates')
        ax.set(xlim=bounds[col],ylim=(0,1),xlabel=f"{axis} (km)",ylabel="Cumulative weight")
        ax.set_title(f'{axis} · full-set weighted CDF',fontsize=10,pad=6)
        ax.grid(alpha=.22,lw=.5);ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.legend(fontsize=8,frameon=False,loc="lower right")
    if args.distance_style:
        cax=fig.add_axes([.55/fw,(map_bottom+map_height*.25)/fh,.10/fw,map_height*.5/fh])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=likelihood_cmap),cax=cax)
        bar.set_label("Relative likelihood on Y plane",fontsize=9)
        bar.ax.yaxis.set_ticks_position("left")
        bar.ax.yaxis.set_label_position("left")
        bar.ax.tick_params(labelsize=8)
    else:
        for col,(norm,cmap,label) in enumerate([
                (Normalize(0,1),likelihood_cmap,"Relative likelihood on Y plane"),
                (weight_norm,weight_cmap,"Candidate weight / run maximum")]):
            position=top_axes[col].get_position()
            cax=fig.add_axes([position.x0,.083,position.width,.022])
            bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation="horizontal")
            bar.set_label(label,fontsize=9)
            bar.ax.tick_params(labelsize=8)
    fig.savefig(PNG/f"{stem}.png",dpi=220,bbox_inches="tight",pad_inches=.12)
    fig.savefig(HERE/f"{stem}.pdf",dpi=220,bbox_inches="tight",pad_inches=.12)
    fig.savefig(PNG/f"{stem}_final.png",dpi=220,bbox_inches="tight",pad_inches=.12)
    fig.savefig(HERE/f"{stem}_final.pdf",dpi=220,bbox_inches="tight",pad_inches=.12)
    if args.distance_style:
        fig.savefig(PNG/f"{stem}_all_candidates_final.png",dpi=220,bbox_inches="tight",pad_inches=.12)
        fig.savefig(HERE/f"{stem}_all_candidates_final.pdf",dpi=220,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    arrays={"slice_relative_likelihood_xz":relative,"x_km":x,"depth_km":z,
            "plane_y_m":np.asarray(plane_y_m),"slab_half_width_m":np.asarray(half_width_m),
                        "true_hypocentre_m":truth_m}
    for name,r in runs.items():
        arrays[name+"_positions_m"]=r["coords_m"]
        arrays[name+"_weights"]=r["weights"]
        arrays[name+"_slab_mask"]=slab_masks[name]
        arrays[name+"_display_mask"]=selected[name]
        arrays[name+"_relative_weights"]=r["weights"]/r["weights"].max()
        arrays[name+"_weighted_mean_m"]=r["weighted_mean_m"]
        arrays[name+"_map_position_m"]=r["map_position_m"]
        arrays[name+"_map_candidate_index"]=np.asarray(r["map_candidate_index"]) 
        if args.distance_style:
            arrays[name+"_distance_to_plane_m"]=plane_distances[name]
            arrays[name+"_point_gray"]=point_gray[name]
    np.savez_compressed(HERE/f"{data_stem}.npz",**arrays)
    report=dict(experiment_id=EXPERIMENT,event_index=event_index,event_id=event_id,iteration=0,
       event_selection="Explicit user-selected saved event" if args.event_id else "First event in saved catalogue order, not chosen for favorable agreement.",
       view_bounds_xz_km=bounds.tolist(),
       common_model="Identical saved iter_0/model.npy; homogeneous 5000 m/s, coarse 1000 m, q=8, fine 125 m.",
       sampling="Deterministic likelihood-ranked cell selection and continuous refinement, not random iid Monte Carlo samples.",
       visualization=("Compact three columns: all dense/sparse candidates projected onto XZ; CDFs stacked at right; one vertical likelihood colorbar at left" if args.distance_style else
                             "Three columns: dense and sparse XZ slab projections, full-set CDFs at right"),
              plane_selection="Nearest fine-grid cell centre to the dense full-set weighted mean Y; same plane for both methods",
              plane_y_m=plane_y_m,slab_thickness_fine_cells=SLAB_THICKNESS_FINE_CELLS,
              slab_half_width_m=half_width_m,slab_selection_tolerance_m=SLAB_TOLERANCE_M,
              slab_counts={name:int(mask.sum()) for name,mask in slab_masks.items()},
              displayed_candidate_counts={name:int(mask.sum()) for name,mask in selected.items()},
              candidate_Y_filter_applied=not args.distance_style,
              slab_probability_mass={name:float(runs[name]["weights"][mask].sum()) for name,mask in slab_masks.items()},
              background="exp(-(cost(X,Y_plane,Z)-min_XZ cost_on_plane)); actual fixed-Y grid slice, not a profile or marginal",
              background_rendering=("White to blue, opaque; 0 is white and 1 is dark blue" if args.distance_style else
                                                  "Opaque Greys_r, limits 0–1; relative likelihood 1 maps to white and 0 maps to black"),
              candidate_colors=("Opaque grayscale: gray=0.8*abs(Y-Yplane)/global_max_distance; nearer=blacker; farthest=light gray" if args.distance_style else
                                              "Small filled point markers, light-blue to dark-blue by saved w / maximum w within each full run"),
                            candidate_opacity_policy="Opaque points; distance encoded by grayscale, not alpha",
                                                        candidate_distance_normalization_m=distance_scale_m if args.distance_style else None,
                            candidate_distance_style=args.distance_style,
                            background_vs_candidates=("Background shows likelihood; point grayscale shows geometric distance to plane, not probability or weight" if args.distance_style else
                                                        "Background is fixed-Y relative likelihood; candidate point weights are evaluated at each candidate's original 3-D position with per-run normalization, so they need not equal the underlying background value"),
              candidate_color_limits=None if args.distance_style else [float(weight_norm.vmin),float(weight_norm.vmax)],
              candidate_marker=".",candidate_opacity=1.,candidate_weights_modified=False,
                            true_hypocentre_m=truth_m.tolist(),truth_marker="Filled burgundy star, no outline" if args.filled_truth else "Unfilled orange star on both XZ projections",
                            truth_y_offset_from_plane_m=float(truth_m[1]-plane_y_m),
                            truth_in_candidate_slab=bool(abs(truth_m[1]-plane_y_m)<=half_width_m+SLAB_TOLERANCE_M),
                            truth_usage="Display only; no effect on candidates, weights, likelihood or plane selection; plot bounds expanded to include star",
              cdf_policy="Full original saved candidate sets, not conditioned or renormalized on the displayed slab; exact weighted empirical CDFs",
              mean_markers_plotted=True,
              summary_markers="Green diamond: sum(w_i * XYZ_i), all saved normalized weights; red diamond: saved argmax-weight candidate. Both projected onto XZ without displacement or jitter.",
              map_tie_policy="numpy argmax: first saved candidate on equal maximum weights",
       mean_location_separation_m=mean_distance,wasserstein_1d_m=wasserstein,
       caveat="Sparse X/depth marginals and ridge coverage are similar for this event. Full 3-D distributions differ, especially the much narrower sparse Y spread; no claim of general equivalence or posterior calibration.",
       runs={name:{key:(value.tolist() if isinstance(value,np.ndarray) else str(value.relative_to(ROOT)) if isinstance(value,Path) else value)
                   for key,value in r.items() if key not in ["meta","positions","weights","cells","coords_m"]}
             for name,r in runs.items()},
       field_reproduction=provenance,script_sha256=sha(Path(__file__)))
    (HERE/f"{stem}.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    assert all(p.suffix==".png" for p in PNG.iterdir())
    print("Mean separation (m):",mean_distance)
    print("Wasserstein X/Y/Z (m):",wasserstein)
    print("Saved",PNG/f"{stem}.png")


if __name__=="__main__":
    main()
