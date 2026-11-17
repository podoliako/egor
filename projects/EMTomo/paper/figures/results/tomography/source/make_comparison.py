"""Five synthetic pairs: True/Soft/Hard rows and XY/YZ/XZ columns.

Uses the article policy: selected V_3 in 4.2--4.4, final models in 4.5--4.6.
No inversions, interpolation smoothing or new checkpoint selection. PNG only in ../png; PDF, section arrays and provenance here.
Run: python -B paper/figures/results/tomography/source/make_comparison.py
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PNG = HERE.parent / "png"
INPUT = ROOT.parent / "forward_modeling/experiments/input"
SPACING_M = 25.
HORIZONTAL_DEPTH_M = 1600.
sys.path.insert(0, str(ROOT / "paper/paper_scripts"))
from sensitivity_envelope import expanded_envelope, cell_boundary_segments

THRESHOLD = 1.0
PADDING_CELLS = 1


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapped_coordinates(path, id_key):
    with path.open(newline="") as stream:
        return {r[id_key].removeprefix("BG."): np.array([float(r[k+"_m"]) for k in "xyz"])
                for r in csv.DictReader(stream)}


def verify_registration(source, fault_source, translation):
    report = {}
    for name,key in [("stations.csv","station_id"),("events.csv","event_id")]:
        points = mapped_coordinates(source/name,key)
        reference = mapped_coordinates(fault_source/name,key)
        ids = points.keys() & reference.keys()
        assert ids
        maximum = max(float(np.linalg.norm(points[i]+translation-reference[i])) for i in ids)
        assert maximum < 1e-6
        report[name] = dict(common_ids=len(ids), max_coordinate_difference_m=maximum)
    return report


def load_block(section, title, pair_id, experiment, records, y, x, offset, limits):
    source = INPUT / experiment
    path = source / "model.npz"
    with np.load(path,allow_pickle=False) as model:
        truth = model["velocity"].copy()
        truth_cell = float(model["cell_size_m"])
        assert np.all(model["origin_m"] == 0)
    extent = np.array(truth.shape)*truth_cell
    block = dict(section=section,title=title,pair_id=pair_id,experiment_id=experiment,
                 source_directory=str(source),native_to_common_translation_m=np.array(offset,dtype=float),
                 section_y_common_m=float(y),section_x_common_m=float(x),extent_m=extent,
                 color_limits_km_s=limits,fields={"true":(truth,truth_cell)},records={},
                 input_sha256={name:sha(source/name) for name in ["model.npz","stations.csv","events.csv"]})
    initials = []
    for mode in ["hard","soft"]:
        record = records[mode]
        run = ROOT / "runs" / record["run_id"]
        meta = read(run/"meta.json")
        assert meta["source_experiment"]["id"] == experiment
        assert meta["source_experiment"]["model_sha256"] == sha(path)
        assert meta["run_params"]["candidate_mode"] == mode
        k = record["model_update_number"]
        checkpoint = ROOT/record["model_path"]
        assert sha(checkpoint) == record["model_sha256"]
        model = np.load(checkpoint,allow_pickle=False)
        cell = float(meta["grid_info"]["coarse_cell_size"])
        assert np.isfinite(model).all() and np.all(model>0)
        np.testing.assert_allclose(np.array(model.shape)*cell,extent)
        common_cell=float(np.gcd(int(cell),int(truth_cell)))
        def repeat_cells(field, factor):
            for axis in range(3):
                field=field.repeat(factor,axis=axis)
            return field
        rmse=float(np.sqrt(np.mean((repeat_cells(model,int(cell/common_cell))
                                    -repeat_cells(truth,int(truth_cell/common_cell)))**2)))
        np.testing.assert_allclose(rmse,record["velocity_rmse_m_s"],rtol=1e-10,atol=1e-8)
        complete_index = k-1 if record["checkpoint_kind"] == "final" else k
        assert read(run/f"iter_{complete_index}/complete.json")["iter"] == complete_index
        initials.append(np.load(run/"initial_model.npy",allow_pickle=False))
        p=meta["run_params"]
        block["fields"][mode] = (model,cell)
        block["records"][mode] = dict(run_id=record["run_id"],model_update_number=k,
             checkpoint_kind=record["checkpoint_kind"],
             checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint),
             meta_sha256=sha(run/"meta.json"),velocity_rmse_m_s=record["velocity_rmse_m_s"],
             coarse_cell_m=cell,tested_candidates=p["n_candidates"],
             used_candidates=1 if mode=="hard" else p["weights_top_n"],
             seed_separation_m=p["weights_min_distance"]*meta["grid_info"]["fine_cell_size"])
        if section in ("4.5", "4.6"):
            diagnostic = run / f"iter_{complete_index}"
            h_path = diagnostic / "sensitivity_diagonal.npy"
            state_path = diagnostic / "model.npy"
            sensitivity = np.load(h_path, allow_pickle=False)
            if (sensitivity.shape != model.shape or not np.isfinite(sensitivity).all()
                    or np.any(sensitivity < 0) or p["coverage_damping_power"] != 0):
                raise ValueError("Expected valid sensitivity and scalar damping in 4.5/4.6")
            damping = p["lambda_reg"] * sensitivity.mean()
            if not np.isfinite(damping) or damping <= 0:
                raise ValueError("Expected positive scalar damping")
            state = np.load(state_path, allow_pickle=False)
            block.setdefault("sensitivity_ratios", {})[mode] = (sensitivity / damping, cell)
            block["records"][mode]["sensitivity_diagnostic"] = dict(
                iteration=complete_index, damping=float(damping),
                timing="Before last velocity update; not recomputed at final model",
                max_final_relative_change=float(np.max(np.abs(model / state - 1))),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (h_path,state_path)})
    np.testing.assert_array_equal(*initials)
    if section == "4.5":
        if np.any(np.diff(truth, axis=1)):
            raise ValueError("Expected fault model invariant along Y")
        interfaces = np.flatnonzero(np.any(np.diff(truth, axis=0) != 0, axis=(1,2))) + 1
        if len(interfaces) != 1:
            raise ValueError("Expected one vertical fault in the original truth")
        block["fault_geometry"] = dict(
            source="Exact interfaces in original source truth array, not fitted inversion features",
            fault_x_native_m=float(interfaces[0] * truth_cell),
            left_layer_depths_m=((np.flatnonzero(np.diff(truth[0,0]) != 0)+1)*truth_cell).tolist(),
            right_layer_depths_m=((np.flatnonzero(np.diff(truth[-1,0]) != 0)+1)*truth_cell).tolist())
    return block


def sample_section(velocity, cell, extent, axis, section):
    assert 0 <= section < extent[axis]
    horizontal = 1 if axis == 0 else 0
    vertical = 1 if axis == 2 else 2
    n = np.rint(extent/SPACING_M).astype(int)
    np.testing.assert_allclose(n*SPACING_M,extent)
    u = (np.arange(n[horizontal])+.5)*SPACING_M
    z = (np.arange(n[vertical])+.5)*SPACING_M
    # Half-open velocity cells [lower, upper); no trilinear interpolation of velocity.
    iu = np.minimum(np.floor(u/cell).astype(int),velocity.shape[horizontal]-1)
    iz = np.minimum(np.floor(z/cell).astype(int),velocity.shape[vertical]-1)
    fixed = min(int(np.floor(section/cell)),velocity.shape[axis]-1)
    if axis==1:
        values = velocity[iu[None,:],fixed,iz[:,None]]
    elif axis == 0:
        values = velocity[fixed,iu[None,:],iz[:,None]]
    else:
        values = velocity[iu[None,:],iz[:,None],fixed]
    assert values.shape==(len(z),len(u)) and np.isfinite(values).all()
    return values/1000


def prepare_sections(block):
    offset=block["native_to_common_translation_m"]
    extent=block["extent_m"]
    block["section_y_native_m"]=block["section_y_common_m"]-offset[1]
    block["section_x_native_m"]=block["section_x_common_m"]-offset[0]
    block["section_z_native_m"] = 4500. if block["section"] == "4.3" else HORIZONTAL_DEPTH_M
    arrays={}
    for name,(field,cell) in block["fields"].items():
        arrays[name+"_xz"]=sample_section(field,cell,extent,1,block["section_y_native_m"])
        arrays[name+"_yz"]=sample_section(field,cell,extent,0,block["section_x_native_m"])
        arrays[name+"_xy"]=sample_section(field,cell,extent,2,block["section_z_native_m"])
    lo,hi=block["color_limits_km_s"]
    assert all(lo<=a.min()<=a.max()<=hi for a in arrays.values()),(block["section"],[(k,a.min(),a.max()) for k,a in arrays.items()])
    block["slices"]=arrays
    block["slice_ranges_km_s"]={name:[float(a.min()),float(a.max())] for name,a in arrays.items()}
    exports = dict(arrays)
    if block["section"] in ("4.5", "4.6"):
        block["display_masks"], block["native_display_masks"], block["sensitivity_envelopes"] = {}, {}, {}
        cell = block["sensitivity_ratios"]["hard"][1]
        np.testing.assert_allclose(cell, block["sensitivity_ratios"]["soft"][1])
        repeat = round(cell / SPACING_M)
        np.testing.assert_allclose(repeat * SPACING_M, cell)
        for plane, axis, cut in [("xy",2,block["section_z_native_m"]),
                                  ("xz",1,block["section_y_native_m"]),
                                  ("yz",0,block["section_x_native_m"])]:
            scores = {mode: sample_section(ratio,cell,extent,axis,cut)*1000
                      for mode,(ratio,cell) in block["sensitivity_ratios"].items()}
            raw = (scores["hard"] >= THRESHOLD) & (scores["soft"] >= THRESHOLD)
            native_raw = raw[::repeat, ::repeat]
            native_display = expanded_envelope(native_raw, cell, padding_m=PADDING_CELLS * cell)
            display = native_display.repeat(repeat, axis=0).repeat(repeat, axis=1)
            np.testing.assert_equal(display.shape, raw.shape)
            block["display_masks"][plane] = display
            block["native_display_masks"][plane] = native_display
            exports[plane+"_native_raw_mask"] = native_raw
            exports[plane+"_native_display_mask"] = native_display
            exports[plane+"_raw_mask"] = raw
            exports[plane+"_display_mask"] = display
            exports[plane+"_added_mask"] = display & ~raw
            for mode,score in scores.items():
                exports[mode+"_"+plane+"_sensitivity_over_damping"] = score
            block["sensitivity_envelopes"][plane] = dict(
                native_cell_size_m=float(cell), padding_m=float(PADDING_CELLS * cell),
                raw_area_km2=float(raw.sum()*SPACING_M**2/1e6),
                display_area_km2=float(display.sum()*SPACING_M**2/1e6),
                added_area_km2=float((display & ~raw).sum()*SPACING_M**2/1e6))
    np.savez_compressed(HERE/f"section_{block['section'].replace('.','_')}_slices.npz",**exports)


def draw_block(fig,spec,block,letter):
    extent=block["extent_m"]/1000
    offset=block["native_to_common_translation_m"]/1000
    norm=Normalize(*block["color_limits_km_s"])
    box=spec.get_position(fig)
    fw,fh=fig.get_size_inches()
    gap=3.8  # Room for the external slice captions and axis labels.
    row_height=max(extent[1],extent[2])
    scale=min(box.width*.95*fw/(2*extent[0]+extent[1]+2*gap),
              box.height*fh/(3*row_height+2*gap))
    for r,mode in enumerate(["true","soft","hard"]):
        row_top=box.y1-r*(row_height+gap)*scale/fh
        label=mode.capitalize() if mode!="true" else "True"
        if mode!="true":
            label+=f' · {block["records"][mode]["model_update_number"]} updates'
        fig.text(box.x0-.035,row_top-row_height*scale/fh/2,label,
                 ha="center",va="center",rotation=90,fontsize=10)
        for plane,h,v,dx,fixed_axis,value in [
            ("xy",0,1,0,"Z",block["section_z_native_m"]),
            ("yz",1,2,extent[0]+gap,"X",block["section_x_common_m"]),
            ("xz",0,2,extent[0]+extent[1]+2*gap,"Y",block["section_y_common_m"]),
        ]:
            ax=fig.add_axes([box.x0+dx*scale/fw,row_top-extent[v]*scale/fh,
                             extent[h]*scale/fw,extent[v]*scale/fh])
            x0=offset[h];x1=x0+extent[h]
            y0=offset[v] if plane=="xy" else 0
            y1=y0+extent[v]
            ax.imshow(block["slices"][mode+"_"+plane],origin="upper",
                      extent=(x0,x1,y1,y0),cmap="RdBu_r",norm=norm,
                      interpolation="nearest",aspect="equal",rasterized=True)
            if block["section"] in ("4.5", "4.6") and mode in ("hard", "soft"):
                mask = block["display_masks"][plane]
                ax.imshow(np.ma.masked_where(mask, np.ones(mask.shape)),
                          origin="upper", extent=(x0,x1,y1,y0),
                          cmap=ListedColormap(["white"]), vmin=0,vmax=1,
                          interpolation="nearest",alpha=.82,aspect="equal",rasterized=True)
                native_mask = block["native_display_masks"][plane]
                cell_km = block["records"]["hard"]["coarse_cell_m"] / 1000
                segments = cell_boundary_segments(native_mask, cell_km, origin=(x0,y0))
                for color,width in [("white",1.6),("black",.65)]:
                    ax.add_collection(LineCollection(segments,colors=color,linewidths=width,zorder=2))
            if block["section"] in ("4.2", "4.3", "4.4") and mode in ("hard", "soft"):
                truth = block["slices"]["true_" + plane]
                # Original checkerboard interfaces, not inversion-cell boundaries.
                columns = np.flatnonzero(np.any(np.diff(truth, axis=1) != 0, axis=0)) + 1
                rows = np.flatnonzero(np.any(np.diff(truth, axis=0) != 0, axis=1)) + 1
                for index in columns:
                    ax.axvline(x0 + index * SPACING_M / 1000,
                               color="black", linewidth=.45, alpha=.8, zorder=2)
                for index in rows:
                    ax.axhline(y0 + index * SPACING_M / 1000,
                               color="black", linewidth=.45, alpha=.8, zorder=2)
            if block["section"] == "4.5":
                geometry = block["fault_geometry"]
                fault_x = geometry["fault_x_native_m"] / 1000 + offset[0]
                if plane in ("xy", "xz"):
                    ax.axvline(fault_x,color="black",linewidth=.85,zorder=4)
                if plane == "xz":
                    for side,left,right in [("left",x0,fault_x),("right",fault_x,x1)]:
                        for depth in geometry[side+"_layer_depths_m"]:
                            ax.plot([left,right],[depth/1000]*2,color="black",
                                    linewidth=.7,linestyle="--",zorder=4)
                elif plane == "yz":
                    side = "left" if block["section_x_native_m"] < geometry["fault_x_native_m"] else "right"
                    for depth in geometry[side+"_layer_depths_m"]:
                        ax.axhline(depth/1000,color="black",linewidth=.7,linestyle="--",zorder=4)
            ax.set_xlim(x0,x1)
            ax.set_ylim((y0,y1) if plane=="xy" else (y1,y0))
            ax.xaxis.set_major_locator(MultipleLocator(4 if extent[h]>=16 else 2))
            ax.yaxis.set_major_locator(MultipleLocator(4 if extent[v]>=10 else 2))
            ax.tick_params(labelsize=7,length=2,pad=1)
            ax.set_xlabel("X (km)" if h==0 else "Y (km)",fontsize=8,labelpad=1)
            ax.set_ylabel("Y (km)" if plane=="xy" else "Depth (km)",fontsize=8,labelpad=1)
            ax.annotate(f"{plane.upper()} | {fixed_axis} = {value/1000:g} km",
                        xy=(0,1),xycoords="axes fraction",xytext=(0,3),
                        textcoords="offset points",ha="left",va="bottom",
                        fontsize=6.5,annotation_clip=False)
    content_width=(2*extent[0]+extent[1]+2*gap)*scale/fw
    content_height=(3*row_height+2*gap)*scale/fh
    bar_height=content_height*.5
    color_ax=fig.add_axes([box.x0+content_width+.012,
                           box.y1-content_height/2-bar_height/2,.008,bar_height])
    bar=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap="RdBu_r"),cax=color_ax)
    bar.set_label("Velocity (km/s)",fontsize=9,labelpad=4)
    bar.ax.tick_params(labelsize=8,width=.5,length=3)
    detail=" · nominal pick noise" if block["section"]=="4.4" else " · truth-informed 1-D start" if block["section"]=="4.6" else ""
    fig.text(box.x0,box.y1+.014,f"({letter})  §{block['section']}  {block['title']}{detail}",
             fontsize=11,weight="bold",ha="left",va="bottom")


def main():
    assert PNG.is_dir()
    metrics_path=ROOT/"paper/figures/article_comparison_metrics.json"
    metrics=read(metrics_path)
    specifications=[
      ("4.2","Checkerboard: one-sided stations","onesided","section2_pair01",6000,12000,[0,0,0],[4.6,5.4]),
      ("4.3","Checkerboard: two station clusters","two_clusters","section3_pair01",3500,6500,[0,0,0],[4.6,5.4]),
      ("4.4","Checkerboard: Geysers geometry","real_geometry_noise","section4_pair07",3500,10000,[1000,0,0],[4.6,5.4]),
      ("4.5","Vertical fault: Geysers geometry","fault_retention","section5_topnsweep_k128_top50",3500,10000,[0,0,0],[3.4,5.8]),
    ]
    blocks=[]
    for section,title,table,pair,y,x,offset,limits in specifications:
        record=next(r for r in metrics["tables"][table] if r["pair_id"]==pair)
        blocks.append(load_block(section,title,pair,record["hard"]["experiment_id"],record,y,x,offset,limits))
    records=next(r for r in metrics["tables"]["heterogeneous"] if r["pair_id"]=="section6_pair01")
    blocks.append(load_block("4.6","Heterogeneous model: Geysers geometry","section6_pair01",
                 records["hard"]["experiment_id"],records,3500,10000,[1000,0,0],[2.8,7.2]))
    fault_source=Path(blocks[3]["source_directory"])
    for block in blocks:
        if block["section"] in ["4.4","4.5","4.6"]:
            block["coordinate_registration"]=verify_registration(Path(block["source_directory"]),fault_source,
                                                               block["native_to_common_translation_m"])
        prepare_sections(block)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                         "savefig.facecolor":"white","axes.linewidth":.5})
    height_ratios=[(3*max(b["extent_m"][1:])+7600)/
                   (2*b["extent_m"][0]+b["extent_m"][1]+7600) for b in blocks]
    fig=plt.figure(figsize=(17,35))
    grid=fig.add_gridspec(5,1,left=.075,right=.95,top=.94,bottom=.025,
                          hspace=.20,height_ratios=height_ratios)
    fig.suptitle("Synthetic tomography · rows: True / Soft / Hard · columns: XY / YZ / XZ",fontsize=16,y=.992)
    fig.text(.5,.973,"XY + two vertical planes per series · Z = 1.6 km (4.3: 4.5 km) · native cells · no smoothing",
             ha="center",fontsize=10,color="#555555")
    fig.text(.5,.958,"Geysers panels share physical planes Y = 3.5 km and X = 10 km in the fault-test frame",
             ha="center",fontsize=9,color="#555555")
    for i,block in enumerate(blocks):draw_block(fig,grid[i,0],block,chr(97+i))
    fig.text(.5,.009,"Color scale shared within each block; the three checkerboards share one range. "
             "4.5/4.6: sensitivity envelopes, not spatial resolution. 4.5 solid: source fault; dashed: source layers. Depth positive downward.",
             ha="center",fontsize=8,color="#555555")
    fig.savefig(PNG/"synthetic_tomography_comparison.png",dpi=200,bbox_inches="tight",pad_inches=.12)
    fig.savefig(PNG/"synthetic_tomography_comparison_three_planes.png",dpi=200,bbox_inches="tight",pad_inches=.12)
    fig.savefig(HERE/"synthetic_tomography_comparison.pdf",dpi=200,bbox_inches="tight",pad_inches=.12)
    plt.close(fig)
    for i,block in enumerate(blocks):
        fig=plt.figure(figsize=(17,10))
        grid=fig.add_gridspec(1,1,left=.075,right=.95,top=.87,bottom=.14)
        draw_block(fig,grid[0,0],block,chr(97+i))
        if block["section"] in ["4.4","4.5","4.6"]:
            fig.text(.5,.035,"Common physical planes across §4.4–4.6; X coordinates are expressed in the fault-test frame.",
                     ha="center",fontsize=8,color="#555555")
        if block["section"] == "4.5":
            fig.text(.5,.095,"Solid black: source fault at X = 6.5 km; dashed black: source layer boundaries on each side. Overlaid on True / Soft / Hard.",
                     ha="center",fontsize=8,color="#555555")
        if block["section"] in ("4.5", "4.6"):
            fig.text(.5,.065,"Outline: shared Hjj/α ≥ 1 envelope + one native cell; holes filled. "
                     "Added area is display-only, NOT spatial resolution; diagnostics precede final update.",
                     ha="center",fontsize=8,color="#555555")
        fig.savefig(PNG/f"section_{block['section'].replace('.','_')}.png",dpi=200,bbox_inches="tight",pad_inches=.12)
        fig.savefig(HERE/f"section_{block['section'].replace('.','_')}.pdf",dpi=200,bbox_inches="tight",pad_inches=.12)
        plt.close(fig)
    report=dict(
        selection="Previously agreed baseline pairs; 4.2--4.4 retain selected V_3, while 4.5--4.6 use final_model.npy.",
        layout="Five blocks, each with True / Soft / Hard rows (top to bottom) and horizontal XY / transverse YZ / longitudinal XZ columns (left to right); equal metric scale.",
                horizontal_depth_m={b["section"]: b["section_z_native_m"] for b in blocks},
                checkerboard_overlay="4.2–4.4 Hard/Soft only, all three planes: true checkerboard interfaces, black 0.45 pt lines, alpha 0.8",
        sampling="25-m pixel centers; exact native piecewise-constant velocity lookup; floor index selects the positive-side cell on exact boundaries. No smoothing, velocity interpolation, averaging to inversion cells, or oracle substitution.",
        physical_planes="Geysers uses the fault-test Cartesian frame: common X=10 km and Y=3.5 km. Native-to-common X translation is +1 km for 4.4/4.6 and zero for 4.5. Thus native cut X=9 km for 4.4/4.6 and X=10 km for 4.5. Full native domains are retained.",
        caveats=["Fault uses 12 final updates in both modes. Heterogeneous 1-D runs end after 9 updates in both modes.",
                 "Heterogeneous pair uses the supplied truth-derived 1-D starting model; favorable initialization, not a blind starting model.",
                 "A colorbar is shared by all three columns and all three planes within a block. The checkerboards share 4.6–5.4 km/s. The heterogeneous scale is 2.8–7.2 km/s. No displayed values clipped.",
                 "No forward solves, inversions, figure insertion, article source edits or published PDF updates."],
        blocks=[{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in b.items()
                 if key not in ["fields","slices","sensitivity_ratios","display_masks","native_display_masks"]} for b in blocks],
        sensitivity_envelope_policy=dict(sections=["4.5","4.6"],threshold=THRESHOLD,padding_native_cells=PADDING_CELLS,
             raw="Hjj/alpha >= threshold in both methods",
             display="Convex hull of native selected cell centres + one-cell Euclidean expansion + hole filling, repeated without interpolation; boundary follows native cell edges",
             interpretation="Display envelope only, not validated spatial resolution; added pixels may fail threshold",
             helper_sha256=sha(ROOT/"paper/paper_scripts/sensitivity_envelope.py")),
        source_metrics_sha256={str(metrics_path.relative_to(ROOT)):sha(metrics_path)},
        script_sha256=sha(Path(__file__)))
    (HERE/"selection_and_sections.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    assert all(p.suffix==".png" for p in PNG.iterdir())
    for block in blocks:
        print(block["section"],block["pair_id"],"native sections Y/X (km)",
              block["section_y_native_m"]/1000,block["section_x_native_m"]/1000,
              "updates",[block["records"][m]["model_update_number"] for m in ["hard","soft"]])
    print("Saved five-block comparison and five individual previews to",PNG)


if __name__=="__main__":
    main()
