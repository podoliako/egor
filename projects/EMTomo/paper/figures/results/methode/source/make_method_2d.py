"""2-D analogue of the EMTomo forward grid and gradient-based ray tracing.

Run with python -B /path/to/make_method_2d.py. PNG only in ../png;
PDF, computed arrays and metadata stay next to this script. No experiment is run.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import numpy as np
from scipy.ndimage import map_coordinates
import skfmm

HERE = Path(__file__).resolve().parent
PNG = HERE.parent / "png"
H = 1.0  # km
Q = 8
DX = H / Q
STATION = np.array([1.35, .30])  # exact continuous X, depth in km
CANDIDATES = np.array([[6.55, 4.85], [7.20, 3.85], [5.15, 5.35]])
RAY_COLORS = ["#b44730", "#256ca0", "#8a639c"]
STATION_COLOR = "#df9427"


def sample(array, point):
    index = point / DX - .5
    return float(map_coordinates(array, index[:, None], order=1, mode="nearest")[0])


def time_field(velocity):
    x = (np.arange(velocity.shape[0]) + .5) * DX
    z = (np.arange(velocity.shape[1]) + .5) * DX
    distance = np.hypot(x[:, None]-STATION[0], z[None, :]-STATION[1])
    radius = 3 * DX
    local_velocity = velocity[tuple(np.floor(STATION/DX).astype(int))]
    inside = distance <= radius
    seed_velocity = velocity.copy()
    seed_velocity[inside] = local_velocity
    times = np.asarray(skfmm.travel_time(distance-radius, seed_velocity, dx=DX, order=2))
    times += radius/local_velocity
    times[inside] = distance[inside]/local_velocity
    assert np.isfinite(times).all() and np.all(times >= 0)
    return x, z, times, radius


def trace(times, start):
    gradients = np.gradient(times, DX, edge_order=2)
    lower = np.full(2, DX/2)
    upper = np.array(times.shape)*DX-DX/2
    point = start.copy()
    path = [point.copy()]
    step = DX*.20
    for _ in range(5000):
        if np.linalg.norm(point-STATION) <= DX*.6:
            path.append(STATION.copy())
            return np.array(path)
        gradient = np.array([sample(g, point) for g in gradients])
        norm = np.linalg.norm(gradient)
        assert norm > 1e-12
        next_point = np.clip(point-step*gradient/norm, lower, upper)
        assert np.linalg.norm(next_point-point) > 1e-10, "Ray stalled"
        assert sample(times,next_point) <= sample(times,point)+2e-4, "Traveltime increased"
        path.append(next_point.copy())
        point = next_point
    raise RuntimeError("Ray failed to reach station")


def coarse_lengths(path, shape):
    """Split every polyline segment at exact coarse-cell faces; lengths in km."""
    lengths = np.zeros(shape)
    pieces, cells = [], []
    for a, b in zip(path[:-1], path[1:]):
        delta = b-a
        cuts = [0., 1.]
        for axis in range(2):
            if abs(delta[axis]) > 1e-14:
                faces = np.arange(1,shape[axis])*H
                cuts.extend(t for t in (faces-a[axis])/delta[axis] if 0 < t < 1)
        cuts = np.unique(cuts)
        for lo, hi in zip(cuts[:-1],cuts[1:]):
            left, right = a+lo*delta, a+hi*delta
            cell = np.minimum(np.floor((left+right)/(2*H)).astype(int), np.array(shape)-1)
            lengths[tuple(cell)] += np.linalg.norm(right-left)
            pieces.append([left,right])
            cells.append(tuple(cell))
    np.testing.assert_allclose(lengths.sum(), np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    return lengths, np.array(pieces), cells


def grid(ax, fine=False):
    if fine:
        for x in np.arange(DX,8,DX):
            ax.axvline(x,color="#ffffff",lw=.22,alpha=.58,zorder=2)
        for z in np.arange(DX,6,DX):
            ax.axhline(z,color="#ffffff",lw=.22,alpha=.58,zorder=2)
    for x in np.arange(0,8+H,H):
        ax.axvline(x,color="#343b42",lw=.65,alpha=.7,zorder=3)
    for z in np.arange(0,6+H,H):
        ax.axhline(z,color="#343b42",lw=.65,alpha=.7,zorder=3)


def points(ax, candidates=True, station_label_color="#4c391b"):
    ax.scatter(*STATION,s=110,marker="^",c=STATION_COLOR,edgecolors="#342b20",lw=.7,zorder=9)
    ax.annotate("Station",STATION,xytext=(8,-17),textcoords="offset points",fontsize=9,
                color=station_label_color,zorder=10)
    if candidates:
        for i, point in enumerate(CANDIDATES):
            ax.scatter(*point,s=90,marker="*",facecolors="white",edgecolors=RAY_COLORS[i],lw=1,zorder=9)
            ax.annotate(f"C{i+1}",point,xytext=(7,5),textcoords="offset points",fontsize=9,
                        color=RAY_COLORS[i],weight="bold",zorder=10)


def main():
    assert PNG.is_dir()
    coarse = np.full((8,6),4.5)
    coarse[:,:2] = 4.0
    coarse[3:5,2:4] = 2.7
    coarse[5:,2:] = 5.8
    fine = coarse.repeat(Q,axis=0).repeat(Q,axis=1)
    x,z,times,radius = time_field(fine)
    paths = [trace(times,start) for start in CANDIDATES]
    lengths = []
    ray_geometry = []
    for path in paths:
        np.testing.assert_array_equal(path[-1],STATION)
        result = coarse_lengths(path,coarse.shape)
        lengths.append(result[0])
        ray_geometry.append(result)
    palette = ListedColormap(["#e9c8bc","#e1e8ef","#bccbd7","#aacbb5"])
    norm = BoundaryNorm([2,3.3,4.25,5.1,6.5],palette.N)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                         "pdf.fonttype":42,"savefig.facecolor":"white"})
    fig = plt.figure(figsize=(15.4,6.5))
    gs = fig.add_gridspec(1,3,left=.055,right=.985,bottom=.29,top=.82,wspace=.25)
    axes = [fig.add_subplot(gs[0,i]) for i in range(3)]
    for ax in axes:
        ax.set(xlim=(0,8),ylim=(6,0),xlabel="X (km)",ylabel="Depth (km)",
               xticks=np.arange(0,9,2),yticks=np.arange(0,7,1))
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9,length=3)
    axes[0].imshow(coarse.T,extent=(0,8,6,0),origin="upper",cmap=palette,norm=norm,interpolation="nearest")
    grid(axes[0],fine=True)
    axes[0].add_patch(Rectangle((3,2),1,1,fill=False,ec="#b44730",lw=2,zorder=5))
    axes[0].text(3.5,2.5,r"$v_\ell$",ha="center",va="center",fontsize=15,zorder=6,
                 bbox=dict(facecolor="white",edgecolor="none",alpha=.85,pad=1.5))
    axes[0].annotate("",xy=(2,5.65),xytext=(1,5.65),
                     arrowprops=dict(arrowstyle="|-|",color="#222222",lw=1),zorder=6)
    axes[0].text(1.5,5.5,r"$h$",ha="center",fontsize=12)
    axes[0].annotate("",xy=(6+DX,1.55),xytext=(6,1.55),
                     arrowprops=dict(arrowstyle="|-|",color="#222222",lw=1),zorder=6)
    axes[0].annotate(r"$h/q$",xy=(6+DX/2,1.55),xytext=(6.45,1.25),
                     arrowprops=dict(arrowstyle="-",color="#222222",lw=.7),fontsize=11)
    points(axes[0],candidates=False)
    image = axes[1].imshow(times.T,extent=(0,8,6,0),origin="upper",cmap="viridis",interpolation="nearest")
    contours = axes[1].contour(x,z,times.T,levels=np.arange(.2,2.5,.2),colors="white",linewidths=.65,alpha=.8)
    axes[1].clabel(contours,fmt="%.1f",fontsize=7,inline=True)
    axes[1].add_patch(Circle(STATION,radius,fill=False,ec="white",lw=1.2,ls="--",zorder=6))
    points(axes[1], station_label_color="white")
    axes[2].imshow(coarse.T,extent=(0,8,6,0),origin="upper",cmap=palette,norm=norm,alpha=.48,interpolation="nearest")
    grid(axes[2])
    axes[2].contour(x,z,times.T,levels=np.arange(.2,2.5,.2),colors="#73818a",linewidths=.45,alpha=.45)
    for i,path in enumerate(paths):
        axes[2].plot(*path.T,c=RAY_COLORS[i],lw=2.1,zorder=5)
        for fraction in (.28,.66):
            j=int((len(path)-5)*fraction)
            axes[2].annotate("",xy=path[j+5],xytext=path[j],
                             arrowprops=dict(arrowstyle="->",lw=1.6,color=RAY_COLORS[i],mutation_scale=12),zorder=7)
    selected_length,pieces,cells = ray_geometry[0]
    selected_cell=max(cells,key=lambda cell:selected_length[cell] if 1<cell[0]<6 and cell[1]>1 else -1)
    highlighted=np.array([piece for piece,cell in zip(pieces,cells) if cell==selected_cell])
    axes[2].add_collection(LineCollection(highlighted,colors="#f3af27",linewidths=4.2,zorder=6))
    axes[2].add_patch(Rectangle(np.array(selected_cell)*H,H,H,fill=False,ec="#9e6c17",lw=1.5,zorder=7))
    midpoint=highlighted.reshape(-1,2).mean(axis=0)
    axes[2].annotate(r"$G_{en\ell}^{(j)}$: length in cell $\ell$",xy=midpoint,xytext=(3.9,5.75),
                     fontsize=8,color="#704b0b",arrowprops=dict(arrowstyle="->",color="#704b0b",lw=.8),
                     bbox=dict(facecolor="white",alpha=.9,edgecolor="none",pad=2),zorder=10)
    points(axes[2])
    for ax,title in zip(axes,["(a)  Velocity cells and computational subcells",
                              "(b)  One station travel-time field",
                              "(c)  Rays from candidate locations"]):
        ax.set_title(title,fontsize=11,pad=13,weight="bold")
    notes=["One velocity per coarse cell\nFine subcells inherit it; no new parameters",
           "Fast marching from the exact station\nOne field is reused for all candidates",
           r"Trace along $-\nabla T_n$ toward the station"+"\nAccumulate path lengths in velocity cells"]
    for ax,note in zip(axes,notes):
        ax.text(.5,-.20,note,transform=ax.transAxes,ha="center",va="top",fontsize=9,linespacing=1.5)
    bar=fig.add_axes([.399,.125,.205,.017])
    fig.colorbar(image,cax=bar,orientation="horizontal",label="Travel time (s)")
    bar.tick_params(labelsize=8,length=2)
    fig.suptitle("From velocity cells to station fields and ray paths",fontsize=16,y=.965)
    fig.text(.5,.905,"2-D illustration · same piecewise-constant velocity model in all panels · station and candidates are off-grid",
             ha="center",fontsize=10,color="#555555")
    handles=[Line2D([],[],marker="^",ls="",color="#342b20",markerfacecolor=STATION_COLOR,markersize=8,label="Station"),
             Line2D([],[],marker="*",ls="",color="#52687b",markerfacecolor="white",markersize=10,label="Candidate locations"),
             Line2D([],[],color="#343b42",lw=1,label="Velocity-cell boundaries"),
             Line2D([],[],color="#f3af27",lw=3,label="Ray segment assigned to one cell")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.025),ncol=4,frameon=False,fontsize=9)
    fig.savefig(PNG/"method_2d.png",dpi=220)
    fig.savefig(HERE/"method_2d.pdf",dpi=220)
    plt.close(fig)
    arrays=dict(coarse_velocity_km_s=coarse,fine_velocity_km_s=fine,travel_time_s=times,
                station_km=STATION,candidates_km=CANDIDATES,coarse_ray_lengths_km=np.array(lengths))
    arrays.update({f"ray_{i+1}_km":path for i,path in enumerate(paths)})
    np.savez_compressed(HERE/"method_2d_data.npz",**arrays)
    metadata=dict(purpose="2-D numerical illustration, not an experimental result or production 3-D run",
                  coarse_shape=list(coarse.shape),coarse_cell_km=H,subdivision=Q,fine_cell_km=DX,
                  solver="scikit-fmm, second order; 2-D analogue of eikonal.py homogeneous source sphere (circle here)",
                  source_radius_km=radius,station_km=STATION.tolist(),candidates_km=CANDIDATES.tolist(),
                  tracing="Bilinearly interpolated gradient, normalized negative-gradient steps of 0.2 fine cells; all three rays reached the exact station",
                  length_accumulation="Each traced polyline segment split at exact coarse-cell faces; sum of cell lengths checked against full path length",
                  ray_lengths_km=[float(a.sum()) for a in lengths],
                  ray_linear_times_s=[float(np.sum(a/coarse)) for a in lengths],
                  candidate_interpolated_times_s=[sample(times,p) for p in CANDIDATES],
                  illustration_parameters="q, velocities, station and candidate positions chosen only for this schematic; not paper experiment settings",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"method_2d.json").write_text(json.dumps(metadata,indent=2)+"\n")
    assert all(p.suffix==".png" for p in PNG.iterdir())
    print(json.dumps(metadata,indent=2))
    print("Saved",PNG/"method_2d.png")


if __name__=="__main__":
    main()
