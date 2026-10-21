# Read-only station-derived voxel terrain sketch, not a DEM or wave solver.
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from projects.forward_modeling.experiments import load_inputs
HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO / "projects/EMTomo/runs/_geysers2011_queue/manifest.json"


def interpolate_surface(station_xy, elevations, x_centers, y_centers):
    # Linear interpolation inside the station hull, nearest outside: no DEM.
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    surface = LinearNDInterpolator(station_xy, elevations)(xx, yy)
    outside = ~np.isfinite(surface)
    nearest = NearestNDInterpolator(station_xy, elevations)(xx, yy)
    return np.where(outside, nearest, surface), outside


def terrain_mask(surface, top_m, cell_m, nz):
    # Round the upper rock boundary up to a voxel face, by less than one cell.
    surface = np.asarray(surface, dtype=float)
    if surface.ndim != 2 or not np.all(np.isfinite(surface)):
        raise ValueError("Expected finite 2D surface")
    if not np.isfinite(top_m) or not np.isfinite(cell_m) or cell_m <= 0 or nz < 1:
        raise ValueError("Invalid grid")
    if np.any(surface > top_m):
        raise ValueError("Surface exceeds model top")
    first = np.clip(np.floor((top_m - surface) / cell_m).astype(int), 0, nz)
    mask = np.arange(nz)[None, None, :] >= first[:, :, None]
    return mask, top_m - first * cell_m


def exposed_faces(solid, velocity, cell_m, top_m):
    # Generate solid faces only; no air cubes and no internal solid-solid faces.
    if solid.shape != velocity.shape or solid.dtype != bool:
        raise ValueError("Boolean solid mask must match velocity shape")
    vertices, values, lighting = [], [], []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for side in (0, 1):
            neighbor = np.zeros_like(solid)
            target, source = [slice(None)] * 3, [slice(None)] * 3
            target[axis] = slice(1, None) if side == 0 else slice(None, -1)
            source[axis] = slice(None, -1) if side == 0 else slice(1, None)
            neighbor[tuple(target)] = solid[tuple(source)]
            indices = np.argwhere(solid & ~neighbor)
            offsets = np.zeros((4, 3))
            offsets[:, axis] = side
            offsets[:, others[0]] = (0, 1, 1, 0)
            offsets[:, others[1]] = (0, 0, 1, 1)
            faces = (indices[:, None, :] + offsets[None, :, :]) * (cell_m / 1000.)
            faces[:, :, 2] = top_m / 1000. - faces[:, :, 2]
            vertices.append(faces)
            values.append(velocity[tuple(indices.T)])
            lighting.append(np.full(len(indices), ((.73, .86), (.86, .72), (1., .58))[axis][side]))
    return np.concatenate(vertices), np.concatenate(values), np.concatenate(lighting)


def add_voxels(ax, mask, velocity, cell_m, top_m, cmap, norm):
    vertices, values, lighting = exposed_faces(mask, velocity, cell_m, top_m)
    colors = cmap(norm(values))
    colors[:, :3] *= lighting[:, None]
    ax.add_collection3d(Poly3DCollection(vertices, facecolors=colors,
        edgecolors=(.12, .16, .20, .14), linewidths=.14, zorder=1))
    return len(vertices)


def preview(manifest_path, output_dir):
    manifest = json.loads(Path(manifest_path).read_text())
    root, experiment = Path(manifest["experiments_root"]), manifest["experiment_id"]
    directory = root / "input" / experiment
    model, stations, events = load_inputs(root, experiment)
    generation = json.loads((directory / "generation.json").read_text())
    metadata = {r["station_id"]: r for r in generation["station_metadata"]}
    heights = np.array([metadata[name]["elevation_m"] for name in stations.ids])
    xy = stations.coordinates_m[:, :2]
    top = float(heights.max())
    dx = model.cell_size_m
    nx, ny, nz = model.velocity.shape
    surface, extrapolated = interpolate_surface(xy, heights, (np.arange(nx)+.5)*dx, (np.arange(ny)+.5)*dx)
    rock, stair_top = terrain_mask(surface, top, dx, nz)
    velocity_sha = hashlib.sha256(model.velocity.tobytes()).hexdigest()
    ij = np.floor(xy / dx).astype(int)
    burial = stair_top[ij[:, 0], ij[:, 1]] - heights
    if np.any(burial < 0):
        raise ValueError("A station is in proxy air; do not silently snap it")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("RdYlBu_r")
    norm = Normalize(float(model.velocity.min()), float(model.velocity.max()))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "white"})
    note = ("ЭСКИЗ, НЕ DEM: линейная интерполяция высот 26 станций; вне их выпуклой оболочки — ближайшая станция.\n"
        "Верх сетки h=1166 м. Верх породы округлён вверх до грани кубика 125 м. Цвета — прежний массив true Vp.\n"
        "Только визуализация: входные данные не изменены. Непроницаемая маска воздуха в солверах пока НЕ реализована.")
    fig = plt.figure(figsize=(19, 10))
    full = fig.add_subplot(121, projection="3d", computed_zorder=False)
    zoom = fig.add_subplot(122, projection="3d", computed_zorder=False)
    face_count = add_voxels(full, rock, model.velocity, dx, top, cmap, norm)
    upper = min(nz, 12)
    add_voxels(zoom, rock[:, :, :upper], model.velocity[:, :, :upper], dx, top, cmap, norm)
    for ax, depth, scale in ((full, nz*dx, 1.3), (zoom, upper*dx, 3.)):
        ax.scatter(xy[:,0]/1000., xy[:,1]/1000., heights/1000., marker="^", s=35,
            color="#e63228", edgecolors="white", linewidths=.4, depthshade=False,
            zorder=4, label="Станции: реальные XYZ")
        ax.set(xlim=(0,nx*dx/1000.), ylim=(0,ny*dx/1000.),
            zlim=((top-depth)/1000.,top/1000.+.5), xlabel="X, км", ylabel="Y, км", zlabel="Высота h, км")
        ax.set_box_aspect((nx*dx,ny*dx,(depth+500)*scale))
        ax.view_init(elev=28, azim=-60)
        ax.legend(loc="upper left", fontsize=9)
    full.set_title("True-модель: ячейки воздуха убраны\nВертикальный масштаб ×1.3", pad=16)
    zoom.set_title("Верхние 1.5 км: ступеньки по 125 м\nВертикальный масштаб ×3", pad=16)
    zoom.text(1.,6.,top/1000.+.25,"ВОЗДУХ: ЯЧЕЕК НЕТ", color="#566573",fontsize=11,zorder=5)
    fig.suptitle("Ступенчатый рельеф на true-модели · воздух полностью прозрачный\n"
        "ЭСКИЗ по высотам станций, не настоящий DEM", fontsize=19,y=.96)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(left=.015,right=.98,top=.84,bottom=.2,wspace=.03)
    fig.colorbar(scalar, ax=[full,zoom],location="bottom",fraction=.025,pad=.05,
        shrink=.6,label="Vp породы, м/с; воздух исключён маской и не имеет скорости в этом эскизе")
    fig.text(.035,.035,note,fontsize=10,color="#444444")
    fig.savefig(output / "topography_voxels.png",dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(18,8.5),sharey=True,gridspec_kw={"width_ratios":(20,7)})
    event = events.coordinates_m / 1000.
    for axis, ax in enumerate(axes):
        if axis == 0:
            fixed=ny//2
            values,solid,profile=model.velocity[:,fixed,:].T,rock[:,fixed,:].T,stair_top[:,fixed]
            width=nx*dx/1000.
            title=f"X-разрез при Y={(fixed+.5)*dx/1000.:.3f} км"
        else:
            fixed=nx//2
            values,solid,profile=model.velocity[fixed,:,:].T,rock[fixed,:,:].T,stair_top[fixed,:]
            width=ny*dx/1000.
            title=f"Y-разрез при X={(fixed+.5)*dx/1000.:.3f} км"
        ax.imshow(np.ma.array(values,mask=~solid),origin="upper",
            extent=(0,width,(top-nz*dx)/1000.,top/1000.), cmap=cmap,norm=norm,interpolation="nearest",aspect="equal")
        ax.stairs(profile/1000.,np.arange(len(profile)+1)*dx/1000.,baseline=None,
            color="#17202a",lw=1.4,label="Ступенчатая граница порода–воздух")
        ax.scatter(event[:,axis],-event[:,2],s=2.,c="#17202a",alpha=.20,
            label="Все события: проекция",rasterized=True)
        ax.scatter(xy[:,axis]/1000.,heights/1000.,marker="^",s=40,color="#e63228",
            edgecolors="white",linewidths=.4,label="Станции: проекция")
        label = ("X", "Y")[axis]
        ax.set(xlim=(0,width),ylim=(-7,1.8),xlabel=f"{label}, км",title=title)
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Высота h, км; события h=−каталожная глубина")
    axes[0].text(.3,1.45,"ВОЗДУХ ПРОЗРАЧНЫЙ; ПРОЕКТНАЯ МАСКА НЕ ПРОПУСКАЕТ ВОЛНЫ",color="#566573",fontsize=10)
    axes[0].legend(loc="lower left",fontsize=9)
    fig.suptitle("Та же true-модель в разрезах: кубики выше поверхности убраны\n"
        "Белое над рельефом — воздух, а не низкоскоростной слой",fontsize=18,y=.97)
    fig.subplots_adjust(left=.06,right=.98,top=.82,bottom=.24,wspace=.1)
    fig.colorbar(scalar,ax=list(axes),location="bottom",fraction=.035,pad=.1,shrink=.5,label="Vp породы, м/с")
    fig.text(.035,.03,note+"\nСтанции/события спроецированы со всех Y/X: их положение относительно линии этого разреза не является проверкой маски.",
        fontsize=9,color="#444444")
    fig.savefig(output / "topography_voxels_sections.png",dpi=160)
    plt.close(fig)
    assert hashlib.sha256(model.velocity.tobytes()).hexdigest() == velocity_sha
    summary={"preview_only":True,"solver_changed":False,"new_solver_run":False,
        "grid_shape":list(model.velocity.shape),"cell_size_m":dx,"top_elevation_m":top,
        "surface_source":"station heights; linear inside hull, nearest outside; NOT DEM",
        "extrapolated_columns":int(extrapolated.sum()),"total_columns":nx*ny,
        "rock_voxels":int(rock.sum()),"air_voxels":int((~rock).sum()),
        "rendered_faces":face_count,"velocity_array_sha256":velocity_sha,
        "station_burial_below_stair_surface_min_m":float(burial.min()),
        "station_burial_below_stair_surface_max_m":float(burial.max()),
        "air":"excluded mask, not a zero/low-speed medium; visualization only",
        "vertical_datum":"station h=elevation, event h=-catalog depth; common datum assumed, not verified",
        "events_below_7km_grid":int(np.sum(events.coordinates_m[:,2]+top>nz*dx))}
    (output / "topography_voxels.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description="Read-only voxel terrain sketch; no propagation")
    parser.add_argument("--manifest",type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir",type=Path,default=HERE)
    args=parser.parse_args()
    print(json.dumps(preview(args.manifest,args.output_dir),indent=2))


if __name__ == "__main__":
    main()
