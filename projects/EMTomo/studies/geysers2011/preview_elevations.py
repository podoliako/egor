"""Read-only geometry preview with actual station elevations; no travel-time solves.

Station height is elevation_m; event height is -catalogue_depth_m. Their common
vertical datum is an explicit preview assumption, not a verified datum conversion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from projects.forward_modeling.experiments import load_inputs

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = REPO / "projects/EMTomo/runs/_geysers2011_queue/manifest.json"


def relative_depths(elevations_m, catalog_depths_m, top_elevation_m):
    """Use a common positive-down coordinate without changing pair separations."""
    return (top_elevation_m - np.asarray(elevations_m, dtype=float),
            top_elevation_m + np.asarray(catalog_depths_m, dtype=float))


def preview(manifest_path, output_dir, *, top_at_highest=False):
    manifest = json.loads(Path(manifest_path).read_text())
    root, experiment = Path(manifest["experiments_root"]), manifest["experiment_id"]
    model, stations, events = load_inputs(root, experiment)
    generation = json.loads((root / "input" / experiment / "generation.json").read_text())
    metadata = {row["station_id"]: row for row in generation["station_metadata"]}
    heights = np.asarray([metadata[name]["elevation_m"] for name in stations.ids]) / 1000.
    if not np.all(np.isfinite(heights)):
        raise ValueError("Nonfinite station elevation")
    station = stations.coordinates_m / 1000.
    event = events.coordinates_m / 1000.
    top = float(heights.max()) if top_at_highest else 0.
    bottom = top - model.velocity.shape[2] * model.cell_size_m / 1000.
    station_depths, event_depths = relative_depths(heights * 1000., event[:, 2] * 1000., top * 1000.)
    outside = event_depths > model.velocity.shape[2] * model.cell_size_m
    suffix = "_top_at_highest" if top_at_highest else ""
    top_label = f"Верх модели: h={top:.3f} км" if top_at_highest else "Прежний верх модели: h=0"
    if not np.array_equal(station[:, :2], np.asarray([
        [metadata[name]["x_m"], metadata[name]["y_m"]] for name in stations.ids
    ]) / 1000.):
        raise ValueError("Station provenance XY mismatch")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "figure.facecolor": "white"})
    station_color = "#c53222"
    event_color = "#102a43"
    top_color = "#247a96"
    note = ("Высота события = −каталожная глубина; общий вертикальный датум пока предполагается. "
            "DEM/рельеф между станциями не построен.\n"
            "Только визуализация: входные файлы не изменены; новый расчёт не запущен.")
    if top_at_highest:
        note += ("\nМодель целиком поднята на 1166 м: массивы скоростей и глубины слоёв от верха прежние. "
                 "Плоская верхняя граница, не реальный рельеф.")

    fig = plt.figure(figsize=(17, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1), wspace=.12)
    ax = fig.add_subplot(grid[0], projection="3d")
    dots = ax.scatter(event[:, 0], event[:, 1], -event[:, 2], c=event[:, 2],
                      cmap="viridis_r", s=3, alpha=.32, rasterized=True)
    corners = [(x, y, h) for x in (0., 20.) for y in (0., 7.) for h in (bottom, top)]
    for i, left in enumerate(corners):
        for right in corners[i + 1:]:
            if sum(a != b for a, b in zip(left, right)) == 1:
                ax.plot(*zip(left, right), color=top_color, lw=.9, alpha=.6)
    xx, yy = np.meshgrid([0., 20.], [0., 7.])
    ax.plot_surface(xx, yy, np.full_like(xx, top), color=top_color, alpha=.07, shade=False)
    for (x, y, _), h in zip(station, heights):
        ax.plot([x, x], [y, y], [top, h], color=station_color, lw=.6, alpha=.65)
    ax.scatter(station[:, 0], station[:, 1], np.full(len(station), top), s=18,
               facecolors="none", edgecolors="gray", label="Проекции на плоский верх")
    if np.any(outside):
        ax.scatter(event[outside, 0], event[outside, 1], -event[outside, 2],
                   marker="x", c="red", s=40, label=f"Ниже дна: {int(outside.sum())} события")
    ax.scatter(station[:, 0], station[:, 1], heights, s=45, marker="^", c=station_color,
               edgecolors="white", linewidths=.4, label="Реальные высоты станций", depthshade=False)
    ax.set(xlim=(0, 20), ylim=(0, 7), zlim=(-7, 1.65), xlabel="X, км", ylabel="Y, км",
           zlabel="Высота h, км", title=("Верх модели поднят до станции LCK" if top_at_highest
                                         else "Станции над прежней моделью 20 × 7 × 7 км"))
    ax.set_box_aspect((20, 7, 8.65))
    ax.view_init(elev=20, azim=-62)
    ax.legend(loc="upper left", fontsize=9)
    cb = fig.colorbar(dots, ax=ax, shrink=.55, pad=.01, location="bottom", fraction=.04)
    cb.set_label("Каталожная глубина события, км")

    plan = fig.add_subplot(grid[1])
    plan.scatter(event[:, 0], event[:, 1], s=3, c="gray", alpha=.18, rasterized=True)
    dots = plan.scatter(station[:, 0], station[:, 1], c=heights * 1000., cmap="plasma",
                        s=80, marker="^", edgecolors="black", linewidths=.4, zorder=3)
    for name, (x, y, _), height in zip(stations.ids, station, heights):
        offset = (4, -15) if name in {"AL3", "AL5", "RGP", "HVC", "NEG"} else (4, 5)
        plan.annotate(name, (x, y), xytext=offset, textcoords="offset points", fontsize=8)
    plan.set(xlim=(-.3, 20.4), ylim=(-.5, 7.5), xlabel="X, км", ylabel="Y, км",
             title="Вид сверху: реальные отметки 26 станций")
    plan.set_aspect("equal")
    plan.grid(alpha=.2)
    fig.colorbar(dots, ax=plan, orientation="horizontal", pad=.12, fraction=.07,
                 label="Высота станции по FDSN, м")
    fig.suptitle("Geysers, 1–20 апреля 2011: реальные высоты станций\n"
                 f"{len(stations.ids)} станций · {len(events.ids)} событий · "
                 f"высоты {heights.min()*1000:.0f}–{heights.max()*1000:.0f} м", fontsize=17, y=.98)
    fig.text(.04, .035, note, fontsize=10, color="#444444")
    fig.subplots_adjust(top=.87, bottom=.17, left=.015, right=.98)
    fig.savefig(output_dir / f"geometry_real_elevations{suffix}.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.4), gridspec_kw={"width_ratios": (20, 7)},
                             sharey=True)
    velocity = model.velocity
    dx = model.cell_size_m / 1000.
    norm = Normalize(float(velocity.min()), float(velocity.max()))
    layers = manifest["baseline_config"]["initial_layer_boundaries_km"]
    for coordinate, ax in enumerate(axes):
        if coordinate == 0:
            section = velocity[:, velocity.shape[1] // 2, :].T
            width = 20.
            title = "Проекция X–высота; true-разрез Y ≈ 3.5 км"
        else:
            section = velocity[velocity.shape[0] // 2, :, :].T
            width = 7.
            title = "Проекция Y–высота; true-разрез X ≈ 10 км"
        # Translate the whole velocity cube; do not stretch or resample it.
        image = ax.imshow(section, extent=(0., width, top - section.shape[0] * dx, top),
                          origin="upper", cmap="RdYlBu_r", norm=norm, aspect="equal", alpha=.82)
        ax.scatter(event[:, coordinate], -event[:, 2], s=2.5, c=event_color, alpha=.25,
                   rasterized=True, label="Все события: проекция, не выборка разреза")
        ax.axhline(top, color=top_color, lw=2, label=top_label)
        if top_at_highest:
            ax.axhline(bottom, color="red", lw=1.2, ls=":", label="Нижняя граница модели")
            ax.scatter(event[outside, coordinate], -event[outside, 2], marker="x", c="red", s=45,
                       label=f"Ниже дна: {int(outside.sum())} события", zorder=5)
        for boundary in layers:
            ax.axhline(top - boundary, color="white", lw=1.1, ls="--", alpha=.9)
        for (x, y, _), height in zip(station, heights):
            position = (x, y)[coordinate]
            ax.plot([position, position], [top, height], color=station_color, lw=.7, alpha=.5)
        ax.scatter(station[:, coordinate], heights, s=48, marker="^", c=station_color,
                   edgecolors="white", linewidths=.4, zorder=4, label="Реальные высоты станций")
        ax.set(xlim=(0, width), ylim=(-7, 1.6), xlabel=f"{'XY'[coordinate]}, км", title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Высота h, км; для событий h = −depth")
    if top_at_highest:
        axes[0].text(.3, 1.35, "LCK НА ПОВЕРХНОСТИ; ОСТАЛЬНЫЕ СТАНЦИИ ВНУТРИ", color=station_color, fontsize=11)
        title = ("Поднимаем всю модель: верх на высоте 1166 м\n"
                 f"Станции: глубины 0–839 м от верха; ниже прежнего дна остаются {int(outside.sum())} из {len(events.ids)} событий")
    else:
        axes[0].text(.3, 1.35, "ЗДЕСЬ НЕТ ЯЧЕЕК ПРЕЖНЕЙ МОДЕЛИ", color=station_color, fontsize=11)
        title = ("Как реальные высоты соотносятся с прежней true-моделью\n"
                 "Все 26 станций оказываются на 0.327–1.166 км выше её верхней границы")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=.95)
    fig.suptitle(title, fontsize=17, y=.97)
    fig.subplots_adjust(top=.82, bottom=.2, left=.06, right=.98, wspace=.10)
    fig.colorbar(image, ax=list(axes), orientation="horizontal", fraction=.04, pad=.12,
                 label="True Vp, м/с; пунктир — границы слоёв начальной модели")
    fig.text(.04, .025, note, fontsize=10, color="#444444")
    fig.savefig(output_dir / f"sections_real_elevations{suffix}.png", dpi=160)
    plt.close(fig)
    summary = {"station_count": len(stations.ids), "event_count": len(events.ids),
               "station_elevation_min_m": float(heights.min() * 1000.),
               "station_elevation_max_m": float(heights.max() * 1000.),
               "station_elevation_range_m": float(np.ptp(heights) * 1000.),
               "stations_above_old_model": int(np.sum(heights > 0)),
               "preview_top_elevation_m": top * 1000.,
               "preview_bottom_elevation_m": bottom * 1000.,
               "station_depths_from_top_m": dict(zip(stations.ids, station_depths.tolist())),
               "deepest_event_from_top_m": float(event_depths.max()),
               "events_below_preview_model": [name for name, flag in zip(events.ids, outside) if flag],
               "velocity_cube_translation_only": bool(top_at_highest),
               "vertical_datum_assumption": "station h=elevation_m; event h=-catalogue_depth_m; not independently verified",
               "input_modified": False, "new_solver_run": False}
    (output_dir / f"preview{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--top-at-highest", action="store_true", help="Translate the cube top to the highest station")
    args = parser.parse_args()
    print(json.dumps(preview(args.manifest, args.output_dir, top_at_highest=args.top_at_highest), indent=2))


if __name__ == "__main__":
    main()
