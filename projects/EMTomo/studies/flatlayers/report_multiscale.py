"""Summarize/plot the completed multiscale control without rerunning inversion.

    OPENBLAS_NUM_THREADS=1 venv/bin/python projects/EMTomo/studies/flatlayers/report_multiscale.py
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiment_data import load_tomography_experiment  # noqa: E402
from velocity_model import block_average_slowness  # noqa: E402


def main():
    result = json.loads((HERE / "multiscale_result.json").read_text())
    run, baseline = ROOT / result["run_dir"], ROOT / result["baseline_run_dir"]
    meta = json.loads((run / "meta.json").read_text())
    data = load_tomography_experiment(meta["source_experiment"]["id"])
    reference = block_average_slowness(data.source_model.velocity, data.source_model.cell_size_m,
                                       tuple(meta["grid_info"]["coarse_shape"]), 500.0)
    warm, cold = result["warm_start"], result["original_hard_c500"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, path, title in zip(axes[0], (baseline, run), ("Original hard-500", "Hard-500 from hard-1000 cycle 8")):
        error = np.load(path / "final_model.npy") - reference
        image = ax.imshow(error[:, :, 0].T, origin="lower", extent=(0, 20, 0, 7),
                          aspect="equal", cmap="RdBu_r", vmin=-500, vmax=500)
        ax.scatter(data.station_coordinates_m[:, 0] / 1000, data.station_coordinates_m[:, 1] / 1000,
                   s=12, facecolor="none", edgecolor="black", linewidth=0.6)
        ax.set(title=title + "\nFinal error, top 0-500 m", xlabel="x, km", ylabel="y, km")
    figure.colorbar(image, ax=axes[0].tolist(), label="Velocity error, m/s", shrink=0.8)
    for rows, label in ((cold, "Original hard-500"), (warm, "From hard-1000 cycle 8")):
        axes[1, 0].plot([r["cycle"] for r in rows], [r["model_metrics"]["rms_error_m_s"] for r in rows],
                        marker="o", markersize=3, label=label)
        axes[1, 1].plot([r["cycle"] for r in rows],
                        [r["model_metrics"]["top_500m"]["rms_error_m_s"] for r in rows],
                        marker="o", markersize=3, label=label)
    for ax, title in zip(axes[1], ("All cells RMSE", "Top 0-500 m RMSE")):
        ax.set(title=title, xlabel="500 m updates (0 = initial model)", ylabel="m/s", xticks=range(0, 13, 2))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    figure.savefig(HERE / "multiscale_comparison.png", dpi=160)
    plt.close(figure)

    lines = ["# hard-c1000, цикл 8 → hard-c500", "",
             "Взята `iter_8/model.npy`: модель **после восьми полных обновлений**. "
             "Каждый километровый блок разделён на 2×2×2 блока 500 м без изменения скорости. "
             "Поле скорости на расчётной сетке 125 м совпало побитово.", "",
             "Гипоцентры не переносились и не фиксировались: обычный hard-поиск, K=5, одна выбранная гипотеза. "
             "Выполнены все 12 циклов; lambda=0.5, coverage power=1, шаг ±2%, "
             "52 станции, 1920 событий. Регуляризация к стартовой модели не добавлялась.", "",
             "Для чистого сравнения использована исходная формула forward из сохранённых hard-c1000/c500, "
             "до последующей v2-правки внутренней среды сферы. Replay действует только внутри этого "
             "study-процесса; production-код не откатывался. Версия записана в meta.json.", "",
             "## Результаты на одной и той же сетке оценки 500 м", "",
             "| Цикл 500 м | Вся модель: исходный / multiscale, м/с | Верхние 500 м: исходный / multiscale, м/с |",
             "|---:|---:|---:|"]
    for cycle in (0, 1, 2, 4, 8, 12):
        a, b = cold[cycle]["model_metrics"], warm[cycle]["model_metrics"]
        lines.append(f"| {cycle} | {a['rms_error_m_s']:.2f} / {b['rms_error_m_s']:.2f} | "
                     f"{a['top_500m']['rms_error_m_s']:.2f} / {b['top_500m']['rms_error_m_s']:.2f} |")
    c, w = cold[-1]["model_metrics"], warm[-1]["model_metrics"]
    lines += ["", "**Верхний слой лучше, модель в целом хуже.** "
              f"Финальная RMS верхних 500 м уменьшается с {c['top_500m']['rms_error_m_s']:.2f} "
              f"до {w['top_500m']['rms_error_m_s']:.2f} м/с "
              f"({100 * (1 - w['top_500m']['rms_error_m_s'] / c['top_500m']['rms_error_m_s']):.1f}%). "
              f"Общая RMS увеличивается с {c['rms_error_m_s']:.2f} до {w['rms_error_m_s']:.2f} м/с.", "",
              "При этом верх не становится правильным: его ошибка растёт с 33.54 до 77.29 м/с "
              "за 12 обновлений на сетке 500 м.", "",
              "Основное ухудшение относительно исходного hard-c500 — не в верхних 500 м. "
              "Например, финальная RMS на глубине 1.5–2 км: 247.8 против 155.6 м/с; "
              "3–3.5 км: 252.3 против 185.5 м/с.", "",
              "Локализация: медиана ошибки первого/последнего E-шага multiscale — "
              f"{warm[0]['hypocentre_error_m']['median']:.1f} / {warm[-2]['hypocentre_error_m']['median']:.1f} м; "
              "для исходного hard-c500 — "
              f"{cold[0]['hypocentre_error_m']['median']:.1f} / {cold[-2]['hypocentre_error_m']['median']:.1f} м.", "",
              "Невязка последнего E-шага почти одинакова: "
              f"{cold[-2]['centered_residual_rms_s'] * 1000:.3f} / "
              f"{warm[-2]['centered_residual_rms_s'] * 1000:.3f} мс (исходный / multiscale). "
              "Это значения **до последнего скоростного обновления**, не повторная локализация в final_model.", "",
              "RMS 83.56 м/с у исходного hard-c1000 цикла 8 нельзя прямо сравнивать с "
              "137.27 м/с у перенесённой модели: эталон усредняется на разных ячейках. "
              "Сам перенос расчётное поле не изменяет; все числа таблицы выше рассчитаны относительно "
              "одного эталона на сетке 500 м.", "",
              "![Сравнение](multiscale_comparison.png)", "", "## Артефакты", "",
              f"- Новый запуск: `{result['run_dir']}` (относительно `projects/EMTomo`).",
              "- Во viewer: `fl-hard-c500-from-c1000-c8`.",
              "- `multiscale_result.json`: метрики всех циклов, источник старта и контрольная сумма.",
              "- `logs/hard-c500-from-c1000-c8.log`: журнал запуска.",
              "- `run_multiscale.py`: воспроизводимый запуск; `report_multiscale.py`: этот отчёт.", "",
              "Проверки: 12 completion markers, контрольная сумма исходного checkpoint, точный перенос "
              "скоростей, 0 отказов из 1 198 080 лучей. Время EM: 705.97 с, около 11.8 минуты."]
    (HERE / "multiscale_report.md").write_text("\n".join(lines) + "\n")
    print(HERE / "multiscale_report.md")
    print(HERE / "multiscale_comparison.png")


if __name__ == "__main__":
    main()
