"""Shallow, read-only run inventory; writes only this directory's reports.

Section boundaries are provisional: historical spherical-anomaly studies and the
240-km checkerboards are listed separately until the article scope is clarified.
A missing final_model.npy alone does NOT establish a failed/stopped run.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import os

STATE = Path(__file__).resolve().parent
EMTOMO = STATE.parents[1]
REPOSITORY = EMTOMO.parents[1]
RUNS = EMTOMO / "runs"
SECTIONS = {
    "1": "Шахматка / tomoDD: 24×12×12 км, nx4/nx6/nx8",
    "2": "Шахматка / односторонние станции",
    "3": "Шахматка / два станционных кластера",
    "4": "Шахматка / реальная геометрия / шум",
    "5": "Вертикальный разлом / реальная геометрия",
    "6": "Неоднородности / model_previous / реальная геометрия",
    "7": "Реальные пики",
    "6_extra": "Неоднородности: прежние слои и сферы на реальной геометрии (уточнить)",
    "1_legacy": "Прежние шахматки 240×120×120 км (вне основного счёта)",
    "unassigned": "Другие/старые/неидентифицированные: не распределены",
}


def read(path):
    return json.loads(path.read_text())


def section(source, kind):
    if kind == "real_picks":
        return "7"
    if source.startswith("checkerboard_24x12x12_nx"):
        return "1"
    if source.startswith("onesided_nx6_"):
        return "2"
    if source.startswith("nx6_bottom_blobs_"):
        return "3"
    if source.startswith(("geysers_realgeom_", "geysers_cb5x2x2_real26_", "geysers_nx6_real31_")):
        return "4"
    if source.startswith("gfsoftscreen28_") or "_fault3_x6p5_" in source:
        return "5"
    if source.startswith("geysers_realistic_previous_"):
        return "6"
    if source.startswith("geysers_spheres_picknoise_") or (source.startswith("geysers11_") and "_flat4_" in source):
        return "6_extra"
    if source.startswith("checkerboard_240x120x120_"):
        return "1_legacy"
    return "unassigned"


def main():
    selected_path = EMTOMO / "paper/figures/checkerboard_nx4_nx6_nx8_comparison.json"
    selected = {Path(row[key]).resolve() for row in read(selected_path)["rows"]
                for key in ("em_run", "dd_run")}
    real_queue = RUNS / "geysers_real_linear35_60_latest_20261110_063859/status.json"
    running_real = False
    queue = read(real_queue) if real_queue.exists() else {}
    if queue.get("status") == "running":
        try:
            os.kill(queue["worker_pid"], 0)
            running_real = True
        except (ProcessLookupError, KeyError):
            pass
    rows = []
    for directory in sorted(RUNS.iterdir()):
        meta_path = directory / "meta.json"
        if not directory.is_dir() or not meta_path.exists():
            continue
        meta = read(meta_path)
        params = meta.get("run_params") or {}
        source = meta.get("source_experiment") or {}
        source_id = source.get("id") or source.get("directory") or ""
        group = section(source_id, source.get("kind"))
        final = (directory / "final_model.npy").exists()
        status = "final_saved" if final else "no_final_model"
        if group == "7" and "linear35_60" in source_id and not final and running_real:
            status = "running"
        stopping = read(directory / "stopping.json") if (directory / "stopping.json").exists() else {}
        rows.append(dict(
            section=group, section_name=SECTIONS[group], method="EMTomo", run_id=directory.name,
            run_dir=str(directory.relative_to(REPOSITORY)), source_id=source_id,
            mode=params.get("candidate_mode") or "unspecified", status=status,
            final_model_exists=final,
            completed_markers=sum(1 for _ in directory.glob("iter_*/complete.json")),
            requested_updates=params.get("n_cycles"), completed_updates=stopping.get("completed_updates"),
            stop_reason=stopping.get("reason"), version=meta.get("run_version", params.get("run_version")),
            n_events=params.get("n_events", len(meta.get("event_locs", []))),
            n_stations=len(meta.get("station_locs", [])),
            n_candidates=params.get("n_candidates"), top_n=params.get("weights_top_n"),
            min_distance=params.get("weights_min_distance"), temperature=params.get("temperature"),
            coverage_power=params.get("coverage_damping_power"),
            initial_kind="layers" if params.get("initial_layer_velocities_m_s") else
                         "gradient" if params.get("initial_gradient_m_s") else "other_or_homogeneous",
            selected_checkerboard_figure=directory.resolve() in selected,
        ))
    for nx in (4, 6, 8):
        result = read(RUNS / f"checkerboard_scaled_20261021/cases/nx{nx}-tomodd/result.json")
        directory = Path(result["run_dir"])
        rows.append(dict(section="1", section_name=SECTIONS["1"], method="tomoDD",
                         run_id=f"nx{nx}-tomodd", run_dir=str(directory.relative_to(REPOSITORY)),
                         source_id=result["experiment_id"], mode="tomoDD", status=result["status"],
                         final_model_exists=(directory / "work/inversion/Vp_model.dat").exists(),
                         selected_checkerboard_figure=directory.resolve() in selected))
    summary = {}
    for key in SECTIONS:
        subset = [row for row in rows if row["section"] == key]
        summary[key] = dict(name=SECTIONS[key], total=len(subset),
                            completed=sum(r["status"] in ("final_saved", "completed") for r in subset),
                            running=sum(r["status"] == "running" for r in subset),
                            without_final=sum(r["status"] == "no_final_model" for r in subset),
                            methods=dict(Counter(r["mode"] for r in subset)))
    inventory = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                     scope="Top-level EMTomo meta.json plus three current scaled-checkerboard tomoDD results; not a global tomoDD archive inventory",
                     queued_real_hard_cases=queue.get("pending", 0),
                     summary=summary, runs=rows)
    (STATE / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (STATE / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Инвентаризация расчётов для статьи", "", f"Снимок UTC: {inventory['generated_utc']}", "",
             "Один ран = одна фактически созданная инверсия, не итерация и не прямая задача.",
             "Основной счёт использует секции 1–7; дополнительные семейства не смешаны с ними.",
             "Нет final_model.npy ≠ доказанная ошибка: это может быть остановленный, текущий или старый формат.", "",
             "| Секция | Всего | С финалом / завершены | Без финала | Текущие | Режимы |",
             "|---|---:|---:|---:|---:|---|"]
    for value in summary.values():
        lines.append(f"| {value['name']} | {value['total']} | {value['completed']} | {value['without_final']} | {value['running']} | {value['methods']} |")
    lines += ["", "## Уточнения к интерпретации", "",
              "- Секция 1: в рисунках выбраны 3 hard EMTomo + 3 tomoDD; остальные EMTomo — подбор coverage и незавершённый дубль. Старый tomoDD содержит horizontal-guard; исправленные nx4/nx6/nx8 не найдены.",
              "- Секция 2: 6 однократных пилотов на 10 событиях, 1 однократный soft на полном каталоге и 1 полная soft/hard-пара (12 обновлений).",
              "- Секция 3: 1 soft/hard-пара (12 обновлений), 22 станции: 10+10+2; 600 событий.",
              "- Секция 4: 5 шумовых уровней × soft/hard на обрезанной геометрии = 10 финалов; coverage=0 добавляет ещё 2 финала (3 обновления). В прежней полной геометрии 4 остановленные попытки.",
              "- Секция 5: исходные 4 рана + 37 поисковых (24 soft и 13 переиспользуемых hard-контролей); 24 сравнения — не 24 независимые пары.",
              "- Секция 6: 4 остановленные попытки, 4 финала с лимитом 7, 4 финала с лимитом 20. Последние — 2 стартовые модели × soft/hard, не новые реализации шума. 1D остановилась на 9, однородная достигла 20.",
              "- Секция 7: старый soft завершён (24); старый hard имеет 21 опубликованное обновление без финала. Новая linear35–60 очередь пока не полная пара; см. текущие статусы и pending ниже.",
              f"- Ожидающие реальные случаи (не считаются ранами до создания): {inventory['queued_real_hard_cases']}.",
              "- 6_extra: ещё 35 завершённых EMTomo на слоях/сферах в реальной геометрии. Их включение в секцию неоднородностей требует подтверждения.",
              "- Не распределённые и legacy-раны сохранены в CSV/JSON, а не объявлены потерянными или незавершёнными только из-за возраста.", "",
              "Полный список: `runs.csv`; машинный снимок: `inventory.json`.",
              "Повторный сбор: `venv/bin/python projects/EMTomo/studies/article_inventory/summarize.py` из корня репозитория."]
    (STATE / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
