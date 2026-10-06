# Архив сценариев EMTomo

Код здесь сохранён для изучения старых постановок, **не** для основной серии
экспериментов статьи. Актуальная схема: независимая генерация наблюдений в
`projects/forward_modeling` и запуск `python main.py EXPERIMENT_ID` из каталога
`projects/EMTomo` (см. `../EXPERIMENTS.md`).

## `legacy_synthetics/`

Семь исторических launchers с синтетикой, создаваемой самим EMTomo.
Их конфигурации образуют цепочку импортов, поэтому файлы перенесены вместе.
Сам legacy-путь находится в `runner.py`, внутренний генератор наблюдений — в
`instruments_synthetic.py`, генерация случайных станций и событий — в
`locations.py`. Рабочие `main.py` и `instruments/instruments.py` больше не
импортируют этот генератор.
Если всё же нужно воспроизвести сценарий, запускайте **из `projects/EMTomo`**:

```sh
python -m archive.legacy_synthetics.checkerboard_4x2x2
python -m archive.legacy_synthetics.checkerboard_4x2x2_small
python -m archive.legacy_synthetics.checkerboard_4x2x2_sub8_suite
python -m archive.legacy_synthetics.checkerboard_4x2x2_sub8_final
python -m archive.legacy_synthetics.checkerboard_pattern_4x2x2_full
python -m archive.legacy_synthetics.checkerboard_pattern_4x2x2_event_grid
python -m archive.legacy_synthetics.geysers_checkerboard
```

**Осторожно:** это долгие расчёты; `sub8_suite` запускает несколько подряд.
Сценарии вызывают `archive.legacy_synthetics.runner.main(config)` и текущую
реализацию инверсии, а не замороженную копию старого метода. В активном
`main.py` теперь требуется `experiment_id`: вызов `main(config)` без ID
выдаёт ошибку. `generate_synthetic_arrivals_table` больше не экспортируется
через `instruments.instruments`; для архивных тестов его импортируют из
`archive.legacy_synthetics.instruments_synthetic`. Старые `run_version="1.0"`
в их конфигурациях отражают исходные сценарии, но **не гарантируют**
воспроизведения численных результатов исторических запусков.
Относительные пути к `runs/cache/` и `data/geysers_2011/` рассчитаны на
запуск из `projects/EMTomo`. Скрипт `../prepare_geysers_geometry.py` и
исходные данные оставлены на прежнем месте.

## `legacy_misc/`

`example_usage.py`, `example_geo_grid.py` — старые примеры API; могут требовать
адаптации к нынешнему коду. `preliminary.py` при исполнении обращается к
внешнему API рельефа и строит изображение — не импортируйте его для тестов.
`delete_misfit.sh` — старый **удаляющий** скрипт с жёстко заданной датой и
зависимостью от формата имён каталогов. Не запускайте его без ручной проверки:
он не предназначен для уборки текущих `runs/`.

Ничего из `runs/`, `pictures/`, `summary/`, `paper/` или исходных данных при
архивации не удалено. В частности, `runs/` содержит не только результаты, но
и локальные скрипты и манифесты исследований.
