# Рисунки текущей статьи

Здесь находятся **11 PDF, реально подключённых** из `paper/main.tex` и `paper/results.tex`, в порядке появления в статье. На 17.11.2026 `experiments.tex` не подключён в `main.tex` и не определяет этот список.

| № | Рисунок | Файл |
|---|---|---|
| 1 | Checkerboard: модель и геометрия, nx6 | [checkerboard_nx6_geometry.pdf](checkerboard_nx6_geometry.pdf) |
| 2 | Checkerboard: сравнение реконструкций nx4/nx6/nx8 | [checkerboard_nx4_nx6_nx8_comparison.pdf](checkerboard_nx4_nx6_nx8_comparison.pdf) |
| 3 | Checkerboard: положения гипоцентров, nx6 | [checkerboard_nx6_locations_y10p2km.pdf](checkerboard_nx6_locations_y10p2km.pdf) |
| 4 | Односторонняя, двухкластерная и реалистичная геометрии | [all_geometries.pdf](all_geometries.pdf) |
| 5 | Общие истинные checkerboard-модели | [checkerboards_4_2-4_4_true_final.pdf](checkerboards_4_2-4_4_true_final.pdf) |
| 6 | Checkerboard: hard/soft-реконструкции | [checkerboards_4_2-4_4_results_final.pdf](checkerboards_4_2-4_4_results_final.pdf) |
| 7 | Вертикальный разлом: истина и реконструкции | [section_4_5_final.pdf](section_4_5_final.pdf) |
| 8 | Неоднородная модель: истина и реконструкции | [section_4_6_final.pdf](section_4_6_final.pdf) |
| 9 | Реальные данные The Geysers: сравнение моделей | [real_picks_comparison_final.pdf](real_picks_comparison_final.pdf) |
| 10 | Плотная и разреженная дискретизация кандидатов EVT_0905 | [dense_vs_sparse_EVT_0905_all_candidates_final.pdf](dense_vs_sparse_EVT_0905_all_candidates_final.pdf) |
| 11 | Связь выигрыша soft по скорости и гипоцентрам | [soft_velocity_gain_vs_hypocenter_gain.pdf](soft_velocity_gain_vs_hypocenter_gain.pdf) |

## Архив и воспроизводимость

- Остальные 197 графических файлов (включая PNG/SVG-превью этих рисунков и старые варианты) перенесены в [`archive/unused_paper_figures/`](../../../archive/unused_paper_figures/) вне `paper`, с сохранением структуры относительно прежнего `paper/figures/`.
- [`manifest.json`](../../../archive/unused_paper_figures/manifest.json) содержит старый и новый путь и SHA-256 каждого перенесённого файла, в том числе 11 используемых PDF.
- Данные, JSON/NPZ, подписи и скрипты не перемещались. Исходные сопутствующие файлы остались в прежних подпапках `paper/figures/`; старый README находится на уровень выше.
- Генераторы пока сохраняют прежние выходные пути. При переработке каждого рисунка нужно обновить его генератор на `paper/figures/used/` либо перенести готовый PDF в эту папку; повторный запуск старого генератора сам по себе не обновит PDF, включённый в статью.
- При первоначальной сортировке менялись только размещение файлов и пути LaTeX. SHA-256 в архивном manifest фиксирует версии на момент переноса, а не последующих правок.

## Подготовка рисунка 1

`paper/paper_scripts/make_checkerboard_geometry.py` теперь пишет PDF, PNG, SVG и JSON непосредственно в `paper/figures/used/`. Палитра — `cmcrameri.cm.lajolla_r` (`cmcrameri==1.10` в зависимостях). Единицы указаны через запятую; заголовки панелей — только (a), (b), (c); следы срезов A и B подписаны только на виде сверху, без координат. (b) соответствует A, (c) — B. Остальные генераторы пока сохраняют старые выходные пути.

Пересборка из EMTomo:

```sh
python -B paper/paper_scripts/make_checkerboard_geometry.py
```

## Подготовка рисунка 2

`paper/paper_scripts/make_checkerboard_comparison.py` пишет PDF, PNG, SVG и JSON в `paper/figures/used/`. Палитра — `vik_r` с центральным цветом, закреплённым в чисто белом. Показаны возмущения ΔV = V − 5 км/с в общем симметричном диапазоне −1…+1 км/с без обрезания данных; белый соответствует V = 5 км/с. Горизонтальная шкала находится сверху и занимает 28% ширины рисунка. Столбцы обозначены (a)–(d); значения nx, расшифровка методов и положения срезов вынесены в подпись статьи. Единицы указаны через запятую, глубина обозначена Z; подписи вертикальных осей расположены горизонтально. Исходные данные и срезы не менялись.

```sh
python -B paper/paper_scripts/make_checkerboard_comparison.py
```
