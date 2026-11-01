# Geysers: три скоростных слоя, разлом x=6,5 км, чистые времена

Эксперимент: `geysers11_20110401-20110420_fault3_x6p5_elev1166_26st_3776ev_clean`.

- Область 20×7×7 км, входная сетка 125 м, независимый PyKonal forward с refinement=2 (62,5 м).
- Все **26 станций и 3776 событий** из той же elevated-геометрии, что в серии pick-noise; 98176 пар.
- Порядок/ID/XYZ не меняются, геометрия проверяется по SHA-256. Нового отбора каталога нет.
- z направлена вниз от плоского верха H=1166 м; высоты станций сохранены. Предположение об общем вертикальном датуме каталога и станций остаётся непроверенным, как в исходном наборе.
- Разлом x=6,5 км, модель неизменна вдоль y, сфер нет.
- Слева: границы 0,7/1,9 км; Vp 3600/4400/5000 м/с.
- Справа: границы 1,1/2,9 км; Vp 4000/4800/5500 м/с.
- Границы представлены по центрам 125-м ячеек: фактически слева 0,75/1,875 км, справа 1,125/2,875 км. Разлом попадает точно на грань ячеек.
- Начальная 1-D для будущей инверсии: 0,85/2,3 км; 3850/4550/5300 м/с. Не участвует в прямой задаче.
- Шум выключен. `experiment.py` выполняет только прямую задачу, без EM/tomoDD и без дополнительных проверок сходимости.

Из `projects/EMTomo`:

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/experiment.py prepare
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/experiment.py run --workers 32
```

`prepare` повторно проверяет существующие входы, но не меняет их. `run` не перезаписывает существующие выходы и не переиспользует молча старый расчёт.

Стандартные входы/выходы — в `projects/forward_modeling/experiments/{input,output}/<ID>/`. Статус расчёта — `runs/geysers_faulted_layers/status.json`; фоновый запуск пишет `forward.log` и `launch.json` рядом.

В выходе сохраняются:
- `arrivals.csv`: чистые времена относительно первого прихода каждого события, стандартный формат EMTomo;
- `metadata.json`: параметры, входные SHA-256, информация о решателе, noise.enabled=false и SHA-256 кэша;
- `clean_times.npz`: **абсолютные чистые времена** `absolute_times_s[event, station]`, Unicode `event_ids`/`station_ids`, `event_coordinates_m`/`station_coordinates_m`.

После окончания автоматически проверяются совместимость с загрузчиком EMTomo, все пары, ID/XYZ, нормализация и SHA-256 кэша.

## Hard-инверсия чистых синтетических данных

`run_hard.py` использует замороженный runtime последней Geysers pick-noise серии (`runs/matched_geometry_20261021/snapshot`), проверяет его SHA-256 и берёт конфигурацию `hard-k15-top1-dmin1` из её манифеста. Не использует скачанные реальные пики.

Сохраняются 12 итераций, 15 кандидатов / top1 / dmin1, cell=1000 м, subdivision=8, nearest, λ=1, coverage power=2, floor=0,05, percentile=75, шаг ±2%, T=1, model sigma=0,02 с, 12 workers. Изменены только начальные слои на согласованные 0,85/2,3 км и 3850/4550/5300 м/с, обе pick-noise sigma на 0 и имя/путь результата. Стартовая модель на километровой сетке задаётся по центрам ячеек: первые три значения по z — 3850, 4550, 5300 м/с, ниже 5300 м/с.

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/run_hard.py --validate-only
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/run_hard.py
```

Не запускать вторую копию, пока первая работает. Результат — стандартный `runs/run_gfault28-hard-...`, доступный viewer. Фоновый лог — `runs/geysers_faulted_layers/hard.log`, статус — `hard_status.json`, команда и PID — `hard_launch.json` в том же каталоге. Начальный dry-run проверяет все наблюдения, модель, шум и frozen runtime, без EM и JIT-прогрева.

## Согласованная широкая пара hard/soft, coverage power=0

28 октября запущены параллельно два дополнительных прогона: `gfault28-hard-k64-top20-dmin4-cov0` и `gfault28-soft-k64-top20-dmin4-cov0`. Первый hard k15/cov2 не остановлен и не изменён.

Оба используют 64 уточняемых кандидата, одинаковый top20, dmin4 (500 м Chebyshev по мелкой сетке ДО уточнения), coverage power=0, λ=1. Hard оставляет one-hot вклад одной гипотезы из тех же top20, soft — их вероятностную смесь; top20 не гарантирует 20 независимых мод или 20 ненулевых весов. Coverage power=0 отключает зависимость демпфирования от покрытия, но не саму регуляризацию λ=1. Остальные научные настройки — как в первом hard: 12 итераций, ячейка 1 км, subdivision8/nearest, T=1, шаг ±2%, model sigma=20 мс, pick-noise sigma=0, те же начальные слои и все 3776×26 наблюдений. По 12 workers на прогон.

Для обоих зафиксирован один snapshot **текущего ускоренного кода**, отдельно от старого runtime: `runs/geysers_faulted_layers/wide_cov0/snapshot`. Включены batch8, compact rays, fast refinement; solver=dense. В `manifest.json` рядом сохранены SHA-256 25 Python-файлов runtime и входных/выходных данных. Различия конфигураций пары автоматически проверены: только candidate_mode и run_name. Для сравнения с первоначальным k15/cov2 надо учитывать также изменение поиска и реализации; это не изолированный тест coverage power.

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/run_wide.py hard --validate-only
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/run_wide.py soft --validate-only
```

Без `--validate-only` команды запускают новые инверсии: не запускать дубликаты уже работающих. Для фоновых процессов заданы OPENBLAS/OMP/MKL/NUMBA_NUM_THREADS=1. Статусы, команды/PID, run_dir и логи — в `runs/geysers_faulted_layers/wide_cov0/{hard_status.json,soft_status.json,launch.json,hard.log,soft.log}`. Результаты — стандартные viewer-совместимые `runs/run_gfault28-{hard,soft}-k64-top20-dmin4-cov0-...`. Новая реализация сохраняет `iter_N/performance.json` с разбивкой времени.

Проверки перед запуском: 168 существующих focused tests, оба dry-run, совпадение конфигураций за исключением режима/имени. Реальные пики не используются; новая прямая задача не запускалась.

## Дополнительный soft T=8

Отдельно, с начальной модели (не продолжение T=1), запущен `gfault28-soft-k64-top20-dmin4-cov0-t8`. Температура **постоянно 8** на всех 12 итерациях, без annealing; все прочие параметры и snapshot/data совпадают с soft T=1. Это диагностический тест tempering likelihood, не изменение заявленного шума данных. Старые прогоны не перезапускаются.

```sh
/mnt/disk01/egor/venv/bin/python3 -B studies/geysers_faulted_layers/run_wide.py soft --temperature 8 --validate-only
```

Без `--validate-only` — новая инверсия; не запускать дубликат. Лог `wide_cov0/soft-t8.log`, статус `soft-t8_status.json`, команда/PID/run_dir — `launch_t8.json` в том же каталоге состояния. У launcher сохранён отдельный SHA-256 runner, исходный `launch.json` пары T=1 не изменён.

## Будущие поднаборы и шум — без повторной прямой задачи

Создавать отдельные дочерние эксперименты, не менять чистый исходник. Выбрать станции из `clean_times.npz`; при необходимости применить существующий `add_arrival_noise` к **абсолютным чистым** временам, затем вычесть минимум среди оставшихся станций для каждого события. Абсолютные времена нужны для корректного масштаба относительной компоненты шума. В текущем EMTomo загрузчик требует полный прямоугольник event×station: общий поднабор станций поддерживается форматом, разные маски по событиям потребуют отдельной поддержки.

Тесты (без FMM):

```sh
/mnt/disk01/egor/venv/bin/python3 -B -m pytest -q -p no:cacheprovider studies/geysers_faulted_layers/test_experiment.py ../forward_modeling/tests/test_experiments.py
```
