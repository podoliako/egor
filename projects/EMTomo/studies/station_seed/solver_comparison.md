# Продолжение: устойчивый forward против точности и полного inverse-контроля

Это продолжение `report.md`. Регуляризация к начальной модели не добавлялась,
параметры обновления/число циклов не менялись. Новые решатели остаются только в
`studies/station_seed`; production по-прежнему использует исправленную формулу
`skfmm ... homogeneous_sphere_v2` из предыдущего этапа.

## Решение

**Не заменять production ни PyKonal-кандидатом, ни factored-кандидатом по этим
результатам.** Первый сохраняет скачки на малых возмущениях; второй устраняет
их в проверках, но теряет точность на части неоднородных задач и ухудшает
полную fixed-c500 инверсию.

Это не означает, что найденная негладкость FMM безвредна. Устойчивый вариант
проходит null-контроль, который текущий forward проваливает. Но устойчивость
сама по себе недостаточна для качественного восстановления: надо учитывать
точность физической прямой задачи и согласованность с используемым `G`.

## 1. Проверенный, но нерелевантный к скачку upstream-баг

В `scikit-fmm 2025.06.23`, `updatePointOrderTwo`, обнаружена дополнительная
ошибка: при выборе более близкого первого соседа из противоположной стороны
`value2` может сохраняться от старой стороны. Если у нового направления второй
сосед недопустим, смешиваются разные стороны stencil.

Reset-only правка `value2 = maxDouble` подтверждена локальным C++ пробником:
время **0.233333 с** вместо ожидаемого fallback **0.25 с**, асимметрия при
отражении оси. После сброса пробник корректен. Аналогичная ошибка имеется в
`distance_marcher.cpp`.

Однако original/reset-only поля **побитово совпали** во всех проверенных ULP
3D/2D примерах, включая ±1, ±8, ±64 и ±1024 ULP. Это отдельный доказанный баг,
но **не найденное исправление исследуемых скачков**. В установленную библиотеку
и зависимости он не вносился.

Контроль — `stencil_controls.py`, результаты —
`stencil_controls_verified.json`. Скрипт скачивает pinned upstream C++ source,
использует C++ compiler и Linux memfd, не меняет venv. Это необязательная
исследовательская проверка, не часть runtime EMTomo.

## 2. PyKonal с защищённой аналитической сферой

Использован уже установленный `pykonal==0.4.1`, без новых зависимостей.
Cell-centred grid сохранён. Аналитические внутренние времена зафиксированы
через `known/unknown`; внутренний launch-front помещён в trial.

На исходном continuity-примере `(48,32,32)`, h=125 м:

| Возмущение Vp | Максимум изменения T, PyKonal | Corrected skfmm |
|---|---:|---:|
| +1 ULP | 0 | 1.743 мс |
| +8 ULP | **2.410 мс** | 2.391 мс |
| около +1e-11 м/с | **2.750 мс** | 2.224 мс |

Следовательно, **одного одно-ULP контроля недостаточно**. PyKonal-кандидат
отклонён до инверсии. На части однородных геометрий аналитическая ошибка ниже,
но это не устраняет негладкость.

Скрипт/результаты: `pykonal_candidate.py`, `pykonal_screen.json`.

## 3. Экспериментальный monotone factored Godunov

Реализация: `factored_candidate.py`. Это собственный study-only численный
контроль, не скрытая замена backend и не второй порядок обычного FMM.

Факторизация `T=r*u`, где `r` — точное расстояние до источника. Для каждой оси
используются обе направленные first-order ветви:

```
L = (r/h + dr/dx) u_i - (r/h) u_left
R = (r/h - dr/dx) u_i - (r/h) u_right
H(u_i) = sum_axes max(L, R, 0)^2 = slowness_i^2
```

Вне seed радиуса 3h коэффициенты при центральном factor положительны.
Локальный непрерывный монотонный корень решается из supersolution, сетка —
восемью направлениями fast sweeping до сходимости **forward equation**.
Это не критерий остановки инверсии.

Важные детали:

- numerical nodes включают физические границы домена, так что surface source
  не оказывается на полшага вне nodal domain;
- missing directions отсутствуют: никаких искусственных Neumann-ghost данных
  для factor не вводится;
- nodal slowness — среднее соседних fine-cell slownesses после homogenization
  внутренней сферы;
- результат в центрах ячеек получается интерполяцией factor и умножением на
  точное расстояние, не интерполяцией source cusp;
- каждый solve начинается с новой supersolution; старое поле не переиспользуется;
- проверяется signed Hamiltonian residual, сходимость и конечность результата;
- изменения API guards после измерений касались только invalid/nonfinite
  параметров и проверки finite outputs; численные kernels не менялись.

Это **другая дискретизация** прямой задачи. Её непрерывность ещё не гарантирует
равноценную физическую точность или согласованность с прежним ray Jacobian.

### Forward screen

49 парных случаев: homogeneous/corners/off-grid, invariants, reflections,
ULP, плоские слои с Snell/head-wave reference и spherical boundary stress.
Все завершились без solver failures. Screen повторён после defensive guards.

- Homogeneous error до **5.6e-16 с** в float64.
- Heterogeneous ±1/8/64 ULP и mixed signs: изменения до **4.3e-15 с**.
- Отражения/перестановки осей: различия до **4.5e-16 с**.
- Nonstation velocities внутри seed не влияют на поле; read-only вход не меняется.
- Signed relative Hamiltonian residual не хуже **1.4e-13**: наблюдаемая
  физическая ошибка не объясняется недосходимостью sweeps.

Но accuracy tradeoff существенен. RMS относительно аналитики, мс,
фиксированный физический радиус seed 375 м:

| h, м | Слои: factored / skfmm | Slow sphere → fast outside: factored / skfmm |
|---:|---:|---:|
| 125 | 10.03 / 10.18 | **12.04 / 2.27** |
| 62.5 | 4.73 / 4.20 | **4.57 / 0.94** |
| 31.25 | 2.24 / 2.10 | **3.24 / 0.48** |

На мелкой сетке oblique transmitted RMS **1.39 против 0.381 мс**;
head-wave RMS **4.79 против 4.94 мс**. Не все ветви ухудшаются одинаково.
На radial stress максимальная ошибка около **13 против 1.67 мс**.
Грубого обхода seed ниже voxelization-envelope не обнаружено; это не
доказательство отсутствия всех граничных погрешностей.

Screen: `factored_screen.py`, `factored_screen.json`.

## 4. Однородные inverse-контроли: полезное улучшение

Те же геометрия и параметры, что на предыдущем этапе: 12 станций, 160 событий,
6×4×4 км, coarse500/fine125 м, lambda0.5, coverage1, шаг ±2%, **12 циклов**.
Истина и старт — 4000 м/с везде; гипоцентры истинные; шума нет. Использованы
неизменённые production normal equations, ray deposition и update.

| Метрика | Corrected skfmm | Factored |
|---|---:|---:|
| Own-forward null: финальная RMS верхнего слоя, м/с | **9.654** | **4.0e-12** |
| Аналитические times: стартовая centered RMS невязки, мс | 1.6836 | **0.1168** |
| Аналитические times: финальная RMS модели, м/с | 19.268 | **1.640** |
| Аналитические times: финальная RMS верхнего слоя, м/с | 37.567 | **4.018** |
| Аналитические times: финальная centered RMS невязки, мс | 0.9478 | **0.0416** |

Устойчивый solve выдерживает самосогласованный null, а не начинает подгонять
миллисекундные скачки после машинно малого первого обновления.

Оставшиеся 0.1168 мс невязки к аналитике обусловлены в том числе обычной
интерполяцией возвращённых cell-centred времён на координаты событий:
численно точные значения поля в центрах ещё не делают эту интерполяцию точной.
Интерполятор EMTomo в этом эксперименте не менялся.

## 5. Полный fixed-c500: замена не оправдалась

Все 1920 событий, 52 станции, те же 12 циклов и сохранённая конфигурация.
В таблице baseline — **corrected skfmm** из предыдущего этапа, а не старая
неисправленная затравка. Число неизвестных, noise weights, limit step,
coverage damping, initial и истинные гипоцентры одинаковы.

| Метрика после 12 циклов | Corrected skfmm | Factored |
|---|---:|---:|
| RMS ошибки всей модели, м/с | **159.479** | **219.182** |
| RMS ошибки верхних 500 м, м/с | **95.741** | **103.835** |
| Латеральная RMS верхних 500 м, м/с | **89.854** | **99.295** |
| RMS ошибки станционных ячеек, м/с | **96.442** | **144.161** |
| Начальная centered RMS невязки, мс | 27.404 | 27.972 |
| Финальная centered RMS невязки, мс | **21.231** | **21.390** |

Factored run завершён за **504.0 с**, около 8.4 минуты.
Отказов лучей **0 из 1 198 080**. Статус `complete`, выполнены все 12 циклов.

Следовательно, улучшение homogeneous/null **нельзя переносить** на
heterogeneous reconstruction. Ray `G` оставлен прежним; изменены marching,
nodal representation и factor interpolation. Этот эксперимент не разделяет
вклад их точности, рассогласования Jacobian и остальных механизмов обратной
задачи. Из него нельзя ни оправдать rollout, ни объявить прежние скачки
единственной причиной верхних артефактов.

Скрипт: `factored_inverse_controls.py`. Результаты и финальные модели:
`factored_inverse_results.json`. Журналы: `factored_homogeneous.log`,
`factored_flat4.log`.

## 6. Что оставлено

- Production backend и зависимости **не заменены**.
- Предыдущее локальное исправление внутренней среды sphere сохранено.
- Устойчивый factored вариант оставлен как исследовательский контроль с явным
  описанием его first-order accuracy tradeoff.
- Добавлены 22 regression tests для experimental solver: homogeneous boundaries,
  ULP continuity, seed/medium contract, local Newton против independent bracketed
  root, convergence failure, rejection invalid parameters.
- Нерешённая проблема production `skfmm order=2` по-прежнему отмечена strict xfail.

Прежде чем рассматривать другую замену forward, надо отдельно отделить точность
и согласованность `G` на **неоднородных** моделях. Одного красивого homogeneous
контроля теперь явно недостаточно. Результаты не дают оснований ни добавлять
prior к initial, ни менять число циклов: этих вмешательств здесь не было.

## Воспроизведение

Из корня `egor`, окружение с уже установленными зависимостями:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python projects/EMTomo/studies/station_seed/pykonal_candidate.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python projects/EMTomo/studies/station_seed/factored_screen.py --reset
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 venv/bin/python projects/EMTomo/studies/station_seed/factored_inverse_controls.py homogeneous --workers 4
OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12 venv/bin/python projects/EMTomo/studies/station_seed/factored_inverse_controls.py flat4 --workers 12
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=projects/EMTomo venv/bin/python -m pytest projects/EMTomo/tests projects/EMTomo/studies/station_seed/test_factored_candidate.py -q
```

Полная последняя валидация: **429 passed, 1 xfailed**, 57.32 с.
`git diff --check` прошёл; существующие пользовательские изменения не откатывались.
