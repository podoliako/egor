# Seismic Velocity Model

Модуль для работы со скоростными моделями в задачах сейсмической томографии.

## Основные возможности

- Задание 3D сетки с географической привязкой
- Хранение параметров Vp и Vs для каждой ячейки
- Быстрый доступ к значениям по индексам (i, j, k)
- Возможность заполнения модели линейным градиентом по глубине
- Массовые операции с numpy массивами
- Генерация уплотненной geo-сетки для raytracing с интерполяцией
- Подменяемые стратегии интерполяции
- Сохранение/загрузка в JSON формате

## Установка

Требуется только numpy:
```bash
pip install numpy
```

## Быстрый старт

```python
from velocity_model import VelocityModel

# Создание модели из конфига
config = {
    'lon': 37.6173,      # Долгота центра опорного кубика
    'lat': 55.7558,      # Широта центра опорного кубика
    'height': 0.0,       # Высота относительно уровня земли (м)
    'azimuth': 45.0,     # Азимут ориентации сетки (градусы от севера)
    'side_size': 100.0,  # Размер стороны кубика (м)
    'n_x': 50,           # Количество ячеек по оси X
    'n_y': 50,           # Количество ячеек по оси Y
    'n_z': 30            # Количество ячеек по оси Z (вглубь)
}

model = VelocityModel.from_config(config)

# Заполнение линейным градиентом
model.fill_linear_gradient('vp', top_value=1500.0, bottom_value=6000.0)
model.fill_linear_gradient('vs', top_value=800.0, bottom_value=3500.0)

# Изменение отдельных значений
model.set_vp(10, 20, 15, 4500.0)
model.set_vs(10, 20, 15, 2500.0)

# Получение значений
vp = model.get_vp(10, 20, 15)
vs = model.get_vs(10, 20, 15)

# Сохранение модели
model.to_json('my_model.json', include_data=True)

# Загрузка модели
loaded_model = VelocityModel.from_json('my_model.json')

# Генерация уплотненной сетки для raytracing
geo_grid = model.get_geo_grid(subdivision=3)  # 27x refinement
vp_array = geo_grid.vp  # numpy array для быстрого доступа
vs_array = geo_grid.vs

## Структура классов

### VelocityModel
Главный класс, объединяющий геометрию и данные.

**Методы:**
- `from_config(config)` - создание из словаря
- `from_json(filepath)` - загрузка из JSON
- `to_json(filepath, include_data=True)` - сохранение в JSON
- `set_vp(i, j, k, value)` - установка Vp в ячейке
- `get_vp(i, j, k)` - получение Vp из ячейки
- `set_vs(i, j, k, value)` - установка Vs в ячейке
- `get_vs(i, j, k)` - получение Vs из ячейки
- `set_vp_array(values)` - установка всего массива Vp
- `set_vs_array(values)` - установка всего массива Vs
- `fill_linear_gradient(param, top_value, bottom_value)` - линейный градиент
- **`get_geo_grid(subdivision, interpolation)`** - генерация geo-сетки для raytracing

### GeoGrid
Уплотненная сетка для raytracing, генерируется из VelocityModel.

**Атрибуты:**
- `shape` - размеры сетки (n_x*sub, n_y*sub, n_z*sub)
- `cell_size` - размер исходной (скоростной) ячейки в метрах
- `subdivision` - коэффициент уплотнения
- `vp` - numpy массив P-волн
- `vs` - numpy массив S-волн

**Использование:**
```python
geo = model.get_geo_grid(subdivision=3, interpolation='trilinear')
vp_data = geo.vp  # прямой доступ к numpy массиву
```

### GridGeometry
Геометрия и пространственная привязка сетки.

**Параметры:**
- `lon` - долгота центра опорного кубика (градусы)
- `lat` - широта центра опорного кубика (градусы)
- `height` - высота относительно уровня земли (метры)
- `azimuth` - азимут ориентации сетки (градусы, по часовой от севера)
- `side_size` - размер стороны кубика (метры)
- `n_x, n_y, n_z` - количество ячеек по осям

### VelocityGrid
Хранилище данных скоростей.

**Атрибуты:**
- `vp` - numpy массив размера (n_x, n_y, n_z) для P-волн
- `vs` - numpy массив размера (n_x, n_y, n_z) для S-волн

## Индексация

Модель использует индексы `(i, j, k)` где:
- `i` - индекс по оси X (0 до n_x-1)
- `j` - индекс по оси Y (0 до n_y-1)
- `k` - индекс по оси Z (0 до n_z-1), **увеличивается вглубь**

## Работа с numpy массивами

Для эффективной работы можно напрямую обращаться к массивам:

```python
# Прямой доступ к массиву
vp_array = model.grid.vp  # numpy array shape (n_x, n_y, n_z)

# Модификация массива
vp_array[10:20, 15:25, 5:10] = 5000.0

# Применение функции к массиву
model.grid.vp = model.grid.vp * 1.1  # увеличить все скорости на 10%

# Создание из функции
import numpy as np
for i in range(model.geometry.n_x):
    for j in range(model.geometry.n_y):
        for k in range(model.geometry.n_z):
            depth = k * model.geometry.side_size
            vp = 1500 + 0.5 * depth  # v = v0 + k*z
            model.set_vp(i, j, k, vp)
```

## Workflow для raytracing

Типичный рабочий процесс для сейсмической томографии:

```python
# 1. Создаем или загружаем скоростную модель
model = VelocityModel.from_json('initial_model.json')

# 2. Генерируем уплотненную geo-сетку для raytracing
geo = model.get_geo_grid(
    subdivision=3,           # 27x refinement (хороший баланс точности/памяти)
    interpolation='trilinear'  # гладкая интерполяция
)

# 3. Получаем numpy массивы для raytracing
vp_array = geo.vp  # shape: (n_x*3, n_y*3, n_z*3)
vs_array = geo.vs

# 4. Raytracing работает напрямую с массивами (очень быстро!)
for ray in rays:
    for step in ray.path:
        i, j, k = calculate_geo_indices(step)
        velocity = vp_array[i, j, k]  # O(1) доступ
        # ... tracing calculations ...

# 5. После инверсии обновляем скоростную модель
updates = calculate_velocity_corrections()
for (i, j, k), correction in updates.items():
    current_vp = model.get_vp(i, j, k)
    model.set_vp(i, j, k, current_vp + correction)

# 6. Новая итерация томографии
model.to_json('iteration_2.json')
geo_new = model.get_geo_grid(subdivision=3)  # пересоздаем geo-сетку
# ... повторяем raytracing ...
```

### Subdivision: баланс точности и памяти

- **subdivision=1**: geo-сетка = velocity model (500 ячеек → 500 ячеек)
- **subdivision=2**: 8x refinement (500 → 4,000 ячеек)
- **subdivision=3**: 27x refinement (500 → 13,500 ячеек) ⭐ **рекомендуется**
- **subdivision=4**: 64x refinement (500 → 32,000 ячеек)

Для модели 100×100×50 ячеек:
- subdivision=1: ~2 MB памяти на параметр
- subdivision=3: ~54 MB на параметр
- subdivision=4: ~128 MB на параметр

### Interpolation методы

```python
# Trilinear (по умолчанию) - гладкая интерполяция
geo = model.get_geo_grid(subdivision=3, interpolation='trilinear')

# Nearest neighbor - быстрее, но блочная
geo = model.get_geo_grid(subdivision=3, interpolation='nearest')

# Кастомная функция
def my_interpolation(values, i, j, k, di, dj, dk):
    """
    values: numpy array скоростей
    i, j, k: индекс базовой ячейки
    di, dj, dk: дробная часть [0..1] внутри ячейки
    """
    # Ваша логика интерполяции
    return interpolated_value

geo = model.get_geo_grid(subdivision=3, interpolation=my_interpolation)
```

## Работа с numpy массивами (скоростная модель)

Для эффективной работы можно напрямую обращаться к массивам скоростной модели:

```python
# Прямой доступ к массиву
vp_array = model.grid.vp  # numpy array shape (n_x, n_y, n_z)

# Модификация массива
vp_array[10:20, 15:25, 5:10] = 5000.0

# Применение функции к массиву
model.grid.vp = model.grid.vp * 1.1  # увеличить все скорости на 10%

# Создание из функции
import numpy as np
for i in range(model.geometry.n_x):
    for j in range(model.geometry.n_y):
        for k in range(model.geometry.n_z):
            depth = k * model.geometry.side_size
            vp = 1500 + 0.5 * depth  # v = v0 + k*z
            model.set_vp(i, j, k, vp)
```

## Формат JSON

Пример файла с геометрией:
```json
{
  "lon": 37.6173,
  "lat": 55.7558,
  "height": 0.0,
  "azimuth": 45.0,
  "side_size": 100.0,
  "n_x": 50,
  "n_y": 50,
  "n_z": 30
}
```

При сохранении с `include_data=True` добавляется секция `data`:
```json
{
  "lon": 37.6173,
  ...
  "data": {
    "vp": [[[...]]],
    "vs": [[[...]]]
  }
}
```

## Примеры использования

См. файлы с примерами:
- `example_usage.py` - базовые операции со скоростной моделью
- `example_geo_grid.py` - работа с geo-сеткой для raytracing

Основные сценарии:
- Создание модели из конфига
- Заполнение градиентом
- Работа с отдельными ячейками
- Работа с массивами напрямую
- Генерация geo-сетки с интерполяцией
- Типичный workflow томографии
- Сохранение и загрузка

## Тесты

Запуск тестов:
```bash
python test_velocity_model.py
```

## Roadmap

Реализовано:
- ✅ Генерация уплотненной geo-сетки с интерполяцией
- ✅ Подменяемые стратегии интерполяции (trilinear, nearest, custom)

Планируемые возможности:
- [ ] Улучшить сохранение/загрузку переиспользуя components/utilities.py (делать по папки на модель и сохранять скорости в .npz)
- [ ] Дополнительные параметры (учет анизотропии)
- [ ] Поддержка неравномерных сеток
- [ ] Кеширование geo-сетки (опционально) 

## Заметки по дизайну

- **Типы данных**: используется `float32` для экономии памяти
- **Разделение**: геометрия и данные разделены для гибкости
- **JSON**: данные хранятся как списки для совместимости, при загрузке конвертируются в numpy
- **Geo-сетка**: генерируется по требованию, не кешируется
- **Интерполяция**: trilinear по умолчанию, но можно подменить кастомной функцией