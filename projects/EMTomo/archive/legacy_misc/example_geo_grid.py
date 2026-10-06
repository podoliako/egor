"""
Example: Using refined geo grid for raytracing.
"""
import numpy as np
from velocity_model import VelocityModel


print("=== Example: Geo Grid for Raytracing ===\n")

# Создаем простую скоростную модель
config = {
    'lon': 30.0,
    'lat': 60.0,
    'height': 0.0,
    'azimuth': 0.0,
    'side_size': 1000.0,  # 1 км на сторону
    'n_x': 10,
    'n_y': 10,
    'n_z': 5
}

model = VelocityModel.from_config(config)

# Заполняем градиентом по глубине
model.fill_linear_gradient('vp', top_value=2000.0, bottom_value=5000.0)
model.fill_linear_gradient('vs', top_value=1000.0, bottom_value=3000.0)

print(f"Velocity model: {model}")
print(f"Model cells: {model.geometry.n_x} x {model.geometry.n_y} x {model.geometry.n_z}")
print(f"Cell size: {model.geometry.side_size} m\n")


# Пример 1: Без subdivision (1:1)
print("--- Subdivision = 1 (no refinement) ---")
geo1 = model.get_geo_grid(subdivision=1)
print(f"Geo grid: {geo1}")
print(f"Geo cell size: {geo1.cell_size} m")
print(f"Vp at geo[0,0,0]: {geo1.vp[0, 0, 0]:.1f} m/s")
print(f"Same as model[0,0,0]: {model.get_vp(0, 0, 0):.1f} m/s\n")


# Пример 2: Subdivision = 2 (8x cells)
print("--- Subdivision = 2 (8x refinement) ---")
geo2 = model.get_geo_grid(subdivision=2)
print(f"Geo grid: {geo2}")
print(f"Total geo cells: {geo2.shape[0] * geo2.shape[1] * geo2.shape[2]:,}")
print(f"Geo cell size: {geo2.cell_size} m")
print(f"\nVp distribution in first velocity cell:")
print(f"  geo[0,0,0]: {geo2.vp[0, 0, 0]:.1f} m/s  (corner)")
print(f"  geo[1,0,0]: {geo2.vp[1, 0, 0]:.1f} m/s  (between cells)")
print(f"  geo[0,1,0]: {geo2.vp[0, 1, 0]:.1f} m/s  (between cells)")
print(f"  geo[1,1,0]: {geo2.vp[1, 1, 0]:.1f} m/s  (interpolated)\n")


# Пример 3: Subdivision = 3 (27x cells) - типичный для raytracing
print("--- Subdivision = 3 (27x refinement) ---")
geo3 = model.get_geo_grid(subdivision=3, interpolation='trilinear')
print(f"Geo grid: {geo3}")
print(f"Total geo cells: {geo3.shape[0] * geo3.shape[1] * geo3.shape[2]:,}")
print(f"Memory footprint: ~{geo3.vp.nbytes / 1024 / 1024:.2f} MB per parameter\n")


# Пример 4: Nearest neighbor (быстрее, но блочный)
print("--- Nearest neighbor interpolation ---")
geo_nearest = model.get_geo_grid(subdivision=3, interpolation='nearest')
print(f"Vp with trilinear at geo[1,1,1]: {geo3.vp[1, 1, 1]:.1f} m/s")
print(f"Vp with nearest at geo[1,1,1]: {geo_nearest.vp[1, 1, 1]:.1f} m/s\n")


# Пример 5: Кастомная интерполяция
print("--- Custom interpolation ---")

def gaussian_interpolation(values, i, j, k, di, dj, dk):
    """Gaussian-weighted interpolation (пример)."""
    n_x, n_y, n_z = values.shape
    i1 = min(i + 1, n_x - 1)
    j1 = min(j + 1, n_y - 1)
    k1 = min(k + 1, n_z - 1)
    
    # Гауссовы веса от центра ячейки
    sigma = 0.5
    weight_i = np.exp(-((di - 0.5)**2) / (2 * sigma**2))
    weight_j = np.exp(-((dj - 0.5)**2) / (2 * sigma**2))
    weight_k = np.exp(-((dk - 0.5)**2) / (2 * sigma**2))
    
    # Простое взвешивание (не совсем правильная Гауссиана, но для примера)
    v0 = values[i, j, k]
    v1 = values[i1, j1, k1] if (i1 != i or j1 != j or k1 != k) else v0
    
    return v0 * weight_i * weight_j * weight_k + v1 * (1 - weight_i * weight_j * weight_k)

geo_custom = model.get_geo_grid(subdivision=2, interpolation=gaussian_interpolation)
print(f"Custom interpolation works! Geo shape: {geo_custom.shape}\n")


# Пример 6: Имитация raytracing workflow
print("--- Typical raytracing workflow ---")

# 1. Генерируем гео-сетку один раз
print("1. Generate geo grid...")
geo = model.get_geo_grid(subdivision=3)

# 2. Получаем numpy массивы для быстрого доступа
vp_array = geo.vp
vs_array = geo.vs

# 3. Raytracing работает с массивами напрямую
print("2. Raytracing loop (simulated)...")
total_cells_visited = 0
for ray_id in range(100):  # 100 лучей
    # Примерный путь луча через сетку
    path_length = np.random.randint(50, 200)
    
    for step in range(path_length):
        # Вычисляем индексы (упрощенно)
        gi = np.random.randint(0, geo.shape[0])
        gj = np.random.randint(0, geo.shape[1])
        gk = np.random.randint(0, geo.shape[2])
        
        # Мгновенный доступ к скорости
        velocity = vp_array[gi, gj, gk]
        total_cells_visited += 1
        
        # ... raytracing calculations ...

print(f"   Processed {total_cells_visited:,} cell queries (instant access)")

# 4. После raytracing - обновляем модель
print("3. Update velocity model based on inversion...")
model.set_vp(5, 5, 2, 4500.0)  # Коррекция из томографии

# 5. Новая итерация - перегенерируем гео-сетку
print("4. Next iteration - regenerate geo grid...")
geo_new = model.get_geo_grid(subdivision=3)
print(f"   New geo grid ready: {geo_new}\n")


print("=== Example completed! ===")
print("\nKey takeaways:")
print("- Use subdivision=3 for good balance of accuracy and memory")
print("- Generate geo_grid once per tomography iteration")
print("- Raytracing works directly with numpy arrays (geo.vp, geo.vs)")
print("- After updating velocity model, regenerate geo_grid")