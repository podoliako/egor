"""
Example usage of the velocity model.
"""
import numpy as np
from velocity_model import VelocityModel, GridGeometry, VelocityGrid


# Пример 1: Создание модели из конфига
print("=== Example 1: Create model from config ===")
config = {
    'lon': 37.6173,  # Москва, например
    'lat': 55.7558,
    'height': 0.0,
    'azimuth': 45.0,  # 45 градусов от севера
    'side_size': 100.0,  # 100 метров на сторону
    'n_x': 50,
    'n_y': 50,
    'n_z': 30
}

model = VelocityModel.from_config(config)
print(model)


# Пример 2: Заполнение линейным градиентом
print("\n=== Example 2: Fill with linear gradient ===")
# Vp от 1500 м/с на поверхности до 6000 м/с на глубине
model.fill_linear_gradient('vp', top_value=1500.0, bottom_value=6000.0)

# Vs от 800 м/с до 3500 м/с
model.fill_linear_gradient('vs', top_value=800.0, bottom_value=3500.0)

print(f"Vp at surface (0,0,0): {model.get_vp(0, 0, 0):.1f} m/s")
print(f"Vp at depth (0,0,29): {model.get_vp(0, 0, 29):.1f} m/s")
print(f"Vs at surface (0,0,0): {model.get_vs(0, 0, 0):.1f} m/s")
print(f"Vs at depth (0,0,29): {model.get_vs(0, 0, 29):.1f} m/s")


# Пример 3: Изменение отдельных ячеек
print("\n=== Example 3: Modify individual cells ===")
# Добавим аномалию - низкоскоростную зону
for i in range(20, 30):
    for j in range(20, 30):
        for k in range(10, 15):
            model.set_vp(i, j, k, 2000.0)  # Низкая скорость
            model.set_vs(i, j, k, 1000.0)

print(f"Vp in anomaly (25,25,12): {model.get_vp(25, 25, 12):.1f} m/s")
print(f"Vp outside anomaly (10,10,12): {model.get_vp(10, 10, 12):.1f} m/s")


# Пример 4: Работа с массивами напрямую
print("\n=== Example 4: Direct array manipulation ===")
# Создаём случайные возмущения
perturbation = np.random.normal(0, 100, size=model.grid.vp.shape)
vp_perturbed = model.grid.vp + perturbation
model.set_vp_array(vp_perturbed)

print(f"Mean Vp: {model.grid.vp.mean():.1f} m/s")
print(f"Std Vp: {model.grid.vp.std():.1f} m/s")


# Пример 5: Сохранение и загрузка
print("\n=== Example 5: Save and load ===")
# Сохраняем с данными
model.to_json('/home/claude/model_full.json', include_data=True)
print("Model saved to model_full.json")

# Сохраняем только геометрию
model.to_json('/home/claude/model_geometry_only.json', include_data=False)
print("Geometry saved to model_geometry_only.json")

# Загружаем обратно
loaded_model = VelocityModel.from_json('/home/claude/model_full.json')
print(f"\nLoaded model: {loaded_model}")
print(f"Vp at (25,25,12) in loaded model: {loaded_model.get_vp(25, 25, 12):.1f} m/s")


# Пример 6: Создание модели с нуля программно
print("\n=== Example 6: Create model programmatically ===")
geometry = GridGeometry(
    lon=30.0,
    lat=60.0,
    height=-100.0,  # Начинаем ниже уровня земли
    azimuth=0.0,     # На север
    side_size=50.0,
    n_x=100,
    n_y=100,
    n_z=50
)

# Создаём свою функцию распределения скорости
grid = VelocityGrid((100, 100, 50))

# Пример: скорость зависит от глубины по формуле
# v(z) = v0 + k * depth^2
for i in range(100):
    for j in range(100):
        for k in range(50):
            depth_m = k * 50.0  # глубина в метрах
            vp = 1500 + 0.001 * depth_m**2
            vs = 0.6 * vp
            grid.set_vp(i, j, k, vp)
            grid.set_vs(i, j, k, vs)

custom_model = VelocityModel(geometry, grid)
print(f"Custom model: {custom_model}")
print(f"Vp at 1000m depth (0,0,20): {custom_model.get_vp(0, 0, 20):.1f} m/s")


print("\n=== All examples completed successfully! ===")