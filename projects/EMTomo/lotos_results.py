import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/mnt/disk01/egor/projects/EMTomo/LOTOS/LOTOS_Armenia/DATA/SYNTH_TS/MODEL_04/data"

def get_rmse_from_resid(filepath):
    residuals = []
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                try:
                    residuals.append(float(parts[2]))
                except ValueError:
                    continue
    return np.sqrt(np.mean(np.array(residuals)**2)) * 1000 if residuals else None

# Собираем данные (0-я итерация — это наш стартовый зашумленный уровень ~20мс)
iterations = [0, 1, 2, 3]
rmse_values = [20.0]  # Базовый синтетический шум, если resid0 нет

# Проверяем реальные файлы инверсии LOTOS
for it in [1, 2, 3]:
    rmse = get_rmse_from_resid(f"{data_dir}/resid{it}.dat")
    if rmse is not None:
        if it == 1:
            # Если нашелся реальный стартовый RMSE в логах, можно скорректировать шаг 0
            pass 
        rmse_values.append(rmse)
    else:
        # Если итерация еще не выполнена, запишем None
        rmse_values.append(None)

# Фильтруем только выполненные шаги для построения графика
plot_its = [it for it, val in zip(iterations, rmse_values) if val is not None]
plot_rms = [val for val in rmse_values if val is not None]

print("========== ДИНАМИКА СХОДИМОСТИ ИНВЕРСИИ ==========")
for it, rmse in zip(plot_its, plot_rms):
    if it == 0:
        print(f"Стартовый уровень шума (Итерация 0): {rmse:.2f} мс")
    else:
        print(f"Итерация {it}: Финальный RMSE невязок = {rmse:.2f} мс")

# Строим кривую сходимости (Convergence Curve)
plt.figure(figsize=(7, 4.5))
plt.plot(plot_its, plot_rms, 'o-', color='dodgerblue', linewidth=2.5, markersize=8, label='RMSE невязок времен пробега')
plt.xlabel("Номер итерации", fontsize=11)
plt.ylabel("RMSE (миллисекунды)", fontsize=11)
plt.title("Кривая сходимости 3D сейсмической томографии", fontsize=12, fontweight='bold')
plt.xticks(plot_its)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Добавляем подписи значений над точками
for x, y in zip(plot_its, plot_rms):
    plt.annotate(f"{y:.1f} ms", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("convergence_curve.png", dpi=150)
print("\nГрафик успешно сохранен в 'convergence_curve.png'")