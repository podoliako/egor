#!/bin/bash

RUNS_DIR="runs"
# Дата отсечения (все, что раньше этой даты)
THRESHOLD_DATE="20260409"

# Ищем только папки run_ в корневом каталоге
find "$RUNS_DIR" -maxdepth 1 -type d -name "run_*" | while read run_dir; do
    # Извлекаем дату из имени: берем вторую часть после run_ (YYYYMMDD)
    folder_name=$(basename "$run_dir")
    folder_date=$(echo "$folder_name" | cut -d'_' -f2)

    # Проверяем, что это число и оно меньше нашей даты
    if [[ "$folder_date" =~ ^[0-9]+$ ]] && [ "$folder_date" -lt "$THRESHOLD_DATE" ]; then
        echo ">>> Проверка рана: $folder_name (дата $folder_date)"

        # Ищем misfit.npy во всей структуре внутри этого рана
        # Это покроет и event_*/misfit.npy и event_*/weight_*/misfit.npy
        find "$run_dir" -type f -name "misfit.npy" | while read misfit_file; do
            # Проверяем путь, чтобы случайно не удалить лишнего 
            # (только внутри папок event_ или weight_)
            if [[ "$misfit_file" =~ "event_" ]] || [[ "$misfit_file" =~ "weight_" ]]; then
                echo "Удаляю: $misfit_file"
                rm "$misfit_file"
            fi
        done
    fi
done

echo "Готово!"