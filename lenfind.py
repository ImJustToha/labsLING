import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "datasets"  # Папка з вашими оригінальними даними


def analyze_audio_lengths():
    durations = []

    # Проходимо по всіх папках і файлах
    for folder_name in os.listdir(INPUT_DIR):
        folder_path = os.path.join(INPUT_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.wav'):
                continue

            file_path = os.path.join(folder_path, file_name)
            try:
                audio_data, sr = sf.read(file_path)

                # Беремо довжину по першій осі (кількість відліків)
                frames = audio_data.shape[0]
                duration = frames / sr
                durations.append(duration)
            except Exception as e:
                print(f"Помилка читання {file_name}: {e}")

    if not durations:
        print(f"Помилка: Аудіофайли в папці '{INPUT_DIR}' не знайдені або пошкоджені!")
        return


    durations = np.array(durations)
    mean_len = np.mean(durations)
    min_len = np.min(durations)
    max_len = np.max(durations)
    p95_len = np.percentile(durations, 95)

    target_len = 1.504
    percentile_of_target = np.mean(durations <= target_len) * 100

    print(f"\n--- Статистика довжини аудіо ---")
    print(f"Всього проаналізовано файлів: {len(durations)}")
    print(f"Мінімальна довжина:  {min_len:.3f} с")
    print(f"Максимальна довжина: {max_len:.3f} с")
    print(f"Середня довжина:     {mean_len:.3f} с")
    print(f"95-й перцентиль:     {p95_len:.3f} с")
    print(f"--------------------------------")
    print(f"Відсоток файлів, коротших за {target_len} с: {percentile_of_target:.2f}%")

    # ==========================================
    # ВІЗУАЛІЗАЦІЯ
    # ==========================================
    plt.figure(figsize=(10, 5))

    plt.hist(durations, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

    plt.axvline(target_len, color='purple', linestyle='-', linewidth=3,
                label=f'Ширина W=48 ({target_len}с): вміщує {percentile_of_target:.1f}%')
    plt.axvline(p95_len, color='red', linestyle='dashed', linewidth=2,
                label=f'95% файлів коротші за {p95_len:.2f}с')
    plt.axvline(mean_len, color='green', linestyle='dotted', linewidth=2,
                label=f'Середнє: {mean_len:.2f}с')

    plt.title('Розподіл довжини аудіозаписів у датасеті', fontsize=14)
    plt.xlabel('Довжина запису (секунди)', fontsize=12)
    plt.ylabel('Кількість файлів', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_audio_lengths()