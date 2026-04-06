import os
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np


MODEL_PATH = "nato_audio_model.pth"
DATA_DIR = "datasets"
TARGET_SR = 16000
TARGET_LENGTH = 24064  #1.504 секунди

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#шлях до довгого аудіофайлу, який розпізнати
TEST_AUDIO_FILE = "example1.ogg"


class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)



def process_and_predict(file_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Помилка: Файл {MODEL_PATH} не знайдено! Спочатку натренуйте модель.")
        return
    if not os.path.exists(file_path):
        print(f"Помилка: Аудіофайл {file_path} не знайдено!")
        return

    classes_set = set()
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.isdir(folder_path): continue
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'): classes_set.add(file_name[0])

    classes = sorted(list(classes_set))
    num_classes = len(classes)

    # Завантажуємо навчену модель
    model = AudioCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # Вимикаємо Dropout

    # Трансформації
    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR, n_fft=1024, hop_length=512,
                                                         n_mels=64).to(DEVICE)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(DEVICE)

    # Завантажуємо аудіо через librosa (автоматично робить моно і потрібну частоту)
    audio_data, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # top_db=30: все, що тихіше за пікову гучність на 30 децибел, вважається тишею.
    # Якщо слова зливаються, зменшіть 30 до 20. Якщо ковтає шматки — збільште до 40.
    intervals = librosa.effects.split(audio_data, top_db=40)

    print(f" Знайдено окремих слів/звуків: {len(intervals)}\n")

    predictions = []

    # Обробляємо кожне знайдене слово
    for i, (start_idx, end_idx) in enumerate(intervals):
        # Вирізаємо слово з масиву
        word_audio = audio_data[start_idx:end_idx]

        # Перетворюємо у тензор PyTorch [1, N]
        waveform = torch.tensor(word_audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Фіксуємо ідеальну довжину (1.504 с)
        if waveform.shape[1] < TARGET_LENGTH:
            padding = TARGET_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > TARGET_LENGTH:
            waveform = waveform[:, :TARGET_LENGTH]

        # Робимо спектрограму + Нормалізація
        mel_spec = mel_transform(waveform)
        mel_spec_db = amplitude_to_db(mel_spec)
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        # Батч-вимір: [1, 1, 64, 48]
        mel_spec_db = mel_spec_db.unsqueeze(0)

        # Прогноз
        with torch.no_grad():
            output = model(mel_spec_db)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item() * 100

        predicted_class = classes[predicted_idx]
        predictions.append(predicted_class)

        # Переводимо індекси в секунди для красивого виводу
        start_sec = start_idx / TARGET_SR
        end_sec = end_idx / TARGET_SR

        print(f"[{start_sec:.2f}c - {end_sec:.2f}c] -> {predicted_class.upper()} (Впевненість: {confidence:.1f}%)")

    print("\n" + "=" * 50)
    print("ФІНАЛЬНА РОЗПІЗНАНА ПОСЛІДОВНІСТЬ:")
    print(" ".join([p.upper() for p in predictions]))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    process_and_predict(TEST_AUDIO_FILE)