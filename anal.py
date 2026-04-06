import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

TEST_DIR = "augmented_datasets"
MODEL_PATH = "nato_audio_model.pth"
TARGET_SR = 16000
TARGET_LENGTH = 24064

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class NatoTestDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = []
        raw_labels = []
        classes_set = set()

        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.wav'): continue
                class_name = file_name[0]  # Беремо першу літеру!
                classes_set.add(class_name)
                self.file_paths.append(os.path.join(folder_path, file_name))
                raw_labels.append(class_name)

        self.classes = sorted(list(classes_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.labels = [self.class_to_idx[lbl] for lbl in raw_labels]
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=TARGET_SR, n_fft=1024, hop_length=512,
                                                                  n_mels=64)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio_data, sr = sf.read(file_path)
        if len(audio_data.shape) > 1: audio_data = audio_data.mean(axis=1)
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        if sr != TARGET_SR: waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

        if waveform.shape[1] < TARGET_LENGTH:
            waveform = torch.nn.functional.pad(waveform, (0, TARGET_LENGTH - waveform.shape[1]))
        elif waveform.shape[1] > TARGET_LENGTH:
            waveform = waveform[:, :TARGET_LENGTH]

        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        # Наша магічна нормалізація!
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        return mel_spec_db, label



def evaluate_model():
    dataset = NatoTestDataset(TEST_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    num_classes = len(dataset.classes)

    model = AudioCNN(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # Переводимо модель у режим тестування (вимикаємо Dropout)

    all_preds = []
    all_labels = []

    with torch.no_grad():  # Вимикаємо градієнти для швидкості
        for spectrograms, labels in dataloader:
            spectrograms = spectrograms.to(DEVICE)
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Розрахунок метрик
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nЗагальна точність на наборі даних: {acc * 100:.2f}%")

    # Побудова Матриці сплутувань
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)

    plt.title('Матриця сплутувань (Confusion Matrix)', fontsize=16)
    plt.xlabel('Передбачений моделлю клас', fontsize=12)
    plt.ylabel('Справжній клас', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"Помилка: Файл моделі '{MODEL_PATH}' не знайдено. Спочатку завершіть тренування!")
    else:
        evaluate_model()