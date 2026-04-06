import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt

DATA_DIR = "augmented_datasets"
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
TARGET_SR = 16000
TARGET_LENGTH = 24064  # 1.504 секунди!
VAL_SPLIT = 0.2

# Автоматичний вибір відеокарти, якщо вона є
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {DEVICE}")



# КЛАС ДАТАСЕТУ
class NatoAlphabetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_paths = []

        raw_labels = []
        classes_set = set()

        for folder_name in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder_path): continue

            for file_name in os.listdir(folder_path):
                if not file_name.endswith('.wav'): continue

                # ВИТЯГУЄМО КЛАС ІЗ НАЗВИ ФАЙЛУ
                class_name = file_name[0]

                classes_set.add(class_name)
                self.file_paths.append(os.path.join(folder_path, file_name))
                raw_labels.append(class_name)

        self.classes = sorted(list(classes_set))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Перетворюємо текстові мітки (напр., 'a', 'b') на математичні індекси (0, 1)
        self.labels = [self.class_to_idx[lbl] for lbl in raw_labels]

        # Трансформація у Мел-спектрограму
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        audio_data, sr = sf.read(file_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

        # Перевірка частоти
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

        # Жорстка фіксація довжини (1.504 с)
        if waveform.shape[1] < TARGET_LENGTH:
            padding = TARGET_LENGTH - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > TARGET_LENGTH:
            waveform = waveform[:, :TARGET_LENGTH]

        # Перетворення у спектрограму
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Нормалізація
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        return mel_spec_db, label


#АРХІТЕКТУРА НЕЙРОМЕРЕЖІ
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
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
        x = self.fc_layers(x)
        return x


#ФУНКЦІЯ МАЛЮВАННЯ ГРАФІКІВ
def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 6))

    # Графік Loss (Втрати)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Навчання (Train Loss)')
    plt.plot(epochs, history['val_loss'], 'r--', label='Валідація (Val Loss)')
    plt.title('Графік функції втрат (Loss)', fontsize=14)
    plt.xlabel('Епоха', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Графік Accuracy (Точність)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Навчання (Train Acc)')
    plt.plot(epochs, history['val_acc'], 'g--', label='Валідація (Val Acc)')
    plt.title('Точність розпізнавання (Accuracy)', fontsize=14)
    plt.xlabel('Епоха', fontsize=12)
    plt.ylabel('Точність (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history_split.png', dpi=300)
    print("\nГрафік навчання збережено у файл 'training_history_split.png'")
    plt.show()


#ОСНОВНИЙ ЦИКЛ ТРЕНУВАННЯ ЗІ СПЛІТОМ
def train():
    full_dataset = NatoAlphabetDataset(DATA_DIR)

    if len(full_dataset) == 0:
        print(f"Помилка: Дані в папці {DATA_DIR} не знайдені!")
        return

    num_classes = len(full_dataset.classes)
    print(f"Знайдено класів: {num_classes} ({full_dataset.classes})")

    # РОЗБИВАЄМО ДАТАСЕТ (80% Навчання / 20% Валідація)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Всього аудіофайлів: {len(full_dataset)}")
    print(f"  - Для навчання: {train_size}")
    print(f"  - Для тестування: {val_size}")

    # Створюємо два завантажувачі
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AudioCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Оновлений словник для логів
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        #НАВЧАННЯ
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(spectrograms)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_running_loss / len(train_loader)
        epoch_train_acc = (train_correct / train_total) * 100

        #ВАЛІДАЦІЯ
        model.eval()  # Вимикаємо Dropout для чесного тесту
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Вимикаємо обчислення градієнтів (економить пам'ять)
            for spectrograms, labels in val_loader:
                spectrograms, labels = spectrograms.to(DEVICE), labels.to(DEVICE)

                outputs = model(spectrograms)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = (val_correct / val_total) * 100

        # Зберігаємо історію
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Епоха [{epoch + 1:03d}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")

    torch.save(model.state_dict(), "nato_audio_model.pth")
    print("\nТренування завершено! Ваги збережено у 'nato_audio_model.pth'")

    plot_training_history(history)


if __name__ == "__main__":
    train()