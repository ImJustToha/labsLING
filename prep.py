import os
import random
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F_audio
import torchaudio.transforms as T


INPUT_DIR = "datasets"
OUTPUT_DIR = "augmented_datasets"
TARGET_SR = 16000
TARGET_LENGTH = 24064  # підібрані 1.504 секунди
NUM_VARIATIONS = 25


def fit_to_length_stretching(waveform):
    """Якщо слово задовге, акуратно прискорює його (Phase Vocoder)"""
    current_len = waveform.shape[1]
    if current_len > TARGET_LENGTH:
        rate = current_len / TARGET_LENGTH
        spec = T.Spectrogram(n_fft=1024, power=None)(waveform)
        stretch = T.TimeStretch(n_freq=1024 // 2 + 1, fixed_rate=rate)
        waveform = T.InverseSpectrogram(n_fft=1024)(stretch(spec))
        if waveform.shape[1] > TARGET_LENGTH:
            waveform = waveform[:, :TARGET_LENGTH]
    return waveform


def random_padding(waveform):
    """Випадково зсуває слово в межах вікна (TARGET_LENGTH)"""
    current_len = waveform.shape[1]
    if current_len < TARGET_LENGTH:
        total_pad = TARGET_LENGTH - current_len
        left_pad = int(total_pad * random.random())
        right_pad = total_pad - left_pad
        waveform = torch.nn.functional.pad(waveform, (left_pad, right_pad))
    else:
        waveform = waveform[:, :TARGET_LENGTH]
    return waveform


def apply_stochastic_pipeline(waveform, sr):

    aug_wave = waveform.clone()
    # 1. Випадкова зміна тону (Ймовірність 50%)
    if random.random() < 0.5:
        n_steps = random.randint(-4, 4)
        if n_steps != 0: aug_wave = F_audio.pitch_shift(aug_wave, sr, n_steps)

    # 2. Випадкова гучність (Ймовірність 80%)
    if random.random() < 0.8:
        gain = random.uniform(0.3, 1.8)
        aug_wave = aug_wave * gain

    # 3. Ефект рації / Bandpass (Ймовірність 30%)
    if random.random() < 0.3:
        cutoff_high = random.uniform(3000, 4500)
        aug_wave = F_audio.lowpass_biquad(aug_wave, sr, cutoff_high)
        if random.random() < 0.5:
            cutoff_low = random.uniform(300, 600)
            aug_wave = F_audio.highpass_biquad(aug_wave, sr, cutoff_low)

    # 4. Перевантаження / Кліппінг (Ймовірність 20%)
    if random.random() < 0.2:
        overdrive = random.uniform(2.0, 5.0)
        aug_wave = torch.clamp(aug_wave * overdrive, min=-1.0, max=1.0)

    # 5. Луна / Реверберація (Ймовірність 30%)
    if random.random() < 0.3:
        delay_ms = random.randint(20, 100)
        delay_samples = int(sr * (delay_ms / 1000.0))
        decay = random.uniform(0.1, 0.4)
        padding = torch.zeros(aug_wave.shape[0], delay_samples)
        echo = torch.cat((padding, aug_wave), dim=1)[:, :aug_wave.shape[1]]
        aug_wave = aug_wave + (echo * decay)

    # 6. Глітч / Втрата пакетів (Ймовірність 25%)
    if random.random() < 0.25:
        glitch_ms = random.randint(20, 60)
        glitch_samples = int(sr * (glitch_ms / 1000.0))
        if aug_wave.shape[1] > glitch_samples:
            start = random.randint(0, aug_wave.shape[1] - glitch_samples)
            aug_wave[:, start: start + glitch_samples] = 0.0

    # 7. Білий шум на фон (Ймовірність 80%)
    if random.random() < 0.8:
        noise_level = random.uniform(0.001, 0.02)
        aug_wave = aug_wave + noise_level * torch.randn_like(aug_wave)

    return torch.clamp(aug_wave, min=-1.0, max=1.0)


def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_files = 0

    for folder_name in os.listdir(INPUT_DIR):
        input_folder = os.path.join(INPUT_DIR, folder_name)
        if not os.path.isdir(input_folder): continue

        output_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(input_folder):
            if not file_name.endswith('.wav'): continue
            file_path = os.path.join(input_folder, file_name)

            try:
                audio_data, sr = sf.read(file_path)
            except Exception as e:
                print(f"Пропущено {file_name} через помилку читання: {e}")
                continue

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)

            if sr != TARGET_SR:
                waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

            waveform = fit_to_length_stretching(waveform)
            base_name = file_name.replace('.wav', '')

            clean_wave = random_padding(waveform.clone())
            # Перетворюємо тензор [1, N] у numpy масив [N]
            sf.write(os.path.join(output_folder, f"{base_name}_orig.wav"), clean_wave.squeeze(0).numpy(), TARGET_SR)
            total_files += 1

            for i in range(NUM_VARIATIONS):
                mutated = apply_stochastic_pipeline(waveform, TARGET_SR)
                mutated = random_padding(mutated)

                out_name = f"{base_name}_mut_{i + 1:02d}.wav"
                # Зберігаємо мутанта
                sf.write(os.path.join(output_folder, out_name), mutated.squeeze(0).numpy(), TARGET_SR)
                total_files += 1

    print(f"\nГотово! Синтезовано {total_files} файлів.")


if __name__ == "__main__":
    generate()