import numpy as np


def arrange_duration(audio, duration, samplerate):
    target_n_samples = int(duration * samplerate)
    
    if target_n_samples > audio.shape[-1]:
        audio = np.concatenate((audio, np.zeros((audio.shape[0], target_n_samples - audio.shape[-1]))), axis=-1)
    elif target_n_samples < audio.shape[-1]:
        audio = audio[:, :target_n_samples]
        
    return audio


def amplify_noise(signal, noise, target_snr_db):
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    target_snr_linear = 10 ** (target_snr_db / 10)
    required_noise_power = signal_power / target_snr_linear
    
    amplification_factor = np.sqrt(required_noise_power / noise_power)
    amplified_noise = noise * amplification_factor
    
    return amplified_noise


def add_gaussian_noise(signal, snr_db):
    signal_power = np.mean(np.square(signal))
    
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    
    noisy_signal = signal + noise
    
    return noisy_signal, noise