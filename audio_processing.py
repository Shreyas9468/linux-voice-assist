import numpy as np
from scipy import signal

def process_audio(audio, fs):
    # Noise reduction
    noise_reduced = reduce_noise(audio)

    # Normalization
    normalized = normalize(noise_reduced)

    # Apply bandpass filter
    lowcut = 300
    highcut = 3000
    filtered = butter_bandpass_filter(normalized, lowcut, highcut, fs, order=6)

    return filtered

def reduce_noise(audio):
    # Simple noise reduction using spectral gating
    noise_sample = audio[:int(len(audio) * 0.1)]  # Use first 10% as noise sample
    noise_profile = np.mean(np.abs(noise_sample))
    return np.where(np.abs(audio) < noise_profile * 2, 0, audio)

def normalize(audio):
    return audio / np.max(np.abs(audio))

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data)
    return y
