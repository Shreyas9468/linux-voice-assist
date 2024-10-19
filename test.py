import sounddevice as sd
import numpy as np

duration = 4  # seconds
fs = 16000  # Sample rate
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print(f"Audio recorded. Shape: {recording.shape}, Max value: {np.max(recording)}, Min value: {np.min(recording)}")
