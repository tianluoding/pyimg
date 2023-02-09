import matplotlib.pyplot as plt
import numpy as np
import librosa
from scipy.fft import fft

def plot_spectrum(filepath):
    sig, sr = librosa.load(filepath)
    ft = fft(sig)
    magnitude = np.absolute(ft)
    frequency = np.linspace(0, sr, len(magnitude))
    l = magnitude.size
    plt.title("Spectrum")
    plt.ylabel("magnitude")
    plt.xlabel("frequency(kHz)")
    plt.plot(frequency[:int(l/2)]/1000, magnitude[:int(l/2)])
    plt.show()


# test
plot_spectrum("1-137-A-32.wav")
