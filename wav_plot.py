import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np


def plot_wav(filepath):
    sig, sr = sf.read(filepath)
    time = np.arange(0, len(sig)) * (1.0 / sr)

    plt.title("Waveform")
    plt.ylabel("magnitude")
    plt.xlabel("time(s)")
    plt.plot(time, sig)
    plt.show()


# test
plot_wav("1-137-A-32.wav")
