import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio
import torchaudio.transforms as T

SAMPLE_WAV_SPEECH_PATH = "1-137-A-32.wav"


def _get_sample(path, resample=None):
    effects = [["remix", "1"]]
    if resample:
        effects.extend(
            [
                ["lowpass", f"{resample // 2}"],
                ["rate", f"{resample}"],
            ]
        )
    return torchaudio.load(path)


def get_speech_sample(*, resample=None):
    return _get_sample(SAMPLE_WAV_SPEECH_PATH, resample=resample)


def get_spectrogram(
        n_fft=1024,
        win_len=None,
        hop_len=None,
        power=2.0,
):
    waveform, sample_rate = get_speech_sample()
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    melspectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=256
    )
    return spectrogram(waveform), melspectrogram(waveform)


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    # plt.show(block=False)
    plt.show()


# example mfcc spectrogram and melspectrogram
mfcc_transform = T.MFCC(
    sample_rate=44100,
    n_mfcc=256,
    melkwargs={
        "n_fft": 1024,
        "n_mels": 256,
    },
)

wave, _ = torchaudio.load(SAMPLE_WAV_SPEECH_PATH)
mfcc = mfcc_transform(wave)
spec, melspec = get_spectrogram(power=None)

plot_spectrogram(torch.abs(spec[0]), title="Spectrogram", aspect="equal", xmax=431)
plot_spectrogram(torch.abs(melspec[0]), title="MelSpectrogram", aspect="equal", xmax=431)
plot_spectrogram(torch.abs(mfcc[0]), title="MFCC", aspect="equal", xmax=431)
