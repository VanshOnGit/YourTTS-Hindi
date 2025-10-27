import struct
from pathlib import Path
from typing import Optional, Union

# import webrtcvad
import librosa
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from TTS.vc.modules.freevc.speaker_encoder.hparams import *

int16_max = (2**15) - 1


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):
    """
    Applies the preprocessing operations used in training the Speaker Encoder to a waveform
    either on disk or in memory. The waveform will be resampled to match the data hyperparameters.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not
    just .wav), either the waveform as a numpy array of floats.
    :param source_sr: if passing an audio waveform, the sampling rate of the waveform before
    preprocessing. After preprocessing, the waveform's sampling rate will match the data
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected and
    this argument will be ignored.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(fpath_or_wav, sr=None)
    else:
        wav = fpath_or_wav

    # Resample the wav if needed
    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(y=wav, orig_sr=source_sr, target_sr=sampling_rate)


    # Apply the preprocessing: normalize volume and shorten long silences
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)

    return wav


def wav_to_mel_spectrogram(wav):
    """
    Derives a mel spectrogram ready to be used by the encoder from a preprocessed audio waveform.
    Note: this not a log-mel spectrogram.
    """
    frames = librosa.feature.melspectrogram(
        y=wav,
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
    )
    return frames.astype(np.float32).T


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    dBFS_change = target_dBFS - 10 * np.log10(np.mean(wav**2))
    if (dBFS_change < 0 and increase_only) or (dBFS_change > 0 and decrease_only):
        return wav
    return wav * (10 ** (dBFS_change / 20))


def trim_long_silences(wav, vad_window_length=30, vad_moving_average_width=8, vad_max_silence_length=6):
    # Compute the VAD
    samples_per_window = int(vad_window_length * sampling_rate / 1000)

    # Trim to length that's multiple of window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    wav_frames = wav.reshape((-1, samples_per_window))

    energies = np.array([
        np.sqrt(np.mean(frame**2)) for frame in wav_frames
    ])
    threshold = np.max(energies) * 0.1
    voiced_frames = energies > threshold

    # Apply moving average smoothing
    mask = np.convolve(
        voiced_frames.astype(np.float32),
        np.ones(vad_moving_average_width) / vad_moving_average_width,
        mode="same"
    ) > 0.5

    # Binary dilation
    mask = binary_dilation(mask, np.ones(vad_max_silence_length + 1))
    mask = np.repeat(mask, samples_per_window)

    return wav[mask]
