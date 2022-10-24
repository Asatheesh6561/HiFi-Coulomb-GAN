import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate
    
mel_basis = {}
hann_window = {}


def get_dataset_filelist(args):
    training_files = ['clnsp' + str(i) for i in range(1, 500)]
    validation_files = ['clnsp' + str(i) for i in range(500, 617)]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, input_wavs_dir, output_wavs_dir, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax, split=True, n_cache_reuse=1, shuffle=True,
                 fmax_loss=None, device=None):
        self.audio_files = training_files
        self.input_wavs_dir = input_wavs_dir
        self.output_wavs_dir = output_wavs_dir
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = False

    def __getitem__(self, index):
        filename = self.audio_files[index]
        input_path = os.path.join(self.input_wavs_dir, filename + '.wav')
        output_path = os.path.join(self.output_wavs_dir, filename + '.wav')
        if self._cache_ref_count == 0:
            input_audio, sampling_rate = load_wav(input_path)
            input_audio = input_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                input_audio = normalize(input_audio) * 0.95
            self.cached_input_wav = input_audio

            output_audio, sampling_rate_ = load_wav(output_path)
            output_audio = output_audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                output_audio = normalize(output_audio) * 0.95
            self.cached_output_wav = output_audio

            if sampling_rate != self.sampling_rate or sampling_rate != sampling_rate_:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            input_audio = self.cached_input_wav
            output_audio = self.cached_output_wav
            self._cache_ref_count -= 1

        input_audio = torch.FloatTensor(input_audio)
        input_audio = input_audio.unsqueeze(0)

        output_audio = torch.FloatTensor(output_audio)
        output_audio = output_audio.unsqueeze(0)

        assert input_audio.size(1) == output_audio.size(1), "Inconsistent dataset length, unable to sampling"

        if self.split:
            if input_audio.size(1) >= self.segment_size:
                max_audio_start = input_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                input_audio = input_audio[:, audio_start:audio_start+self.segment_size]
                output_audio = output_audio[:, audio_start:audio_start+self.segment_size]
            else:
                input_audio = torch.nn.functional.pad(input_audio, (0, self.segment_size - input_audio.size(1)), 'constant')
                output_audio = torch.nn.functional.pad(output_audio, (0, self.segment_size - output_audio.size(1)), 'constant')
        return (input_audio.squeeze(0), output_audio.squeeze(0), filename)

    def __len__(self):
        return len(self.audio_files)