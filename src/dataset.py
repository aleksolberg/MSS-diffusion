import librosa
import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SpectrogramDataset(Dataset):
    """Dataset returning spectrograms from audio in a folder."""

    def __init__(self,
                 target_dir,
                 condition_dir=None,
                 transforms=None,
                 paired=True,
                 return_pair=False,
                 return_mask = False,
                 sample_rate = 11025,
                 window_length=511,
                 hop_length=128,
                 out_channels = 3,
                 out_shape = (256, 256)):
        self.target_dir = target_dir
        self.condition_dir = condition_dir
        self.transforms = transforms
        self.return_pair=return_pair
        self.return_mask = return_mask

        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.out_channels = out_channels
        self.out_shape=out_shape
        
        # check files
        supported_formats=['wav', 'mp3']        
        self.files=[el for el in os.listdir(self.target_dir) if el.split('.')[-1] in supported_formats]
        if condition_dir is not None:
            self.condition_files=[el for el in os.listdir(self.condition_dir) if el.split('.')[-1] in supported_formats]
            assert self.files == self.condition_files, 'Filenames do not match'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        target_audio, sr = librosa.load(os.path.join(self.target_dir, self.files[idx]), sr=self.sample_rate)
        target_spectrogram = librosa.stft(target_audio, n_fft=self.window_length, hop_length=self.hop_length)

        target_amplitude = torch.from_numpy(librosa.amplitude_to_db(np.abs(target_spectrogram), ref=1e-5)/255)

        # Shaping tensor
        if self.out_shape is not None:
            target_amplitude = target_amplitude[:self.out_shape[0], :self.out_shape[1]]
        target = target_amplitude.unsqueeze(0).repeat(self.out_channels, 1, 1)

        if self.condition_dir is not None:
            condition_audio, sr = librosa.load(os.path.join(self.condition_dir, self.files[idx]), sr=self.sample_rate)
            condition_spectrogram = librosa.stft(condition_audio, n_fft=self.window_length, hop_length=self.hop_length)

            condition_amplitude = torch.from_numpy(librosa.amplitude_to_db(np.abs(condition_spectrogram), ref=1e-5)/255)

            # Shaping tensor
            if self.out_shape is not None:
                condition_amplitude = condition_amplitude[:self.out_shape[0], :self.out_shape[1]]
            condition = condition_amplitude.unsqueeze(0).repeat(self.out_channels, 1, 1)
        
            if self.return_mask:
                target = self.get_ideal_mask(target, condition)

        if self.return_pair:
            return condition, target
        else:
            return target
    
    def get_phase(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        dir = self.condition_dir if self.condition_dir is not None else self.target_dir

        audio, sr = librosa.load(os.path.join(dir, self.files[idx]), sr=self.sample_rate)
        spectrogram = librosa.stft(audio, n_fft=self.window_length, hop_length=self.hop_length)
        phase = torch.from_numpy(np.angle(spectrogram))
        if self.out_shape is not None:
            phase = phase[:self.out_shape[0], :self.out_shape[1]]
        phase = phase.unsqueeze(0).repeat(self.out_channels, 1, 1)

        return phase
        
    def to_audio(self, amplitude, phase):
        # Taking mean because of how some pretrained autoencoders require 3 channels
        amplitude = torch.mean(amplitude, dim=0) * 255
        amplitude = librosa.db_to_amplitude(amplitude, ref=1e-5).numpy()
        phase = torch.mean(phase, dim=0).numpy()

        spectrogram = amplitude*np.exp(1j*phase)
        audio = librosa.istft(spectrogram, hop_length=self.hop_length, n_fft=self.window_length)

        return audio


    def save_audio(self, spectrogram, phase, name='test.wav', output_folder = 'results'):
        audio = self.to_audio(spectrogram, phase)

        sf.write(os.path.join(output_folder, name), audio, self.sample_rate)
        return audio
    
    def get_ideal_mask(self, source_spectrogram, mix_spectrogram):
        mask = torch.div(source_spectrogram, torch.max(source_spectrogram, mix_spectrogram))
        return mask