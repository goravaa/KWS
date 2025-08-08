# dataset.py
#
# This file defines the custom PyTorch Dataset for handling our audio data.
# It finds audio files, loads them, applies necessary transformations like
# padding/truncating and MFCC conversion, and serves them to the DataLoader.

import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and processing audio files.
    """
    def __init__(self, config, data_path, transform=None):
        """
        Initializes the dataset.

        Args:
            config (dict): The configuration dictionary.
            data_path (str): Path to the root dataset directory.
            transform (callable, optional): A function/transform to apply to the waveform.
        """
        self.config = config
        self.data_path = data_path
        self.transform = transform
        self.walker = []  # This will store tuples of (filepath, label_index)
        
        # Get the sorted list of labels from the folder names
        self.labels = sorted(os.listdir(data_path))
        # Create a mapping from label name to an integer index
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        # Walk through the directory and create a list of all files and their labels
        for label_name in self.labels:
            label_dir = os.path.join(data_path, label_name)
            if os.path.isdir(label_dir):
                for filename in os.listdir(label_dir):
                    if filename.lower().endswith('.wav'):
                        filepath = os.path.join(label_dir, filename)
                        self.walker.append((filepath, self.label_map[label_name]))

    def __len__(self):
        """Returns the total number of audio files in the dataset."""
        return len(self.walker)

    def __getitem__(self, idx):
        """
        Gets a single data point (an audio file and its label) from the dataset.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            A tuple of (waveform_tensor, label_index).
        """
        filepath, label = self.walker[idx]
        
        # Load the audio waveform
        waveform, sr = torchaudio.load(filepath)
        
        # Resample the audio if its sample rate doesn't match the target rate
        if sr != self.config['audio']['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, self.config['audio']['sample_rate'])
            waveform = resampler(waveform)

        # Ensure the waveform is mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate the waveform to a fixed length specified in the config
        target_len = int(self.config['audio']['sample_rate'] * self.config['audio']['clip_duration_s'])
        
        # Truncate if longer
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        # Pad with silence if shorter
        else:
            padding_needed = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

        # Apply the transformation (e.g., MFCC) if one is provided
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label

def get_mfcc_transform(config):
    """
    Creates an MFCC transformation object from the configuration settings.
    
    Args:
        config (dict): The configuration dictionary.
    
    Returns:
        A torchaudio.transforms.MFCC object.
    """
    audio_cfg = config['audio']
    # n_mels is often set to a value like 2*n_mfcc, it's a common practice
    n_mels = audio_cfg.get('n_mels', 2 * audio_cfg['n_mfcc'])

    return torchaudio.transforms.MFCC(
        sample_rate=audio_cfg['sample_rate'],
        n_mfcc=audio_cfg['n_mfcc'],
        melkwargs={
            'n_fft': audio_cfg['n_fft'],
            'hop_length': audio_cfg['hop_length'],
            'win_length': audio_cfg['win_length'],
            'n_mels': n_mels, 
        }
    )