# augment_data.py (Final Version, Corrected for AttributeError)
#
# This script is tailored to your installed library version.
# FIX: Removed the '.samples' call and 'output_type' arguments to match the
#      actual behavior of your library version, which returns a Tensor directly.

import torch
import torchaudio
import os
import random
from torch_audiomentations import Compose, AddBackgroundNoise, PitchShift, Shift

def augment_and_save(input_dir, output_dir, noise_dir, num_augmentations, sample_rate=16000):
    """
    Takes an input directory of audio files, applies random augmentations,
    and saves them to the output directory.
    """
    if num_augmentations == 0:
        print(f"Skipping augmentation for {os.path.basename(input_dir)} (0 augmentations requested).")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Defining augmentation pipeline with noise from: {noise_dir}")

    # --- Define Augmentation Pipeline (Corrected for your library version) ---
    augment = Compose([
        AddBackgroundNoise(
            background_paths=noise_dir,
            min_snr_in_db=3.0,
            max_snr_in_db=15.0,
            p=0.7,
            sample_rate=sample_rate
        ),
        Shift(
            min_shift=-0.1,
            max_shift=0.1,
            p=0.5,
            sample_rate=sample_rate
        ),
        PitchShift(
            min_transpose_semitones=-2.0,
            max_transpose_semitones=2.0,
            p=0.5,
            sample_rate=sample_rate
        )
    ])

    print(f"Starting augmentation for '{os.path.basename(input_dir)}' -> creating {num_augmentations} new files per original...")

    # --- Process each file in the input directory ---
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            filepath = os.path.join(input_dir, filename)
            try:
                waveform, sr = torchaudio.load(filepath)
            except Exception as e:
                print(f"Warning: Could not load file {filepath}. Skipping. Error: {e}")
                continue

            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            for i in range(num_augmentations):
                # Apply the augmentation pipeline.
                # FIX: Removed '.samples' as the output is a Tensor directly.
                augmented_waveform = augment(samples=waveform.unsqueeze(0), sample_rate=sample_rate).squeeze(0)

                # Create a new filename and save the augmented audio
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.wav"
                output_path = os.path.join(output_dir, output_filename)
                torchaudio.save(output_path, augmented_waveform, sample_rate)

    print(f"Finished augmenting files and saved them to {output_dir}")


if __name__ == '__main__':
    # --- Main Configuration ---
    original_base_folder = "dataset"
    output_base_folder = "dataset_augmented"
    noise_folder = os.path.join(original_base_folder, "_background_noise_")

    # **TUNE YOUR AUGMENTATIONS HERE**
    augmentation_plan = {
        'lap': 90,
        'reset': 80,
        'start': 70,
        'stop': 20
    }

    # --- Execution Loop ---
    print("Starting dataset balancing process...")


    if not os.path.isdir(noise_folder):
        print(f"FATAL ERROR: Noise directory not found at '{noise_folder}'. Please check your paths.")
    else:
        for keyword, num_augs in augmentation_plan.items():
            original_data_folder = os.path.join(original_base_folder, keyword)
            output_data_folder = os.path.join(output_base_folder, keyword)

            if not os.path.exists(original_data_folder):
                print(f"Warning: Source folder not found at '{original_data_folder}'. Skipping.")
                continue

            augment_and_save(
                input_dir=original_data_folder,
                output_dir=output_data_folder,
                noise_dir=noise_folder,
                num_augmentations=num_augs
            )

        print("\nDataset balancing process complete!")
        print(f"Check the '{output_base_folder}' directory for your new files.")