import os

import numpy as np
from pydub import AudioSegment
from tqdm import tqdm


def normalize_audio_files(input_directory, target_lufs=-14.0, output_directory=None):
    """
    Normalize audio files to a target LUFS (Loudness Units Full Scale) level.

    Args:
        input_directory (str): Directory containing audio files
        target_lufs (float): Target LUFS level (default: -14.0 LUFS, standard for streaming)
        output_directory (str): Directory to save normalized files (default: None, creates 'normalized' subdirectory)
    """
    if output_directory is None:
        output_directory = os.path.join(input_directory, "normalized")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Supported audio formats
    supported_formats = (".mp3", ".wav", ".m4a", ".ogg")

    # Get list of audio files
    audio_files = [
        f for f in os.listdir(input_directory) if f.lower().endswith(supported_formats)
    ]

    print(f"Found {len(audio_files)} audio files to process")

    for audio_file in tqdm(audio_files, desc="Processing files"):
        try:
            # Load audio file
            input_path = os.path.join(input_directory, audio_file)
            audio = AudioSegment.from_file(input_path)

            # Calculate current LUFS (approximate)
            # Note: This is a simplified LUFS calculation
            rms = audio.rms
            current_lufs = 20 * np.log10(rms) - 10

            # Calculate required gain
            gain_db = target_lufs - current_lufs

            # Apply gain
            normalized_audio = audio + gain_db

            # Export normalized file
            output_filename = f"normalized_{audio_file}"
            output_path = os.path.join(output_directory, output_filename)

            # Export with original format
            file_format = audio_file.split(".")[-1].lower()
            normalized_audio.export(output_path, format=file_format)

            print(f"Processed {audio_file}: Applied {gain_db:.1f}dB gain")

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")


if __name__ == "__main__":
    print("Audio Normalization Tool")
    print("-----------------------")

    # Get input from user
    input_dir = input("Enter the directory path containing your audio files: ").strip()
    target_lufs = float(input("Enter target LUFS level (default -14.0): ") or -14.0)

    if not os.path.exists(input_dir):
        print("Error: Directory not found!")
        exit(1)

    print(f"\nNormalizing audio files to {target_lufs} LUFS...")
    normalize_audio_files(input_dir, target_lufs)
    print("\nNormalization complete!")
