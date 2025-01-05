import soundfile as sf
import pyloudnorm as pyln
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime


class HighQualityAudioNormalizer:
    def __init__(
        self,
        input_directory,
        output_directory=None,
        target_lufs=-14.0,
        true_peak_limit=-1.0,
        sample_rate=96000,
    ):
        """
        Initialize the High Quality Audio Normalizer.

        Args:
            input_directory (str): Source directory for audio files
            output_directory (str): Output directory (default: creates 'normalized_hq' subdirectory)
            target_lufs (float): Target LUFS level
            true_peak_limit (float): Maximum true peak level in dBTP
            sample_rate (int): Target sample rate (default: 96kHz)
        """
        self.input_directory = Path(input_directory)
        self.output_directory = (
            Path(output_directory)
            if output_directory
            else self.input_directory / "normalized_hq"
        )
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.sample_rate = sample_rate
        self.processed_files_log = []

        # Create output directory if it doesn't exist
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def analyze_audio(self, audio_data, sample_rate):
        """
        Perform detailed audio analysis.
        """
        # Create BS.1770-4 meter
        meter = pyln.Meter(sample_rate)

        # Measure integrated loudness
        loudness = meter.integrated_loudness(audio_data)

        # Calculate true peak
        true_peak = np.max(np.abs(audio_data))
        true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -np.inf

        # Calculate dynamic range
        rms = np.sqrt(np.mean(np.square(audio_data)))
        crest_factor = 20 * np.log10(np.max(np.abs(audio_data)) / rms)

        return {
            "loudness": loudness,
            "true_peak_db": true_peak_db,
            "dynamic_range": crest_factor,
        }

    def normalize_audio(self, audio_data, current_loudness):
        """
        Normalize audio with lookahead limiting for true peak control.
        """
        # Calculate required gain
        gain_db = self.target_lufs - current_loudness

        # Apply gain
        audio_data_normalized = audio_data * (10 ** (gain_db / 20))

        # Simple lookahead limiting for true peak control
        if self.true_peak_limit is not None:
            peak_limit = 10 ** (self.true_peak_limit / 20)
            if np.max(np.abs(audio_data_normalized)) > peak_limit:
                audio_data_normalized = np.clip(
                    audio_data_normalized, -peak_limit, peak_limit
                )

        return audio_data_normalized

    def process_file(self, file_path):
        """
        Process a single audio file with detailed analysis and logging.
        """
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(file_path)

            # Convert to mono if necessary for consistent loudness measurement
            if len(audio_data.shape) > 1:
                audio_data_mono = np.mean(audio_data, axis=1)
            else:
                audio_data_mono = audio_data

            # Analyze original audio
            original_analysis = self.analyze_audio(audio_data_mono, sample_rate)

            # Normalize audio
            audio_normalized = self.normalize_audio(
                audio_data, original_analysis["loudness"]
            )

            # Analyze normalized audio
            normalized_analysis = self.analyze_audio(
                audio_normalized
                if len(audio_data.shape) == 1
                else np.mean(audio_normalized, axis=1),
                sample_rate,
            )

            # Prepare output path
            output_path = self.output_directory / f"normalized_{file_path.name}"

            # Save normalized audio
            sf.write(output_path, audio_normalized, sample_rate, subtype="FLOAT")

            # Log processing details
            processing_log = {
                "file_name": file_path.name,
                "timestamp": datetime.now().isoformat(),
                "original_analysis": original_analysis,
                "normalized_analysis": normalized_analysis,
                "processing_params": {
                    "target_lufs": self.target_lufs,
                    "true_peak_limit": self.true_peak_limit,
                    "sample_rate": sample_rate,
                },
            }

            self.processed_files_log.append(processing_log)
            return True, processing_log

        except Exception as e:
            return False, str(e)

    def process_directory(self):
        """
        Process all supported audio files in the directory.
        """
        supported_formats = {".wav", ".flac", ".aif", ".aiff"}
        audio_files = [
            f
            for f in self.input_directory.iterdir()
            if f.suffix.lower() in supported_formats
        ]

        print(f"Found {len(audio_files)} high-quality audio files to process")

        for audio_file in tqdm(audio_files, desc="Processing files"):
            success, result = self.process_file(audio_file)
            if not success:
                print(f"Error processing {audio_file.name}: {result}")

        # Save processing log
        log_path = self.output_directory / "processing_log.json"
        with open(log_path, "w") as f:
            json.dump(self.processed_files_log, f, indent=2)

        print(f"\nProcessing complete! Log saved to {log_path}")


if __name__ == "__main__":
    print("High Quality Audio Normalization Tool")
    print("------------------------------------")

    input_dir = input("Enter the directory path containing your audio files: ").strip()
    target_lufs = float(input("Enter target LUFS level (default -14.0): ") or -14.0)
    true_peak = float(input("Enter maximum true peak in dBTP (default -1.0): ") or -1.0)
    sample_rate = int(
        input("Enter desired sample rate in Hz (default 96000): ") or 96000
    )

    normalizer = HighQualityAudioNormalizer(
        input_directory=input_dir,
        target_lufs=target_lufs,
        true_peak_limit=true_peak,
        sample_rate=sample_rate,
    )

    normalizer.process_directory()
