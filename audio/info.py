import base64
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal
import soundfile as sf
from tqdm import tqdm


class AdvancedAudioAnalyzer:
    def __init__(
        self,
        input_directory: str,
        output_directory: str = None,
        target_lufs: float = -14.0,
        true_peak_limit: float = -1.0,
        sample_rate: int = 96000,
        max_threads: int = 4,
    ):
        """
        Initialize the Advanced Audio Analyzer.
        """
        self.input_directory = Path(input_directory)
        self.output_directory = (
            Path(output_directory)
            if output_directory
            else self.input_directory / "normalized_analyzed"
        )
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.sample_rate = sample_rate
        self.max_threads = max_threads
        self.processed_files_log = []

        # Create output directory structure
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "spectrum_plots").mkdir(exist_ok=True)

    def analyze_spectrum(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Perform detailed spectrum analysis.
        """
        # Calculate spectrum using STFT
        hop_length = 2048
        n_fft = 4096

        # Get spectrogram
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Calculate average spectrum
        avg_spectrum = np.mean(np.abs(D), axis=1)
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)

        # Identify peaks in spectrum
        peaks, _ = signal.find_peaks(avg_spectrum, height=np.mean(avg_spectrum))
        peak_freqs = freq_bins[peaks]

        # Generate spectrum plot
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        librosa.display.specshow(
            S_db, sr=sample_rate, hop_length=hop_length, x_axis="time", y_axis="log"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")

        plt.subplot(2, 1, 2)
        plt.semilogx(freq_bins, librosa.amplitude_to_db(avg_spectrum))
        plt.grid(True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Average Spectrum")

        # Save plot to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        spectrum_plot = base64.b64encode(buf.getvalue()).decode()

        return {
            "peak_frequencies": peak_freqs.tolist(),
            "spectrum_plot": spectrum_plot,
            "spectral_centroid": librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate
            )[0].mean(),
            "spectral_rolloff": librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate
            )[0].mean(),
        }

    def analyze_musical_features(
        self, audio_data: np.ndarray, sample_rate: int
    ) -> Dict[str, Any]:
        """
        Analyze musical features (key and BPM).
        """
        # Estimate BPM
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)

        # Estimate key
        chromagram = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_correlation = []

        # Template for major and minor keys
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

        for i in range(12):
            major_correlation = np.correlate(
                chromagram.mean(axis=1), np.roll(major_template, i)
            )
            minor_correlation = np.correlate(
                chromagram.mean(axis=1), np.roll(minor_template, i)
            )
            key_correlation.append((major_correlation[0], "major"))
            key_correlation.append((minor_correlation[0], "minor"))

        key_index, mode = max(enumerate(key_correlation), key=lambda x: x[1][0])[0]
        estimated_key = f"{key_names[key_index // 2]} {key_correlation[key_index][1]}"

        return {
            "bpm": float(tempo),
            "key": estimated_key,
            "beat_frames": beat_frames.tolist(),
        }

    def process_file(self, file_path: Path) -> tuple:
        """
        Process a single audio file with comprehensive analysis.
        """
        try:
            # Load audio file
            audio_data, orig_sample_rate = sf.read(file_path)

            # Resample if necessary
            if orig_sample_rate != self.sample_rate:
                audio_data = librosa.resample(
                    audio_data, orig_sr=orig_sample_rate, target_sr=self.sample_rate
                )

            # Convert to mono if necessary
            if len(audio_data.shape) > 1:
                audio_data_mono = np.mean(audio_data, axis=1)
            else:
                audio_data_mono = audio_data

            # Create meter for loudness measurement
            meter = pyln.Meter(self.sample_rate)

            # Analyze original audio
            original_loudness = meter.integrated_loudness(audio_data_mono)

            # Normalize audio
            gain_db = self.target_lufs - original_loudness
            audio_normalized = audio_data * (10 ** (gain_db / 20))

            # Perform spectrum analysis
            spectrum_analysis = self.analyze_spectrum(audio_data_mono, self.sample_rate)

            # Analyze musical features
            musical_features = self.analyze_musical_features(
                audio_data_mono, self.sample_rate
            )

            # Save normalized audio
            output_path = self.output_directory / f"normalized_{file_path.name}"
            sf.write(output_path, audio_normalized, self.sample_rate, subtype="FLOAT")

            # Create analysis report
            analysis_report = {
                "file_name": file_path.name,
                "timestamp": datetime.now().isoformat(),
                "original_loudness": original_loudness,
                "normalized_loudness": self.target_lufs,
                "gain_applied": gain_db,
                "spectrum_analysis": spectrum_analysis,
                "musical_features": musical_features,
                "processing_params": {
                    "target_lufs": self.target_lufs,
                    "true_peak_limit": self.true_peak_limit,
                    "sample_rate": self.sample_rate,
                },
            }

            return True, analysis_report

        except Exception as e:
            return False, str(e)

    def process_directory(self):
        """
        Process all audio files in parallel using multi-threading.
        """
        supported_formats = {".wav", ".flac", ".aif", ".aiff"}
        audio_files = [
            f
            for f in self.input_directory.iterdir()
            if f.suffix.lower() in supported_formats
        ]

        print(f"Found {len(audio_files)} audio files to process")

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_file = {
                executor.submit(self.process_file, audio_file): audio_file
                for audio_file in audio_files
            }

            for future in tqdm(future_to_file, desc="Processing files"):
                audio_file = future_to_file[future]
                try:
                    success, result = future.result()
                    if success:
                        self.processed_files_log.append(result)
                    else:
                        print(f"Error processing {audio_file.name}: {result}")
                except Exception as e:
                    print(f"Error processing {audio_file.name}: {str(e)}")

        # Save processing log
        log_path = self.output_directory / "analysis_log.json"
        with open(log_path, "w") as f:
            json.dump(self.processed_files_log, f, indent=2)

        # Generate summary report
        self._generate_summary_report()

        print(
            f"\nProcessing complete! Analysis logs and reports saved to {self.output_directory}"
        )

    def _generate_summary_report(self):
        """
        Generate a summary report of all processed files.
        """
        summary = {
            "total_files": len(self.processed_files_log),
            "average_bpm": np.mean(
                [log["musical_features"]["bpm"] for log in self.processed_files_log]
            ),
            "key_distribution": {},
            "processing_time": datetime.now().isoformat(),
        }

        # Count key distribution
        for log in self.processed_files_log:
            key = log["musical_features"]["key"]
            summary["key_distribution"][key] = (
                summary["key_distribution"].get(key, 0) + 1
            )

        # Save summary
        with open(self.output_directory / "summary_report.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    print("Advanced Audio Analysis and Normalization Tool")
    print("--------------------------------------------")

    input_dir = input("Enter the directory path containing your audio files: ").strip()
    target_lufs = float(input("Enter target LUFS level (default -14.0): ") or -14.0)
    true_peak = float(input("Enter maximum true peak in dBTP (default -1.0): ") or -1.0)
    sample_rate = int(
        input("Enter desired sample rate in Hz (default 96000): ") or 96000
    )
    max_threads = int(
        input("Enter maximum number of processing threads (default 4): ") or 4
    )

    analyzer = AdvancedAudioAnalyzer(
        input_directory=input_dir,
        target_lufs=target_lufs,
        true_peak_limit=true_peak,
        sample_rate=sample_rate,
        max_threads=max_threads,
    )

    analyzer.process_directory()
