import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Tuple
import json
from pathlib import Path
import librosa
import numpy as np


@dataclass
class AudioFingerprint:
    hash: str
    peak_pairs: List[Tuple[float, float]]
    timestamp: str
    duration: float
    sample_rate: int


class AudioFingerprinter:
    def __init__(self, target_sr: int = 44100):
        """
        Initialize the Audio Fingerprinter.

        Args:
            target_sr: Target sample rate for analysis
        """
        self.target_sr = target_sr
        self.fingerprint_database = {}

    def _generate_constellation_map(
        self, audio_data: np.ndarray, sr: int
    ) -> List[Tuple[float, float]]:
        """
        Generate frequency peak pairs for fingerprinting.
        """
        # Calculate spectrogram
        hop_length = 512
        n_fft = 2048

        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Find local peaks
        peaks = []
        for i in range(1, D_db.shape[1] - 1):
            for j in range(1, D_db.shape[0] - 1):
                if D_db[j, i] > D_db[j - 1 : j + 2, i - 1 : i + 2].mean() + 20:
                    peaks.append(
                        (
                            librosa.fft_frequencies(sr=sr, n_fft=n_fft)[j],
                            i * hop_length / sr,
                        )
                    )

        return sorted(peaks, key=lambda x: (x[1], x[0]))

    def _create_hash(self, peak_pairs: List[Tuple[float, float]]) -> str:
        """
        Create a unique hash from peak pairs.
        """
        # Convert peak pairs to bytes
        peak_bytes = b""
        for freq, time in peak_pairs:
            peak_bytes += np.array([freq, time]).tobytes()

        return hashlib.sha256(peak_bytes).hexdigest()

    def fingerprint_audio(self, audio_data: np.ndarray, sr: int) -> AudioFingerprint:
        """
        Generate audio fingerprint from audio data.
        """
        # Resample if necessary
        if sr != self.target_sr:
            audio_data = librosa.resample(
                audio_data, orig_sr=sr, target_sr=self.target_sr
            )
            sr = self.target_sr

        # Generate constellation map
        peak_pairs = self._generate_constellation_map(audio_data, sr)

        # Create fingerprint
        fingerprint = AudioFingerprint(
            hash=self._create_hash(peak_pairs),
            peak_pairs=peak_pairs,
            timestamp=datetime.now().isoformat(),
            duration=len(audio_data) / sr,
            sample_rate=sr,
        )

        return fingerprint

    def save_fingerprint(self, fingerprint: AudioFingerprint, file_path: str) -> None:
        """
        Save fingerprint to database.
        """
        self.fingerprint_database[file_path] = asdict(fingerprint)

    def find_matches(
        self, query_fingerprint: AudioFingerprint, threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Find matching audio files in database.
        """
        matches = []
        query_peaks = set(query_fingerprint.peak_pairs)

        for file_path, stored_fp in self.fingerprint_database.items():
            stored_peaks = set(stored_fp["peak_pairs"])
            similarity = len(query_peaks.intersection(stored_peaks)) / len(
                query_peaks.union(stored_peaks)
            )

            if similarity >= threshold:
                matches.append((file_path, similarity))

        return sorted(matches, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    print("Audio Fingerpring Module Demo")
    print("-------------------------")

    # Example usage
    file_path = input("Enter path to audio file: ").strip()

    # Load audio
    audio_data, sr = librosa.load(file_path, sr=None)

    # Initialize analyzers
    fingerprinter = AudioFingerprinter()

    # Generate fingerprint
    print("\nGenerating audio fingerprint...")
    fingerprint = fingerprinter.fingerprint_audio(audio_data, sr)
    print(f"Fingerprint hash: {fingerprint.hash}")

    # Save results
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    fingerprint = asdict(fingerprint)

    with open(output_dir / "fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)

    print(f"\nResults saved to {output_dir / 'fingerprint.json'}")
