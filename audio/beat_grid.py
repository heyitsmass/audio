import base64
import json
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import List

import librosa
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BeatGridInfo:
    bpm: float
    beat_positions: List[float]
    confidence: float
    phase_offset: float
    grid_strength: float
    downbeats: List[float]


class BeatGridAnalyzer:
    def __init__(self, min_bpm: float = 70, max_bpm: float = 200):
        """
        Initialize the Beat Grid Analyzer.

        Args:
            min_bpm: Minimum BPM to consider
            max_bpm: Maximum BPM to consider
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def _calculate_grid_strength(self, beat_positions: np.ndarray) -> float:
        """
        Calculate how consistent the beat grid is.
        """
        # Calculate inter-beat intervals
        ibis = np.diff(beat_positions)

        # Calculate variance in intervals
        variance = np.var(ibis)
        mean_ibi = np.mean(ibis)

        # Return normalized strength metric
        return 1.0 / (1.0 + variance / mean_ibi)

    def _find_downbeats(
        self, audio_data: np.ndarray, sr: int, beat_positions: np.ndarray
    ) -> List[float]:
        """
        Detect likely downbeat positions.
        """
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)

        # Get onset peaks near beat positions
        downbeats = []
        for beat_time in beat_positions[::4]:  # Check every 4th beat
            beat_frame = int(beat_time * sr / 512)  # Convert to onset frames
            if beat_frame < len(onset_env):
                local_max = np.argmax(
                    onset_env[
                        max(0, beat_frame - 2) : min(len(onset_env), beat_frame + 3)
                    ]
                )
                downbeats.append(beat_time + (local_max - 2) * 512 / sr)

        return downbeats

    def analyze_beat_grid(self, audio_data: np.ndarray, sr: int) -> BeatGridInfo:
        """
        Perform detailed beat grid analysis.
        """
        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio_data,
            sr=sr,
            trim=False,
            tightness=100,
            bpm_min=self.min_bpm,
            bpm_max=self.max_bpm,
        )

        # Convert frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # Calculate grid strength
        grid_strength = self._calculate_grid_strength(beat_times)

        # Find phase offset (time to first beat)
        phase_offset = beat_times[0] if len(beat_times) > 0 else 0.0

        # Detect downbeats
        downbeats = self._find_downbeats(audio_data, sr, beat_times)

        # Calculate confidence based on grid strength and tempo stability
        confidence = grid_strength * (1.0 - abs(tempo - 120) / 120)

        return BeatGridInfo(
            bpm=float(tempo),
            beat_positions=beat_times.tolist(),
            confidence=float(confidence),
            phase_offset=float(phase_offset),
            grid_strength=float(grid_strength),
            downbeats=downbeats,
        )

    def visualize_beat_grid(
        self, audio_data: np.ndarray, sr: int, beat_info: BeatGridInfo
    ) -> str:
        """
        Create visualization of the beat grid.
        """
        plt.figure(figsize=(15, 5))

        # Plot waveform
        plt.subplot(211)
        librosa.display.waveshow(audio_data, sr=sr)

        # Plot beat positions
        for beat in beat_info.beat_positions:
            plt.axvline(x=beat, color="r", alpha=0.5, linestyle="--")

        # Plot downbeats
        for downbeat in beat_info.downbeats:
            plt.axvline(x=downbeat, color="g", alpha=0.7, linestyle="-")

        plt.title(
            f"Beat Grid Analysis (BPM: {beat_info.bpm:.1f}, "
            f"Confidence: {beat_info.confidence:.2f})"
        )

        # Plot onset strength
        plt.subplot(212)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        plt.plot(times, onset_env)
        plt.title("Onset Strength")

        # Save plot to string
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    print("Beat Analyzer Module Demo")
    print("-------------------------")

    # Example usage
    file_path = input("Enter path to audio file: ").strip()

    # Load audio
    audio_data, sr = librosa.load(file_path, sr=None)

    # Initialize analyzers
    beat_analyzer = BeatGridAnalyzer()

    print("\nAnalyzing beat grid...")
    beat_info = beat_analyzer.analyze_beat_grid(audio_data, sr)
    print(f"Detected BPM: {beat_info.bpm:.1f}")
    print(f"Grid confidence: {beat_info.confidence:.2f}")

    # Save results
    output_dir = Path("analysis_output")
    output_dir.mkdir(exist_ok=True)

    beat_grid = asdict(beat_info)

    with open(output_dir / "beat_grid.json", "w") as f:
        json.dump(beat_grid, f, indent=2)

    print(f"\nResults saved to {output_dir / 'beat_grid.json'}")
