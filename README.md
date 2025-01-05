

# Advanced Audio Analysis Tools

A comprehensive Python package for audio analysis, normalization, fingerprinting, and beat grid analysis. This package provides tools for DJs, music producers, and audio engineers to analyze, normalize, and process audio files.

## Features

-   **Audio Normalization**

    -   LUFS-based loudness normalization
    -   True peak limiting
    -   High-quality audio processing
    -   Multi-threaded batch processing

-   **Spectrum Analysis**

    -   Detailed spectrogram generation
    -   Peak frequency detection
    -   Spectral centroid and rolloff analysis
    -   Visual spectrum plots

-   **Musical Feature Detection**

    -   BPM detection
    -   Musical key estimation
    -   Beat frame analysis
    -   Chromagram analysis

-   **Audio Fingerprinting**

    -   Unique audio fingerprint generation
    -   Similarity matching
    -   Fingerprint database management
    -   Fast audio identification

-   **Beat Grid Analysis**
    -   Precise BPM detection
    -   Beat position mapping
    -   Downbeat detection
    -   Grid strength analysis
    -   Beat grid visualization

## Installation

```bash
pip install advanced-audio-tools
```

### Dependencies

-   Python 3.8+
-   librosa
-   numpy
-   scipy
-   soundfile
-   pyloudnorm
-   matplotlib
-   tqdm

## Quick Start

### Basic Usage

```python
from audio_tools import AdvancedAudioAnalyzer

# Initialize analyzer
analyzer = AdvancedAudioAnalyzer(
    input_directory="path/to/audio/files",
    target_lufs=-14.0,
    sample_rate=96000
)

# Process all files in directory
analyzer.process_directory()
```

### Audio Fingerprinting

```python
from audio_tools import AudioFingerprinter

# Initialize fingerprinter
fingerprinter = AudioFingerprinter()

# Load and fingerprint audio
import soundfile as sf
audio_data, sr = sf.read("song.wav")
fingerprint = fingerprinter.fingerprint_audio(audio_data, sr)

# Save fingerprint
fingerprinter.save_fingerprint(fingerprint, "song.wav")

# Find matches
matches = fingerprinter.find_matches(fingerprint, threshold=0.8)
for file_path, similarity in matches:
    print(f"Match found: {file_path} (similarity: {similarity:.2f})")
```

### Beat Grid Analysis

```python
from audio_tools import BeatGridAnalyzer

# Initialize beat analyzer
beat_analyzer = BeatGridAnalyzer(min_bpm=70, max_bpm=200)

# Analyze beats
beat_info = beat_analyzer.analyze_beat_grid(audio_data, sr)

print(f"BPM: {beat_info.bpm}")
print(f"Grid Strength: {beat_info.grid_strength}")
print(f"Confidence: {beat_info.confidence}")

# Visualize beat grid
visualization = beat_analyzer.visualize_beat_grid(audio_data, sr, beat_info)
```

### Comprehensive Analysis Example

```python
from audio_tools import (
    AdvancedAudioAnalyzer,
    AudioFingerprinter,
    BeatGridAnalyzer
)

class ComprehensiveAnalyzer:
    def __init__(self, input_dir):
        self.audio_analyzer = AdvancedAudioAnalyzer(input_dir)
        self.fingerprinter = AudioFingerprinter()
        self.beat_analyzer = BeatGridAnalyzer()

    def analyze_file(self, file_path):
        # Load audio
        audio_data, sr = sf.read(file_path)

        # Run all analyses
        analysis_result = self.audio_analyzer.process_file(file_path)
        fingerprint = self.fingerprinter.fingerprint_audio(audio_data, sr)
        beat_info = self.beat_analyzer.analyze_beat_grid(audio_data, sr)

        return {
            'analysis': analysis_result,
            'fingerprint': fingerprint,
            'beat_grid': beat_info
        }

# Usage
analyzer = ComprehensiveAnalyzer("path/to/audio")
results = analyzer.analyze_file("song.wav")
```

## Advanced Usage

### Custom Normalization Settings

```python
analyzer = AdvancedAudioAnalyzer(
    input_directory="audio/files",
    target_lufs=-14.0,
    true_peak_limit=-1.0,
    sample_rate=96000,
    max_threads=4
)
```

### Fingerprint Database Management

```python
fingerprinter = AudioFingerprinter()

# Add multiple files to database
for audio_file in audio_files:
    audio_data, sr = sf.read(audio_file)
    fingerprint = fingerprinter.fingerprint_audio(audio_data, sr)
    fingerprinter.save_fingerprint(fingerprint, audio_file)

# Export database
with open('fingerprint_db.json', 'w') as f:
    json.dump(fingerprinter.fingerprint_database, f)
```

### Custom Beat Grid Analysis

```python
beat_analyzer = BeatGridAnalyzer(
    min_bpm=60,
    max_bpm=200
)

# Analyze with custom parameters
beat_info = beat_analyzer.analyze_beat_grid(
    audio_data,
    sr,
)

# Access detailed beat information
print(f"Beat positions: {beat_info.beat_positions}")
print(f"Downbeats: {beat_info.downbeats}")
print(f"Phase offset: {beat_info.phase_offset}")
```

## Output Formats

### Analysis Results

The package generates comprehensive JSON reports containing:

```json
{
    "file_info": {
        "name": "song.wav",
        "duration": 180.5,
        "sample_rate": 96000
    },
    "loudness": {
        "original_lufs": -18.2,
        "normalized_lufs": -14.0,
        "true_peak": -1.0
    },
    "musical_features": {
        "bpm": 128.5,
        "key": "Am",
        "beat_positions": [...],
        "downbeats": [...]
    },
    "spectrum_analysis": {
        "peak_frequencies": [...],
        "spectral_centroid": 2250.0
    },
    "fingerprint": {
        "hash": "a1b2c3...",
        "peak_pairs": [...]
    }
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

-   Built using librosa for audio processing
-   Uses pyloudnorm for LUFS normalization
-   Inspired by various audio analysis techniques and tools

## Support

For support, please open an issue

---

Made with ♥ for the music community
