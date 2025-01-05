import os
import tempfile
import unittest
from pathlib import Path


import numpy as np
import pytest
import soundfile as sf

from audio import (
    AdvancedAudioAnalyzer,
    AudioFingerprint,
    AudioFingerprinter,
    BeatGridAnalyzer,
    BeatGridInfo,
)


class TestAudioData:
    """Test data generator for audio processing tests."""

    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate=44100):
        """Generate a simple sine wave for testing."""
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * frequency * t)

    @staticmethod
    def generate_beat_pattern(bpm, duration, sample_rate=44100):
        """Generate a test beat pattern."""
        beat_length = int(sample_rate * 60 / bpm)
        pattern = np.zeros(int(sample_rate * duration))
        for i in range(0, len(pattern), beat_length):
            pattern[i : i + 100] = 1.0
        return pattern

    @staticmethod
    def create_temp_audio_file(audio_data, sample_rate, suffix=".wav"):
        """Create a temporary audio file for testing."""
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        sf.write(temp_file.name, audio_data, sample_rate)
        return temp_file.name


class TestAdvancedAudioAnalyzer(unittest.TestCase):
    """Test cases for the AdvancedAudioAnalyzer class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.test_data = TestAudioData()
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = AdvancedAudioAnalyzer(
            input_directory=self.temp_dir, target_lufs=-14.0, sample_rate=44100
        )

        # Create test audio file
        self.audio_data = self.test_data.generate_sine_wave(440, 2.0)
        self.test_file = self.test_data.create_temp_audio_file(self.audio_data, 44100)

    def tearDown(self):
        """Clean up test environment after each test."""
        os.remove(self.test_file)
        os.rmdir(self.temp_dir)

    def test_normalization(self):
        """Test audio normalization functionality."""
        result = self.analyzer.process_file(self.test_file)
        self.assertIsNotNone(result)

        # Check if normalization achieved target LUFS
        normalized_loudness = result["loudness"]
        self.assertAlmostEqual(normalized_loudness, -14.0, places=1)

    def test_spectrum_analysis(self):
        """Test spectrum analysis functionality."""
        result = self.analyzer.process_file(self.test_file)

        # Check if spectrum analysis contains expected components
        self.assertIn("spectrum_analysis", result)
        self.assertIn("peak_frequencies", result["spectrum_analysis"])

        # For a 440Hz sine wave, should find peak near 440Hz
        peaks = result["spectrum_analysis"]["peak_frequencies"]
        self.assertTrue(any(abs(peak - 440) < 10 for peak in peaks))

    @pytest.mark.slow
    def test_batch_processing(self):
        """Test processing multiple files."""
        # Create multiple test files
        test_files = []
        for i in range(3):
            audio_data = self.test_data.generate_sine_wave(440 * (i + 1), 1.0)
            test_files.append(self.test_data.create_temp_audio_file(audio_data, 44100))

        # Process all files
        self.analyzer.process_directory()

        # Check if output files exist
        for file in test_files:
            normalized_path = Path(self.temp_dir) / f"normalized_{Path(file).name}"
            self.assertTrue(normalized_path.exists())

        # Cleanup
        for file in test_files:
            os.remove(file)


class TestAudioFingerprinter(unittest.TestCase):
    """Test cases for the AudioFingerprinter class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.test_data = TestAudioData()
        self.fingerprinter = AudioFingerprinter()

        # Create test audio
        self.audio_data = self.test_data.generate_sine_wave(440, 2.0)
        self.test_file = self.test_data.create_temp_audio_file(self.audio_data, 44100)

    def tearDown(self):
        """Clean up test environment after each test."""
        os.remove(self.test_file)

    def test_fingerprint_generation(self):
        """Test fingerprint generation."""
        fingerprint = self.fingerprinter.fingerprint_audio(self.audio_data, 44100)

        self.assertIsInstance(fingerprint, AudioFingerprint)
        self.assertIsNotNone(fingerprint.hash)
        self.assertGreater(len(fingerprint.peak_pairs), 0)

    def test_fingerprint_matching(self):
        """Test fingerprint matching functionality."""
        # Generate two similar audio samples
        audio1 = self.test_data.generate_sine_wave(440, 2.0)
        audio2 = audio1 + np.random.normal(0, 0.01, len(audio1))  # Add slight noise

        fp1 = self.fingerprinter.fingerprint_audio(audio1, 44100)
        fp2 = self.fingerprinter.fingerprint_audio(audio2, 44100)

        # Save fingerprints
        self.fingerprinter.save_fingerprint(fp1, "test1.wav")

        # Find matches
        matches = self.fingerprinter.find_matches(fp2, threshold=0.8)
        self.assertGreater(len(matches), 0)


class TestBeatGridAnalyzer(unittest.TestCase):
    """Test cases for the BeatGridAnalyzer class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.test_data = TestAudioData()
        self.analyzer = BeatGridAnalyzer(min_bpm=70, max_bpm=200)

    def test_beat_detection(self):
        """Test beat detection functionality."""
        # Create test audio with known BPM
        test_bpm = 120
        audio_data = self.test_data.generate_beat_pattern(test_bpm, 10.0)

        beat_info = self.analyzer.analyze_beat_grid(audio_data, 44100)

        self.assertIsInstance(beat_info, BeatGridInfo)
        self.assertAlmostEqual(beat_info.bpm, test_bpm, delta=1.0)
        self.assertGreater(beat_info.confidence, 0.8)

    def test_downbeat_detection(self):
        """Test downbeat detection."""
        # Create test audio with emphasized downbeats
        test_bpm = 120
        audio_data = self.test_data.generate_beat_pattern(test_bpm, 10.0)
        # Emphasize every 4th beat
        for i in range(0, len(audio_data), int(44100 * 60 / test_bpm * 4)):
            if i + 100 < len(audio_data):
                audio_data[i : i + 100] *= 2

        beat_info = self.analyzer.analyze_beat_grid(audio_data, 44100)

        self.assertGreater(len(beat_info.downbeats), 0)
        # Check if downbeats are approximately 4 beats apart
        if len(beat_info.downbeats) > 1:
            avg_spacing = np.mean(np.diff(beat_info.downbeats))
            expected_spacing = 60 / test_bpm * 4
            self.assertAlmostEqual(avg_spacing, expected_spacing, delta=0.1)


@pytest.mark.integration
class TestIntegration(unittest.TestCase):
    """Integration tests for the complete audio analysis system."""

    def setUp(self):
        self.test_data = TestAudioData()
        self.temp_dir = tempfile.mkdtemp()

        # Initialize all components
        self.audio_analyzer = AdvancedAudioAnalyzer(self.temp_dir)
        self.fingerprinter = AudioFingerprinter()
        self.beat_analyzer = BeatGridAnalyzer()

    def tearDown(self):
        os.rmdir(self.temp_dir)

    def test_complete_workflow(self):
        """Test complete workflow from analysis to fingerprinting to beat detection."""
        # Create test audio
        audio_data = self.test_data.generate_beat_pattern(120, 10.0)
        test_file = self.test_data.create_temp_audio_file(audio_data, 44100)

        try:
            # Run complete analysis
            analysis_result = self.audio_analyzer.process_file(test_file)
            fingerprint = self.fingerprinter.fingerprint_audio(audio_data, 44100)
            beat_info = self.beat_analyzer.analyze_beat_grid(audio_data, 44100)

            # Verify all components worked together
            self.assertIsNotNone(analysis_result)
            self.assertIsNotNone(fingerprint)
            self.assertIsNotNone(beat_info)

            # Verify data consistency
            self.assertAlmostEqual(
                analysis_result["musical_features"]["bpm"], beat_info.bpm, delta=1.0
            )
        finally:
            os.remove(test_file)


def run_tests():
    """Run all tests with pytest."""
    pytest.main(
        [
            "--verbose",
            "--cov=audio_tools",
            "--cov-report=term-missing",
            "-m",
            "not slow",  # Skip slow tests by default
        ]
    )


if __name__ == "__main__":
    run_tests()
