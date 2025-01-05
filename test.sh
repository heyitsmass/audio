# Run all tests
python -m pytest

# Run only fast tests
python -m pytest -m "not slow"

# Run with coverage report
python -m pytest --cov=audio_tools

# Run specific test class
python -m pytest -k "TestAudioFingerprinter"