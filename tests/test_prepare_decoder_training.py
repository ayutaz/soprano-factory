"""Tests for prepare_decoder_training.py"""

import os
import tempfile
import pytest


def create_mock_dataset(tmpdir, num_files=10):
    """Create a mock LJSpeech-style dataset directory."""
    wavs_dir = os.path.join(tmpdir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    for i in range(num_files):
        wav_path = os.path.join(wavs_dir, f"audio_{i:04d}.wav")
        with open(wav_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 40)  # Minimal WAV header

    return tmpdir


class TestPrepareDecoderTraining:
    def test_file_list_creation(self, tmp_path):
        """Test that train and val file lists are created correctly."""
        from tools.prepare_decoder_training import main
        import sys

        dataset_dir = create_mock_dataset(str(tmp_path), num_files=100)

        # Mock sys.argv
        original_argv = sys.argv
        sys.argv = [
            "prepare_decoder_training.py",
            "--input-dir", dataset_dir,
            "--val-ratio", "0.1",
            "--seed", "42"
        ]

        try:
            main()
        finally:
            sys.argv = original_argv

        train_path = os.path.join(dataset_dir, "train_decoder.txt")
        val_path = os.path.join(dataset_dir, "val_decoder.txt")

        assert os.path.exists(train_path), "train_decoder.txt should be created"
        assert os.path.exists(val_path), "val_decoder.txt should be created"

        with open(train_path, "r") as f:
            train_files = [line.strip() for line in f if line.strip()]

        with open(val_path, "r") as f:
            val_files = [line.strip() for line in f if line.strip()]

        assert len(train_files) == 90, f"Expected 90 train files, got {len(train_files)}"
        assert len(val_files) == 10, f"Expected 10 val files, got {len(val_files)}"

        # Check that all files are absolute paths
        for f in train_files + val_files:
            assert os.path.isabs(f), f"Path should be absolute: {f}"

    def test_no_overlap_between_train_and_val(self, tmp_path):
        """Test that train and val sets don't overlap."""
        from tools.prepare_decoder_training import main
        import sys

        dataset_dir = create_mock_dataset(str(tmp_path), num_files=50)

        original_argv = sys.argv
        sys.argv = [
            "prepare_decoder_training.py",
            "--input-dir", dataset_dir,
            "--val-ratio", "0.2",
        ]

        try:
            main()
        finally:
            sys.argv = original_argv

        train_path = os.path.join(dataset_dir, "train_decoder.txt")
        val_path = os.path.join(dataset_dir, "val_decoder.txt")

        with open(train_path, "r") as f:
            train_files = set(line.strip() for line in f if line.strip())

        with open(val_path, "r") as f:
            val_files = set(line.strip() for line in f if line.strip())

        overlap = train_files & val_files
        assert len(overlap) == 0, f"Train and val should not overlap: {overlap}"

    def test_reproducibility_with_seed(self, tmp_path):
        """Test that same seed produces same split."""
        from tools.prepare_decoder_training import main
        import sys

        dataset_dir = create_mock_dataset(str(tmp_path), num_files=20)

        results = []
        for _ in range(2):
            original_argv = sys.argv
            sys.argv = [
                "prepare_decoder_training.py",
                "--input-dir", dataset_dir,
                "--val-ratio", "0.2",
                "--seed", "123"
            ]

            try:
                main()
            finally:
                sys.argv = original_argv

            val_path = os.path.join(dataset_dir, "val_decoder.txt")
            with open(val_path, "r") as f:
                val_files = [line.strip() for line in f if line.strip()]
            results.append(val_files)

        assert results[0] == results[1], "Same seed should produce same results"

    def test_missing_wavs_directory(self, tmp_path):
        """Test that error is raised when wavs directory doesn't exist."""
        from tools.prepare_decoder_training import main
        import sys

        original_argv = sys.argv
        sys.argv = [
            "prepare_decoder_training.py",
            "--input-dir", str(tmp_path),
        ]

        try:
            with pytest.raises(ValueError, match="wavs directory not found"):
                main()
        finally:
            sys.argv = original_argv
