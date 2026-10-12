import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from src.infer.infer import load_checkpoint, resolve_checkpoint


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = {
            "weight": torch.tensor([[0.25, -0.5]]),
            "bias": torch.tensor([0.1]),
        }
        self.cached = self.root / "cache" / "model.pt"
        self.cached.parent.mkdir()
        torch.save(self.weights, self.cached)
        self.missing = self.root / "missing" / "model.pt"

    def assert_loaded(self, path):
        model = load_checkpoint(torch.nn.Linear(2, 1), path, torch.device("cpu"))
        for key, value in self.weights.items():
            torch.testing.assert_close(model.state_dict()[key], value)

    @patch("huggingface_hub.hf_hub_download")
    def test_existing_local_weights_do_not_use_hub(self, download):
        self.assert_loaded(self.cached)
        download.assert_not_called()

    @patch("huggingface_hub.hf_hub_download")
    def test_missing_local_weights_can_use_offline_cache(self, download):
        download.return_value = str(self.cached)
        self.assert_loaded(self.missing)
        download.assert_called_once_with(
            repo_id="ASLP-lab/MSVAE", filename="model.pt", local_files_only=True,
        )
        self.assertFalse(self.missing.exists())

    @patch("huggingface_hub.hf_hub_download")
    def test_cache_miss_downloads_and_loads_release_weights(self, download):
        download.side_effect = [LocalEntryNotFoundError("Not cached"), str(self.cached)]
        self.assert_loaded(self.missing)
        self.assertEqual(download.call_count, 2)
        self.assertEqual(
            download.call_args.kwargs,
            {"repo_id": "ASLP-lab/MSVAE", "filename": "model.pt"},
        )

    @patch("huggingface_hub.hf_hub_download")
    def test_failed_download_does_not_load_another_checkpoint(self, download):
        download.side_effect = [LocalEntryNotFoundError("Not cached"), OSError("Download failed")]
        with self.assertRaisesRegex(OSError, "Download failed"):
            self.assert_loaded(self.missing)
        self.assertFalse(self.missing.exists())

    @patch("huggingface_hub.hf_hub_download")
    def test_directory_is_rejected_without_downloading(self, download):
        with self.assertRaisesRegex(ValueError, "not a file"):
            resolve_checkpoint(self.root)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
