"""Dependency stubs for multimodal_io tests.

Adds torchaudio and espnet_model_zoo stubs so that audio.py can be imported
without those packages. Test files handle their own mocking via
unittest.mock.patch for transformers classes (AutoTokenizer, AutoConfig, etc.).
"""

import importlib.machinery
import os
import sys
import types


def pytest_configure():
    """Install stubs and force CPU before any test collection."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # ---- torchaudio stub (must come before any audio.py import) ----
    if "torchaudio" not in sys.modules:
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.__spec__ = importlib.machinery.ModuleSpec(
            "torchaudio", loader=None
        )

        functional = types.ModuleType("torchaudio.functional")
        functional.__spec__ = importlib.machinery.ModuleSpec(
            "torchaudio.functional", loader=None
        )

        def resample(waveform, orig_freq, new_freq):
            return waveform

        functional.resample = resample
        torchaudio.functional = functional

        sys.modules.setdefault("torchaudio", torchaudio)
        sys.modules.setdefault("torchaudio.functional", functional)

    # ---- espnet_model_zoo stub ----
    if "espnet_model_zoo" not in sys.modules:
        emz = types.ModuleType("espnet_model_zoo")
        emz.__spec__ = importlib.machinery.ModuleSpec(
            "espnet_model_zoo", loader=None
        )
        dl_mod = types.ModuleType("espnet_model_zoo.downloader")
        dl_mod.__spec__ = importlib.machinery.ModuleSpec(
            "espnet_model_zoo.downloader", loader=None
        )

        class ModelDownloader:
            def download_and_unpack(self, *args, **kwargs):
                return {}

        dl_mod.ModelDownloader = ModelDownloader
        emz.downloader = dl_mod
        sys.modules.setdefault("espnet_model_zoo", emz)
        sys.modules.setdefault("espnet_model_zoo.downloader", dl_mod)
