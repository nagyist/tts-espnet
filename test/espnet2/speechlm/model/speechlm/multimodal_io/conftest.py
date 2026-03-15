"""Dependency stubs for multimodal_io tests.

Enriches parent conftest stubs (transformers, joblib, humanfriendly) and adds
torchaudio stub so that audio.py and text.py can be imported without real deps.
"""

import importlib.machinery
import os
import sys
import types

import torch.nn as nn


def pytest_configure():
    """Enrich stubs and force CPU before any test collection."""

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


def _enrich_transformers():
    """Enrich the transformers stub with methods needed by multimodal_io.

    Called lazily via a pytest session-scoped fixture to ensure
    the parent conftest has already created the base transformers stub.
    """
    transformers = sys.modules.get("transformers")
    if transformers is None:
        return

    tok_cls = transformers.AutoTokenizer

    # Add tokenizer methods needed by HuggingFaceTextIO
    if not hasattr(tok_cls, "encode"):
        tok_cls.encode = lambda self, text, **kw: list(
            range(min(len(text.split()) + 2, 50))
        )

    if not hasattr(tok_cls, "decode"):
        tok_cls.decode = lambda self, token_ids, **kw: " ".join(
            f"w{i}" for i in token_ids
        )

    if not hasattr(tok_cls, "get_vocab"):
        tok_cls.get_vocab = lambda self: {f"tok_{i}": i for i in range(80)}

    # Add AutoProcessor stub
    if not hasattr(transformers, "AutoProcessor"):

        class _MockFeatureExtractor:
            sampling_rate = 16000
            hop_length = 160
            n_samples = 480000
            feature_size = 128

            def __call__(self, *args, **kwargs):
                import numpy as np

                return {"input_features": np.zeros((1, 128, 3000))}

        class AutoProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                proc = AutoProcessor()
                proc.feature_extractor = _MockFeatureExtractor()
                return proc

        transformers.AutoProcessor = AutoProcessor

    # Add Qwen model stubs
    if not hasattr(transformers, "Qwen2_5OmniForConditionalGeneration"):

        class Qwen2_5OmniForConditionalGeneration(nn.Module):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("Model loading disabled in tests")

        transformers.Qwen2_5OmniForConditionalGeneration = (
            Qwen2_5OmniForConditionalGeneration
        )

    if not hasattr(transformers, "Qwen3OmniMoeForConditionalGeneration"):

        class Qwen3OmniMoeForConditionalGeneration(nn.Module):
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("Model loading disabled in tests")

        transformers.Qwen3OmniMoeForConditionalGeneration = (
            Qwen3OmniMoeForConditionalGeneration
        )


def pytest_collection_modifyitems(session, config, items):
    """Hook that runs after all conftest pytest_configure hooks and collection.

    Use this to enrich the transformers stub since the parent conftest
    has guaranteed to have run by this point.
    """
    _enrich_transformers()


def pytest_runtest_setup(item):
    """Ensure enrichment runs before each test as a safety net."""
    _enrich_transformers()
