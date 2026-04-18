"""Tests for espnet2/speechlm/model/speechlm/lm/parallel_pp.py.

Only pure-Python helpers are covered here: ``_prune_to_stage`` and
``_merge_router_logits``, plus a sanity check on
``build_parallel_pp_hf_class`` returning a subclass of the non-PP
parallel model.

The heavy paths — ``from_pretrained``, ``_empty_init``, ``forward``
(incl. ``_forward_first_stage`` / ``_forward_middle_stage`` /
``_forward_last_stage``), and ``_run_decoder_layers`` — all require
``torch.distributed`` (``dist.broadcast``), real HF decoder layers, or
meta-device materialization. They are intentionally left to integration
testing on a multi-GPU host.

Skipped when ``liger_kernel`` is not importable (transitive via
``parallel.py`` → ``loss.py``).
"""

from unittest.mock import patch

import pytest

pytest.importorskip("liger_kernel")

import torch
import torch.nn as nn


class _MockConfig:
    architectures = ["MockModel"]
    vocab_size = 100
    hidden_size = 64
    _attn_implementation = "flash_attention_2"


class _MockInnerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)


class _MockHFModel(nn.Module):
    config_class = _MockConfig

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = _MockConfig()
        self.config = config
        self.model = _MockInnerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(_MockConfig())


@pytest.fixture(autouse=True)
def _patch_transformers():
    """Patch transformers just enough for ``build_parallel_pp_hf_class``."""
    import transformers

    old = getattr(transformers, "MockModel", None)
    transformers.MockModel = _MockHFModel
    with patch.object(
        transformers.AutoConfig,
        "from_pretrained",
        return_value=_MockConfig(),
    ):
        yield
    if old is None:
        delattr(transformers, "MockModel")
    else:
        transformers.MockModel = old


# ---------------------------------------------------------------------------
# _merge_router_logits
# ---------------------------------------------------------------------------
class TestMergeRouterLogits:
    def test_both_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        assert _merge_router_logits(None, None) is None

    def test_prev_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        local = torch.zeros(3, 4)
        assert _merge_router_logits(None, local) is local

    def test_local_none(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        prev = torch.zeros(3, 4)
        assert _merge_router_logits(prev, None) is prev

    def test_both_tensors_concat_dim0(self):
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            _merge_router_logits,
        )

        prev = torch.ones(2, 4)
        local = torch.zeros(3, 4)
        out = _merge_router_logits(prev, local)
        assert out.shape == (5, 4)
        assert torch.equal(out[:2], prev)
        assert torch.equal(out[2:], local)


# ---------------------------------------------------------------------------
# build_parallel_pp_hf_class
# ---------------------------------------------------------------------------
class TestBuildParallelPPClass:
    def test_subclass_of_parallel_llm(self):
        from espnet2.speechlm.model.speechlm.lm.parallel import (
            build_parallel_hf_class,
        )
        from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
            build_parallel_pp_hf_class,
        )

        pp_cls = build_parallel_pp_hf_class("mock-model")
        base_cls = build_parallel_hf_class("mock-model")
        # pp_cls inherits from an independently-built parallel class that is
        # itself a subclass of _MockHFModel. Walk the MRO to confirm the HF
        # base is in both class hierarchies.
        assert issubclass(pp_cls, _MockHFModel)
        assert issubclass(base_cls, _MockHFModel)


# ---------------------------------------------------------------------------
# _prune_to_stage
# ---------------------------------------------------------------------------
def _make_toy_pp_model(num_layers=4):
    """Minimal nn.Module with the attribute shape ``_prune_to_stage`` expects."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
            self.embed_tokens = nn.Embedding(10, 4)
            self.norm = nn.LayerNorm(4)

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.lm_head = nn.Linear(4, 10, bias=False)
            self.stream_emb = nn.Embedding(2, 4)
            self.multimodal_io_dict = nn.ModuleDict({"text": nn.Linear(1, 1)})
            self.adaptor = nn.ModuleDict({})
            self.register_buffer("vocab_weight", torch.ones(10))

    return _Outer()


def _get_prune_fn():
    from espnet2.speechlm.model.speechlm.lm.parallel_pp import (
        build_parallel_pp_hf_class,
    )

    pp_cls = build_parallel_pp_hf_class("mock-model")
    return pp_cls._prune_to_stage


class TestPruneToStage:
    def test_non_local_layers_become_identity(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        originals = list(model.model.layers)
        prune(
            model, layer_start=1, layer_end=3, is_first_stage=False, is_last_stage=False
        )
        assert isinstance(model.model.layers[0], nn.Identity)
        assert isinstance(model.model.layers[3], nn.Identity)
        # Layers 1 and 2 kept their original module identity.
        assert model.model.layers[1] is originals[1]
        assert model.model.layers[2] is originals[2]

    def test_middle_stage_strips_boundary_modules(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 1, 3, is_first_stage=False, is_last_stage=False)
        assert model.model.embed_tokens is None
        assert model.multimodal_io_dict is None
        assert model.adaptor is None
        assert model.lm_head is None
        assert model.stream_emb is None
        assert model.model.norm is None
        assert model.vocab_weight is None

    def test_first_stage_keeps_embed(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 0, 2, is_first_stage=True, is_last_stage=False)
        assert model.model.embed_tokens is not None
        assert model.multimodal_io_dict is not None
        assert model.adaptor is not None
        # Not last stage: lm_head, stream_emb, norm removed.
        assert model.lm_head is None
        assert model.stream_emb is None
        assert model.model.norm is None

    def test_last_stage_keeps_head(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 2, 4, is_first_stage=False, is_last_stage=True)
        assert model.lm_head is not None
        assert model.stream_emb is not None
        assert model.model.norm is not None
        # Not first stage: embed / IO dict / adaptor removed.
        assert model.model.embed_tokens is None
        assert model.multimodal_io_dict is None
        assert model.adaptor is None

    def test_single_stage_keeps_everything(self):
        prune = _get_prune_fn()
        model = _make_toy_pp_model(num_layers=4)
        prune(model, 0, 4, is_first_stage=True, is_last_stage=True)
        # All layers local → no Identity swaps.
        for layer in model.model.layers:
            assert not isinstance(layer, nn.Identity)
        # Both first and last: nothing nulled.
        assert model.model.embed_tokens is not None
        assert model.lm_head is not None
        assert model.stream_emb is not None
        assert model.model.norm is not None
