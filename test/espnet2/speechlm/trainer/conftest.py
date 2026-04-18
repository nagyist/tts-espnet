"""Dependency stubs for espnet2/speechlm/trainer tests.

Injects lightweight stubs for heavy optional dependencies (deepspeed,
wandb, humanfriendly, torchtitan, parallel_utils) so that tests can run
in CI without those packages. The real packages are used when available.
"""

import importlib.machinery
import sys
import types

import torch
import torch.nn as nn


def _install_stub(name, module):
    sys.modules.setdefault(name, module)


def _make_module(name):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return mod


def pytest_configure():
    """Inject stubs before any test module is collected."""

    # ---- deepspeed stub ----
    if "deepspeed" not in sys.modules:
        ds = types.ModuleType("deepspeed")
        ds.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None)

        def initialize(model=None, model_parameters=None, config=None, **kwargs):
            class _MockEngine(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.module = model

                def forward(self, **kwargs):
                    return self.module(**kwargs)

                def train(self, mode=True):
                    self.module.train(mode)
                    return self

                def eval(self):
                    self.module.eval()
                    return self

                def backward(self, loss):
                    loss.backward()

                def step(self):
                    pass

                def get_lr(self):
                    return [1e-4]

                def get_global_grad_norm(self):
                    return 0.0

                def save_checkpoint(self, path, client_state=None):
                    pass

                def load_checkpoint(self, path):
                    return None, None

                def parameters(self):
                    return self.module.parameters()

                def named_parameters(self):
                    return self.module.named_parameters()

            engine = _MockEngine(model)
            return engine, None, None, None

        ds.initialize = initialize
        _install_stub("deepspeed", ds)

    # ---- wandb stub ----
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

        class _MockRun:
            pass

        class _MockConfig:
            def update(self, d):
                pass

        wandb.run = _MockRun()
        wandb.config = _MockConfig()

        def log(data, step=None):
            pass

        def init(*args, **kwargs):
            return _MockRun()

        def finish(*args, **kwargs):
            return None

        wandb.log = log
        wandb.init = init
        wandb.finish = finish
        _install_stub("wandb", wandb)

    # ---- humanfriendly stub ----
    if "humanfriendly" not in sys.modules:
        humanfriendly = types.ModuleType("humanfriendly")
        humanfriendly.__spec__ = importlib.machinery.ModuleSpec(
            "humanfriendly", loader=None
        )

        def format_size(num_bytes, **kwargs):
            return f"{num_bytes} bytes"

        humanfriendly.format_size = format_size
        _install_stub("humanfriendly", humanfriendly)

    # ---- torchtitan stub ----
    # TitanTrainer imports `from torchtitan.distributed import utils as dist_utils`
    # and calls `dist_utils.clip_grad_norm_`. In CI we don't need a real impl
    # since the methods that call clip_grad_norm_ run only on GPU+dist.
    if "torchtitan" not in sys.modules:
        tt = _make_module("torchtitan")
        tt_dist = _make_module("torchtitan.distributed")
        tt_dist_utils = _make_module("torchtitan.distributed.utils")

        def _stub_clip_grad_norm_(parameters, max_norm, **kwargs):
            return torch.tensor(0.0)

        tt_dist_utils.clip_grad_norm_ = _stub_clip_grad_norm_
        tt_dist.utils = tt_dist_utils
        tt.distributed = tt_dist
        _install_stub("torchtitan", tt)
        _install_stub("torchtitan.distributed", tt_dist)
        _install_stub("torchtitan.distributed.utils", tt_dist_utils)

    # ---- espnet2.speechlm.model.speechlm.parallel_utils stub ----
    # The real parallel_utils package lives on a separate feature branch and
    # may not be present on this branch. TitanTrainer imports
    # `init_parallel_dims`, `parallel_strategies`, `build_pipeline` from it
    # at module load time.
    pu_name = "espnet2.speechlm.model.speechlm.parallel_utils"
    _real_pu = None
    try:
        __import__(pu_name)
        _real_pu = sys.modules.get(pu_name)
    except Exception:
        _real_pu = None

    # Install stub when the real package is missing OR is a namespace-package
    # placeholder (missing init_parallel_dims means it's not the real impl).
    if _real_pu is None or not hasattr(_real_pu, "init_parallel_dims"):
        pu = _make_module(pu_name)

        class _StubMesh:
            def __init__(self, size=1):
                self._size = size

            def size(self):
                return self._size

            def get_group(self):
                return None

            def get_local_rank(self):
                return 0

            def __getitem__(self, key):
                return _StubMesh(self._size)

        class _StubParallelDims:
            def __init__(self):
                self.dp_replicate = 1
                self.dp_shard = 1
                self.tp = 1
                self.pp = 1
                self.ep = 1
                self.world_size = 1
                self.fsdp_enabled = False
                self.dp_replicate_enabled = False
                self.pp_enabled = False
                self.ep_enabled = False

            def get_mesh(self, name):
                return _StubMesh(1)

            def get_optional_mesh(self, name):
                return None

        def _stub_init_parallel_dims(titan_config):
            return _StubParallelDims(), 0, 0

        def _stub_parallelize(model, parallel_dims, titan_config, **kwargs):
            return model

        def _stub_build_pipeline(model, parallel_dims, titan_config, **kwargs):
            class _Schedule:
                def step(self, *args, **kwargs):
                    return None

            return _Schedule(), True

        pu.init_parallel_dims = _stub_init_parallel_dims
        pu.parallel_strategies = {"qwen3": _stub_parallelize}
        pu.build_pipeline = _stub_build_pipeline
        pu._StubMesh = _StubMesh
        pu._StubParallelDims = _StubParallelDims
        sys.modules[pu_name] = pu
