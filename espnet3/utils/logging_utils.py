"""Logging helpers for espnet3 experiments."""

from __future__ import annotations

import contextvars
import logging
import os
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Mapping

import torch
from humanfriendly import format_number, format_size

from espnet3.components.data.dataset import CombinedDataset, DatasetWithTransform
from espnet3.components.optimizers.multiple_optimizer import MultipleOptimizer

LOG_FORMAT = (
    "[%(hostname)s] %(asctime)s (%(filename)s:%(lineno)d) "
    "%(levelname)s:\t[%(stage)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# =============================================================================
# Logging Record Setup
# =============================================================================
_LOG_STAGE = contextvars.ContextVar("espnet3_log_stage", default="main")
_LOGGED_DATALOADER: bool = False
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


def _build_record(*args, **kwargs):
    # Inject custom fields used by LOG_FORMAT (stage/hostname) into each LogRecord.
    record = _BASE_RECORD_FACTORY(*args, **kwargs)
    record.stage = _LOG_STAGE.get()
    record.hostname = socket.gethostname()
    return record


logging.setLogRecordFactory(_build_record)


@contextmanager
def log_stage(name: str):
    """Temporarily set the logging stage label used in log records."""
    token = _LOG_STAGE.set(name)
    try:
        yield
    finally:
        _LOG_STAGE.reset(token)


# =============================================================================
# Logging Configuration (Handlers/Formatters)
# =============================================================================
def _get_next_rotated_log_path(target: Path) -> Path:
    """Return the next available rotated log path (e.g., run1.log)."""
    suffixes = target.suffixes
    suffix = "".join(suffixes)
    base = target.name[: -len(suffix)] if suffix else target.name

    index = 1
    while True:
        candidate = target.with_name(f"{base}{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def configure_logging(
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure logging for an ESPnet3 run.

    This sets up:
      - A root logger with a stream handler (console).
      - An optional file handler at `log_dir/filename`.
      - If `log_dir/filename` already exists, it is rotated to the next
        available suffix (e.g., `run1.log`) and a fresh `run.log` is created.
      - Warning capture into the logging system.

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging

        logger = configure_logging(log_dir=Path("exp/run1"), level=logging.INFO)
        logger.info("hello")
        ```

    Example log output:
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | hello
        ```

    Example directory tree (when `log_dir/filename` already exists):
        ```
        exp/run1/
        ├── run.log          # new logs (current run)
        └── run1.log         # rotated older logs
        ```

    Args:
        log_dir (Path | None): Directory to store the log file.
            If None, only console logging is configured.
        level (int): Logging level (e.g., logging.INFO).
        filename (str): Log file name when `log_dir` is provided.

    Returns:
        logging.Logger: Logger instance named "espnet3".
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        target = (log_dir / filename).resolve()
        has_file = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None)
            and Path(h.baseFilename).resolve() == target
            for h in root.handlers
        )
        if not has_file:
            if target.exists():
                rotated = _get_next_rotated_log_path(target)
                os.replace(target, rotated)
            file_handler = logging.FileHandler(target)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    logging.captureWarnings(True)
    return logging.getLogger("espnet3")


# =============================================================================
# Run Metadata (Command/Git/Requirements)
# =============================================================================
def _run_git_command(cmd: list[str], cwd: Path | None) -> str | None:
    """Run a git command and return stdout, or None on failure."""
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def get_git_metadata(cwd: Path | None = None) -> dict[str, str]:
    """Return git metadata for the current repository.

    This attempts to read commit hash, short hash, branch name, and worktree
    status from the git repository rooted at `cwd`.

    Args:
        cwd (Path | None): Directory within the target git repo.

    Returns:
        dict[str, str]: Collected metadata keys, possibly including:
            - "commit": Full commit hash.
            - "short_commit": Abbreviated commit hash.
            - "branch": Current branch name.
            - "worktree": "clean", "dirty", or "unknown".
    """
    cwd = cwd or Path.cwd()
    status = _run_git_command(["git", "status", "--short"], cwd)

    dirty = "clean"
    if status is None:
        dirty = "unknown"
    elif status:
        dirty = "dirty"

    meta = {
        "commit": _run_git_command(["git", "rev-parse", "HEAD"], cwd),
        "short_commit": _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd),
        "branch": _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd),
    }
    meta["worktree"] = dirty
    return meta


def _run_pip_freeze() -> str:
    """Return dependency snapshot output."""
    if which("uv") is not None:
        completed = subprocess.run(
            ["uv", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _get_log_dir_from_logger(logger: logging.Logger) -> Path | None:
    """Return the log directory from any configured file handler."""
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ):
            return Path(handler.baseFilename).resolve().parent

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(
            handler, "baseFilename", None
        ):
            return Path(handler.baseFilename).resolve().parent

    return None


def _write_requirements_snapshot(logger: logging.Logger) -> None:
    """Write a requirements snapshot alongside the configured log file."""
    log_dir = _get_log_dir_from_logger(logger)
    if log_dir is None:
        logger.log(
            logging.WARNING,
            "Skipping requirements export: no file logger configured.",
            stacklevel=2,
        )
        return

    requirements = _run_pip_freeze()

    target = log_dir / "requirements.txt"
    target.write_text(requirements + "\n", encoding="utf-8")
    logger.log(
        logging.INFO,
        "Wrote requirements snapshot: %s",
        target,
        stacklevel=2,
    )


def log_run_metadata(
    logger: logging.Logger,
    *,
    argv: Iterable[str] | None = None,
    configs: Mapping[str, Path | None] | None = None,
    write_requirements: bool = False,
) -> None:
    """Log runtime metadata for the current run.

    Logged fields include:
      - Start timestamp.
      - Python executable and command-line arguments.
      - Working directory (current process directory).
      - Python version.
      - Config paths (if provided).
      - Git metadata (commit/branch/dirty), when available.
      - Optional requirements snapshot (pip freeze).

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging, log_run_metadata

        logger = configure_logging(log_dir=Path("exp/run1"))
        log_run_metadata(
            logger,
            argv=["espnet3-train", "--config", "conf/train.yaml"],
        configs={"train": Path("conf/train.yaml")},
        )
        ```

    Example log output (wrapped for readability, <= 88 chars):
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | === ESPnet3 run started: \
            2026-02-04T10:15:22 ===
        2026-02-04 10:15:22 | INFO | espnet3 | Command: /usr/bin/python3 \
            espnet3-train --config conf/train.yaml
        2026-02-04 10:15:22 | INFO | espnet3 | Python: 3.10.12 (GCC 11.4.0)
        2026-02-04 10:15:22 | INFO | espnet3 | Working directory: /home/user/espnet3
        2026-02-04 10:15:22 | INFO | espnet3 | train config: /home/user/espnet3/conf/\
            train.yaml
        2026-02-04 10:15:22 | INFO | espnet3 | Git: commit=0123456789abcdef, \
            short_commit=0123456, branch=main, worktree=clean
        ```

    Args:
        logger (logging.Logger): Logger used to emit metadata.
        argv (Iterable[str] | None): Command arguments; defaults to sys.argv.
        configs (Mapping[str, Path | None] | None): Named config paths to log.
        write_requirements (bool): If True, export pip freeze output to
            requirements.txt alongside the log file.
    """
    logger.info("=== ESPnet3 run started: %s ===", datetime.now().isoformat())
    logger.log(
        logging.INFO,
        "=== ESPnet3 run started: %s ===",
        datetime.now().isoformat(),
        stacklevel=2,
    )
    cmd_argv = list(argv) if argv is not None else sys.argv
    cmd_text = " ".join(shlex.quote(str(a)) for a in cmd_argv)
    logger.log(
        logging.INFO,
        "Command: %s %s",
        sys.executable,
        cmd_text,
        stacklevel=2,
    )
    logger.log(
        logging.INFO,
        "Python: %s",
        sys.version.replace("\n", " "),
        stacklevel=2,
    )

    cwd = Path.cwd()
    logger.log(logging.INFO, "Working directory: %s", cwd, stacklevel=2)

    if configs:
        for name, path in configs.items():
            if path is None:
                continue
            logger.log(
                logging.INFO,
                "%s config: %s",
                name,
                Path(path).resolve(),
                stacklevel=2,
            )

    git_info = get_git_metadata(cwd)
    if git_info:
        git_parts = [f"{k}={v}" for k, v in git_info.items()]
        logger.log(logging.INFO, "Git: %s", ", ".join(git_parts), stacklevel=2)

    if write_requirements:
        _write_requirements_snapshot(logger)


# =============================================================================
# Environment Metadata (Cluster/Runtime/Torch)
# =============================================================================
def _collect_env(
    *,
    prefixes: Iterable[str] | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, str]:
    """Collect environment variables matching prefixes or explicit keys.

    Args:
        prefixes (Iterable[str] | None): Prefixes to match (e.g., "CUDA_").
        keys (Iterable[str] | None): Exact variable names to include.

    Returns:
        dict[str, str]: Sorted environment variables that match.
    """
    prefixes = tuple(prefixes or ())
    key_set = {k for k in (keys or ())}
    collected: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in key_set or any(name.startswith(prefix) for prefix in prefixes):
            collected[name] = value
    return dict(sorted(collected.items()))


def log_env_metadata(
    logger: logging.Logger,
    *,
    cluster_prefixes: Iterable[str] | None = None,
    runtime_prefixes: Iterable[str] | None = None,
    runtime_keys: Iterable[str] | None = None,
) -> None:
    """Log selected cluster and runtime environment variables.

    The output includes two blocks:
      - Cluster environment variables (scheduler/runtime IDs).
      - Runtime environment variables (CUDA/NCCL/OMP/PATH, etc.).

    Environment variables collected by default:

    Cluster prefixes:
    | Prefix | Purpose |
    |---|---|
    | `SLURM_` | Slurm job/step metadata (job id, task id, node info) |
    | `PBS_` | PBS/Torque job metadata |
    | `LSF_` | LSF job metadata |
    | `SGE_` | SGE job metadata |
    | `COBALT_` | Cobalt job metadata |
    | `OMPI_` | Open MPI runtime metadata |
    | `PMI_` | PMI (Process Management Interface) metadata |
    | `MPI_` | MPI runtime metadata (generic prefix) |

    Runtime prefixes:
    | Prefix | Purpose |
    |---|---|
    | `NCCL_` | NCCL configuration (multi-GPU comms) |
    | `CUDA_` | CUDA runtime configuration |
    | `ROCM_` | ROCm runtime configuration |
    | `OMP_` | OpenMP threading configuration |
    | `MKL_` | Intel MKL configuration |
    | `OPENBLAS_` | OpenBLAS configuration |
    | `UCX_` | UCX communication configuration |
    | `NVIDIA_` | NVIDIA runtime configuration |

    Explicit runtime keys:
    | Key | Purpose |
    |---|---|
    | `PATH` | Executable search path |
    | `PYTHONPATH` | Python module search path |
    | `LD_LIBRARY_PATH` | Shared library search path |
    | `CUDA_VISIBLE_DEVICES` | GPU visibility mask |
    | `RANK` | Global rank (distributed) |
    | `LOCAL_RANK` | Local rank on node |
    | `NODE_RANK` | Node rank in job |
    | `WORLD_SIZE` | Total process count |
    | `MASTER_ADDR` | Distributed master address |
    | `MASTER_PORT` | Distributed master port |

    Example usage:
        ```python
        from pathlib import Path
        from espnet3.utils.logging_utils import configure_logging, log_env_metadata

        logger = configure_logging(log_dir=Path("exp/run1"))
        log_env_metadata(logger)
        ```

    Example log output:
        ```
        2026-02-04 10:15:22 | INFO | espnet3 | Cluster env:
        SLURM_JOB_ID=12345
        SLURM_PROCID=0
        2026-02-04 10:15:22 | INFO | espnet3 | Runtime env:
        CUDA_VISIBLE_DEVICES=0
        NCCL_DEBUG=INFO
        PATH=/usr/local/bin:/usr/bin:/bin
        ```

    Args:
        logger (logging.Logger): Logger used to emit metadata.
        cluster_prefixes (Iterable[str] | None): Prefixes for cluster variables.
        runtime_prefixes (Iterable[str] | None): Prefixes for runtime variables.
        runtime_keys (Iterable[str] | None): Explicit runtime keys to include.
    """
    cluster_prefixes = cluster_prefixes or (
        "SLURM_",
        "PBS_",
        "LSF_",
        "SGE_",
        "COBALT_",
        "OMPI_",
        "PMI_",
        "MPI_",
    )
    runtime_prefixes = runtime_prefixes or (
        "NCCL_",
        "CUDA_",
        "ROCM_",
        "OMP_",
        "MKL_",
        "OPENBLAS_",
        "UCX_",
        "NVIDIA_",
    )
    runtime_keys = runtime_keys or (
        "PATH",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    )

    cluster_env = _collect_env(prefixes=cluster_prefixes)
    runtime_env = _collect_env(prefixes=runtime_prefixes, keys=runtime_keys)

    cluster_dump = "\n".join(f"{k}={v}" for k, v in cluster_env.items()) or "(none)"
    runtime_dump = "\n".join(f"{k}={v}" for k, v in runtime_env.items()) or "(none)"
    logger.info("Cluster env:\n%s", cluster_dump)
    logger.log(logging.INFO, "Cluster env:\n%s", cluster_dump, stacklevel=2)
    logger.log(logging.INFO, "Runtime env:\n%s", runtime_dump, stacklevel=2)

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    logger.log(
        logging.INFO,
        "PyTorch: version=%s, cuda.available=%s, cudnn.version=%s, "
        "cudnn.benchmark=%s, cudnn.deterministic=%s",
        getattr(torch, "__version__", "unknown"),
        torch.cuda.is_available(),
        cudnn_version,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        stacklevel=2,
    )


# =============================================================================
# Introspection Helpers
# =============================================================================
def _build_qualified_name(obj) -> str:
    """Return fully-qualified class name for an object or class.

    Example:
        ```python
        _build_qualified_name(Path("/tmp"))
        # => 'pathlib.PosixPath'
        ```
    """
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _build_callable_name(fn) -> str:
    """Return fully-qualified callable name when possible.

    Example:
        ```python
        _build_callable_name(len)
        # => 'builtins.len'
        ```
    """
    if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
        return f"{fn.__module__}.{fn.__name__}"
    return _build_qualified_name(fn)


def _iter_attrs(obj) -> Iterable[tuple[str, object]]:
    if not hasattr(obj, "__dict__"):
        return []
    return sorted(
        ((k, v) for k, v in obj.__dict__.items() if not k.startswith("_")),
        key=lambda kv: kv[0],
    )


def _truncate_text(text: str, *, max_len: int = 200) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _dump_attrs(
    logger: logging.Logger,
    obj,
    indent: str,
    depth: int,
    max_depth: int,
    seen: set[int],
) -> None:
    if depth > max_depth:
        logger.log(logging.INFO, "%s...", indent, stacklevel=3)
        return
    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    for key, value in _iter_attrs(obj):
        if isinstance(value, torch.nn.Module):
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                _build_qualified_name(value),
                stacklevel=3,
            )
            continue
        if isinstance(value, Iterator):
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                _truncate_text(str(value)),
                stacklevel=3,
            )
            continue
        if value is None:
            summary = "None"
        elif isinstance(value, (str, bytes, int, float, bool)):
            summary = repr(value)
        elif isinstance(value, Path):
            summary = repr(str(value))
        else:
            summary = None

        if summary is not None:
            logger.log(
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                summary,
                stacklevel=3,
            )
            continue

        logger.log(
            logging.INFO,
            "%s%s: %s",
            indent,
            key,
            _build_qualified_name(value),
            stacklevel=3,
        )
        _dump_attrs(
            logger,
            value,
            indent=indent * 2,
            depth=depth + 1,
            max_depth=max_depth,
            seen=seen,
        )


def _log_component(
    logger: logging.Logger,
    kind: str,
    label: str,
    obj,
    max_depth: int,
) -> None:
    if obj is None:
        return
    logger.log(
        logging.INFO,
        "%s[%s] class: %s",
        kind,
        label,
        _build_qualified_name(obj),
        stacklevel=2,
    )
    logger.log(logging.INFO, "%s[%s]: %s", kind, label, obj, stacklevel=2)
    _dump_attrs(
        logger,
        obj,
        indent="  ",
        depth=0,
        max_depth=max_depth,
        seen=set(),
    )


# =============================================================================
# Model/Optimizer Summary
# =============================================================================
def _summarize_param_modules(model, params: Iterable) -> str | None:
    """Summarize parameter distribution by top-level module name.

    Note: We derive `params` from the optimizer (not from `model.parameters()`)
    so the summary reflects the parameters actually optimized, which matters
    when multiple optimizers or frozen parameter subsets are used.
    """
    param_name_map = {id(p): name for name, p in model.named_parameters()}

    counts: dict[str, int] = {}
    total = 0
    for p in params:
        numel = int(p.numel())
        total += numel
        name = param_name_map[id(p)]
        key = name.split(".", 1)[0]
        counts[key] = counts.get(key, 0) + numel

    if total == 0:
        return None

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{k}({v / total * 100:.1f}%)" for k, v in items]
    return ", ".join(parts)


def log_training_summary(
    logger: logging.Logger,
    model,
    optimizer=None,
    scheduler=None,
) -> None:
    """Log model architecture/summary and optimizer/scheduler details."""
    logger.log(logging.INFO, "Model:\n%s", model, stacklevel=2)

    params = list(model.parameters())
    total_params = sum(p.numel() for p in params)
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in params)

    dtype_counts: dict[str, int] = {}
    for p in params:
        dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + p.numel()
    dtype_items = sorted(dtype_counts.items(), key=lambda kv: kv[1], reverse=True)
    dtype_desc = ", ".join(
        f"{k}({v / total_params * 100:.1f}%)" for k, v in dtype_items
    )

    logger.log(logging.INFO, "Model summary:", stacklevel=2)
    logger.log(logging.INFO, "    Class Name: %s", type(model).__name__, stacklevel=2)
    logger.log(
        logging.INFO,
        "    Total Number of model parameters: %s",
        format_number(total_params),
        stacklevel=2,
    )
    logger.log(
        logging.INFO,
        "    Number of trainable parameters: %s (%.1f%%)",
        format_number(trainable_params),
        (trainable_params / total_params * 100.0) if total_params else 0.0,
        stacklevel=2,
    )
    logger.log(logging.INFO, "    Size: %s", format_size(size_bytes), stacklevel=2)
    logger.log(logging.INFO, "    Type: %s", dtype_desc, stacklevel=2)

    optimizers = []
    if isinstance(optimizer, MultipleOptimizer):
        optimizers = list(optimizer.optimizers)
    else:
        optimizers = [optimizer]

    for idx, optim in enumerate(optimizers):
        logger.log(logging.INFO, "Optimizer[%d]:", idx, stacklevel=2)
        logger.log(logging.INFO, "%s", optim, stacklevel=2)
        all_params = [p for g in optim.param_groups for p in g.get("params", [])]
        module_summary = _summarize_param_modules(model, all_params)
        if module_summary:
            logger.log(logging.INFO, "    modules: %s", module_summary, stacklevel=2)

    if isinstance(scheduler, list):
        schedulers = scheduler
    else:
        schedulers = [scheduler]
    for idx, sch in enumerate(schedulers):
        logger.log(logging.INFO, "Scheduler[%d]:", idx, stacklevel=2)
        logger.log(logging.INFO, "%s", sch, stacklevel=2)


# =============================================================================
# Dataset/Data Organizer Logging
# =============================================================================
def _log_dataset(
    logger: logging.Logger,
    dataset,
    *,
    label: str,
    indent: str = "  ",
    depth: int = 0,
) -> None:
    prefix = indent * (depth + 1)
    logger.log(
        logging.INFO,
        "%s%s class: %s",
        indent * depth,
        label,
        _build_qualified_name(dataset),
        stacklevel=3,
    )

    if isinstance(dataset, CombinedDataset):
        _dump_attrs(
            logger,
            dataset,
            indent=prefix + "  ",
            depth=0,
            max_depth=2,
            seen=set(),
        )
        for i, (child, (transform, preprocessor)) in enumerate(
            zip(dataset.datasets, dataset.transforms)
        ):
            logger.log(logging.INFO, "%scombined[%d]:", prefix, i, stacklevel=3)
            _log_dataset_block(
                logger,
                indent=indent,
                depth=depth + 2,
                transform=transform,
                preprocessor=preprocessor,
                dataset=child,
            )
    elif isinstance(dataset, DatasetWithTransform):
        _log_dataset_block(
            logger,
            indent=indent,
            depth=depth + 1,
            transform=dataset.transform,
            preprocessor=dataset.preprocessor,
            dataset=dataset.dataset,
        )


def _log_dataset_block(
    logger: logging.Logger,
    *,
    indent: str,
    depth: int,
    transform,
    preprocessor,
    dataset,
) -> None:
    logger.log(
        logging.INFO,
        "%stransform: %s",
        indent * depth,
        _build_callable_name(transform),
        stacklevel=3,
    )
    logger.log(
        logging.INFO,
        "%spreprocessor: %s",
        indent * depth,
        _build_callable_name(preprocessor),
        stacklevel=3,
    )
    logger.log(
        logging.INFO,
        "%sdataset class: %s",
        indent * depth,
        _build_qualified_name(dataset),
        stacklevel=3,
    )
    logger.log(
        logging.INFO,
        "%slen: %s",
        indent * (depth + 1),
        len(dataset),
        stacklevel=3,
    )


def log_data_organizer(logger: logging.Logger, data_organizer) -> None:
    """Log dataset organizer and dataset details."""
    logger.log(
        logging.INFO,
        "Data organizer: %s",
        _build_qualified_name(data_organizer),
        stacklevel=2,
    )

    train = getattr(data_organizer, "train", None)
    valid = getattr(data_organizer, "valid", None)
    test = getattr(data_organizer, "test", None)

    if train is None:
        logger.log(logging.INFO, "train dataset: None", stacklevel=2)
    else:
        _log_dataset(logger, train, label="train")

    if valid is None:
        logger.log(logging.INFO, "valid dataset: None", stacklevel=2)
    else:
        _log_dataset(logger, valid, label="valid")

    if not test:
        logger.log(logging.INFO, "test datasets: None", stacklevel=2)
        return

    if isinstance(test, dict):
        logger.log(logging.INFO, "test datasets: %d", len(test), stacklevel=2)
        for name, ds in test.items():
            _log_dataset(logger, ds, label=f"test[{name}]")
    else:
        _log_dataset(logger, test, label="test")


def log_dataloader(
    logger: logging.Logger,
    loader,
    label: str,
    sampler=None,
    batch_sampler=None,
    iter_factory=None,
    batches=None,
) -> None:
    """Log dataloader/iterator details including samplers and iter factories."""
    global _LOGGED_DATALOADER
    if _LOGGED_DATALOADER:
        return
    _LOGGED_DATALOADER = True
    dataset = loader.dataset
    dataset_len = len(dataset)
    dataset_desc = f"{_build_qualified_name(dataset)}(len={dataset_len})"

    sampler_obj = getattr(loader, "sampler", None)
    sampler_name = _build_qualified_name(sampler_obj) if sampler_obj is not None else "None"
    batch_sampler_obj = getattr(loader, "batch_sampler", None)
    batch_sampler_name = (
        _build_qualified_name(batch_sampler_obj) if batch_sampler_obj is not None else "None"
    )
    collate_fn = getattr(loader, "collate_fn", None)
    collate_name = _build_callable_name(collate_fn) if collate_fn is not None else "None"
    multiprocessing_ctx = getattr(loader, "multiprocessing_context", None)
    if multiprocessing_ctx is not None:
        multiprocessing_ctx = getattr(
            multiprocessing_ctx, "get_start_method", lambda: multiprocessing_ctx
        )()

    lines = [
        "[DataLoader]",
        f"  dataset           : {dataset_desc}",
        f"  batch_size        : {loader.batch_size}",
        f"  shuffle           : {getattr(loader, 'shuffle', None)}",
        f"  sampler           : {sampler_name}",
        f"  batch_sampler     : {batch_sampler_name}",
        f"  num_workers       : {loader.num_workers}",
        f"  drop_last         : {loader.drop_last}",
        f"  pin_memory        : {loader.pin_memory}",
        f"  persistent_workers: {loader.persistent_workers}",
        f"  prefetch_factor   : {loader.prefetch_factor}",
        f"  timeout           : {loader.timeout}",
        f"  multiprocessing   : {multiprocessing_ctx}",
        f"  collate_fn        : {collate_name}",
    ]
    logger.log(
        logging.INFO,
        "DataLoader[%s]:\n%s",
        label,
        "\n".join(lines),
        stacklevel=2,
    )

    _log_component(
        logger,
        kind="Sampler",
        label=label,
        obj=sampler,
        max_depth=2,
    )
    _log_component(
        logger,
        kind="BatchSampler",
        label=label,
        obj=batch_sampler,
        max_depth=2,
    )
    _log_component(
        logger,
        kind="IterFactory",
        label=label,
        obj=iter_factory,
        max_depth=3,
    )

    if batches is not None:
        try:
            batch_count = len(batches)
        except Exception:
            batch_count = None
        logger.log(
            logging.INFO,
            "IterBatches[%s]: %s",
            label,
            _truncate_text(str(batches)),
            stacklevel=2,
        )
        if batch_count is not None:
            logger.log(
                logging.INFO,
                "IterBatches[%s]: %d batches",
                label,
                batch_count,
                stacklevel=2,
            )
