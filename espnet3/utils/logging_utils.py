"""Logging helpers for espnet3 experiments."""

from __future__ import annotations

import contextvars
import logging
import os
import shlex
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from shutil import which
from typing import Iterable, Mapping

from humanfriendly import format_number, format_size

from espnet2.samplers.abs_sampler import AbsSampler

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

LOG_FORMAT = (
    "[%(hostname)s] %(asctime)s (%(filename)s:%(lineno)d) "
    "%(levelname)s:\t[%(stage)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOG_STAGE = contextvars.ContextVar("espnet3_log_stage", default="main")
_BASE_RECORD_FACTORY = logging.getLogRecordFactory()


def _record_factory(*args, **kwargs):
    record = _BASE_RECORD_FACTORY(*args, **kwargs)
    record.stage = _LOG_STAGE.get()
    record.hostname = socket.gethostname()
    return record


def _ensure_log_record_factory() -> None:
    if logging.getLogRecordFactory() is not _record_factory:
        logging.setLogRecordFactory(_record_factory)


_ensure_log_record_factory()


@contextmanager
def log_stage(name: str):
    """Temporarily set the logging stage label used in log records."""
    token = _LOG_STAGE.set(name)
    try:
        yield
    finally:
        _LOG_STAGE.reset(token)


def _log(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args,
    stacklevel: int = 2,
    **kwargs,
) -> None:
    logger.log(level, msg, *args, stacklevel=stacklevel, **kwargs)


def _next_rotated_log_path(target: Path) -> Path:
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

    _ensure_log_record_factory()
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
                rotated = _next_rotated_log_path(target)
                os.replace(target, rotated)
            file_handler = logging.FileHandler(target)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    logging.captureWarnings(True)
    return logging.getLogger("espnet3")


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
    head = _run_git_command(["git", "rev-parse", "HEAD"], cwd)
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd)
    short = _run_git_command(["git", "rev-parse", "--short", "HEAD"], cwd)
    status = _run_git_command(["git", "status", "--short"], cwd)

    dirty = "clean"
    if status is None:
        dirty = "unknown"
    elif status:
        dirty = "dirty"

    meta: dict[str, str] = {}
    if head:
        meta["commit"] = head
    if short:
        meta["short_commit"] = short
    if branch:
        meta["branch"] = branch
    meta["worktree"] = dirty
    return meta


def format_command(argv: Iterable[str] | None = None) -> str:
    """Format command arguments into a shell-escaped string."""
    argv = list(argv) if argv is not None else sys.argv
    return " ".join(shlex.quote(str(a)) for a in argv)


def _run_pip_freeze() -> str | None:
    """Return `pip freeze` output, or None on failure."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        if which("uv") is None:
            return None
        try:
            completed = subprocess.run(
                ["uv", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.stdout.strip()
        except Exception:
            return None


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
        _log(
            logger,
            logging.WARNING,
            "Skipping requirements export: no file logger configured.",
        )
        return

    requirements = _run_pip_freeze()
    if requirements is None:
        _log(logger, logging.WARNING, "Failed to export requirements via pip freeze.")
        return

    target = log_dir / "requirements.txt"
    target.write_text(requirements + "\n", encoding="utf-8")
    _log(logger, logging.INFO, "Wrote requirements snapshot: %s", target)


def log_run_metadata(
    logger: logging.Logger,
    *,
    argv: Iterable[str] | None = None,
    workdir: Path | None = None,
    configs: Mapping[str, Path | None] | None = None,
    write_requirements: bool = False,
) -> None:
    """Log runtime metadata for the current run.

    Logged fields include:
      - Start timestamp.
      - Python executable and command-line arguments.
      - Working directory.
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
            workdir=Path("/home/user/espnet3"),
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
        workdir (Path | None): Working directory to report.
        configs (Mapping[str, Path | None] | None): Named config paths to log.
        write_requirements (bool): If True, export pip freeze output to
            requirements.txt alongside the log file.
    """
    logger.info("=== ESPnet3 run started: %s ===", datetime.now().isoformat())
    _log(
        logger,
        logging.INFO,
        "=== ESPnet3 run started: %s ===",
        datetime.now().isoformat(),
    )
    _log(logger, logging.INFO, "Command: %s %s", sys.executable, format_command(argv))
    _log(logger, logging.INFO, "Python: %s", sys.version.replace("\n", " "))

    cwd = workdir or Path.cwd()
    _log(logger, logging.INFO, "Working directory: %s", cwd)

    if configs:
        for name, path in configs.items():
            if path is None:
                continue
            _log(logger, logging.INFO, "%s config: %s", name, Path(path).resolve())

    git_info = get_git_metadata(cwd)
    if git_info:
        git_parts = [f"{k}={v}" for k, v in git_info.items()]
        _log(logger, logging.INFO, "Git: %s", ", ".join(git_parts))

    if write_requirements:
        _write_requirements_snapshot(logger)


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
    _log(logger, logging.INFO, "Cluster env:\n%s", cluster_dump)
    _log(logger, logging.INFO, "Runtime env:\n%s", runtime_dump)

    if torch is None:
        _log(logger, logging.INFO, "PyTorch: unavailable")
        return

    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None

    _log(
        logger,
        logging.INFO,
        "PyTorch: version=%s, cuda.available=%s, cudnn.version=%s, "
        "cudnn.benchmark=%s, cudnn.deterministic=%s",
        getattr(torch, "__version__", "unknown"),
        torch.cuda.is_available(),
        cudnn_version,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )


def _format_param_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    if value >= 1_000:
        return f"{value / 1_000:.2f} K"
    return format_number(value)


def _format_size(num_bytes: int) -> str:
    return format_size(num_bytes)


def _collect_param_name_map(model) -> dict[int, str]:
    return {id(p): name for name, p in model.named_parameters()}


def _summarize_param_modules(model, params: Iterable) -> str | None:
    try:
        param_name_map = _collect_param_name_map(model)
    except Exception:
        return None

    counts: dict[str, int] = {}
    total = 0
    for p in params:
        numel = int(getattr(p, "numel", lambda: 0)())
        total += numel
        name = param_name_map.get(id(p))
        if not name:
            key = "unknown"
        else:
            key = name.split(".", 1)[0]
        counts[key] = counts.get(key, 0) + numel

    if total == 0:
        return None

    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    parts = [f"{k}({v / total * 100:.1f}%)" for k, v in items]
    return ", ".join(parts)


def _qualified_name(obj) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__name__}"


def _summarize_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return repr(value)
    if isinstance(value, Path):
        return repr(str(value))
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                parts.append(repr(item))
            elif isinstance(item, Path):
                parts.append(repr(str(item)))
            else:
                parts.append(_qualified_name(item))
        return f"[{', '.join(parts)}]"
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{k}={_summarize_value(v)}")
        return "{" + ", ".join(items) + "}"
    return _qualified_name(value)


def _summarize_attrs(obj) -> str:
    if not hasattr(obj, "__dict__"):
        return ""
    items = []
    for key in sorted(obj.__dict__.keys()):
        value = obj.__dict__[key]
        if key.startswith("_"):
            continue
        items.append(f"{key}={_summarize_value(value)}")
    return ", ".join(items)


def _callable_name(fn) -> str:
    if hasattr(fn, "__module__") and hasattr(fn, "__name__"):
        return f"{fn.__module__}.{fn.__name__}"
    return _qualified_name(fn)


def _iter_attrs(obj) -> Iterable[tuple[str, object]]:
    if not hasattr(obj, "__dict__"):
        return []
    return sorted(
        ((k, v) for k, v in obj.__dict__.items() if not k.startswith("_")),
        key=lambda kv: kv[0],
    )


def _is_iterable_but_not_str(value) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes))


def _is_leaf_value(value) -> bool:
    return isinstance(value, (str, bytes, int, float, bool, Path)) or value is None


def _has_custom_repr(obj) -> bool:
    obj_repr = obj.__class__.__repr__
    return obj_repr is not object.__repr__


def _dump_attrs(
    logger: logging.Logger,
    obj,
    *,
    indent: str,
    depth: int,
    max_depth: int,
    seen: set[int],
    skip_custom_repr: bool = False,
) -> None:
    if depth > max_depth:
        _log(logger, logging.INFO, "%s...", indent, stacklevel=3)
        return
    obj_id = id(obj)
    if obj_id in seen:
        _log(logger, logging.INFO, "%s<recursive>", indent, stacklevel=3)
        return
    seen.add(obj_id)

    if _has_custom_repr(obj) and not skip_custom_repr:
        _log(logger, logging.INFO, "%s%s", indent, obj, stacklevel=3)
        return

    for key, value in _iter_attrs(obj):
        if isinstance(value, torch.nn.Module):
            _log(
                logger,
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                _qualified_name(value),
                stacklevel=3,
            )
            continue
        if key == "batches":
            _log(
                logger,
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                value,
                stacklevel=3,
            )
            continue
        if _is_iterable_but_not_str(value):
            items = []
            try:
                for item in value:
                    items.append(_summarize_value(item))
            except Exception:
                items = None
            if items is None:
                rendered = _qualified_name(value)
            else:
                rendered = "[" + ", ".join(items) + "]"
            _log(
                logger,
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                rendered,
                stacklevel=3,
            )
            continue
        if _is_leaf_value(value):
            _log(
                logger,
                logging.INFO,
                "%s%s: %s",
                indent,
                key,
                _summarize_value(value),
                stacklevel=3,
            )
            continue
        _log(
            logger,
            logging.INFO,
            "%s%s: %s",
            indent,
            key,
            _qualified_name(value),
            stacklevel=3,
        )
        _dump_attrs(
            logger,
            value,
            indent=indent + "  ",
            depth=depth + 1,
            max_depth=max_depth,
            seen=seen,
            skip_custom_repr=True,
        )


def log_training_summary(
    logger: logging.Logger,
    model,
    *,
    optimizer=None,
    scheduler=None,
) -> None:
    """Log model architecture/summary and optimizer/scheduler details."""
    _log(logger, logging.INFO, "Model:\n%s", model, stacklevel=2)

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

    _log(logger, logging.INFO, "Model summary:", stacklevel=2)
    _log(logger, logging.INFO, "    Class Name: %s", type(model).__name__, stacklevel=2)
    _log(
        logger,
        logging.INFO,
        "    Total Number of model parameters: %s",
        _format_param_count(total_params),
        stacklevel=2,
    )
    _log(
        logger,
        logging.INFO,
        "    Number of trainable parameters: %s (%.1f%%)",
        _format_param_count(trainable_params),
        (trainable_params / total_params * 100.0) if total_params else 0.0,
        stacklevel=2,
    )
    _log(logger, logging.INFO, "    Size: %s", _format_size(size_bytes), stacklevel=2)
    _log(logger, logging.INFO, "    Type: %s", dtype_desc, stacklevel=2)

    if optimizer is None:
        return

    try:
        from espnet3.components.optimizers.multiple_optimizer import MultipleOptimizer
    except Exception:
        MultipleOptimizer = None

    optimizers = []
    if MultipleOptimizer is not None and isinstance(optimizer, MultipleOptimizer):
        optimizers = list(optimizer.optimizers)
    else:
        optimizers = [optimizer]

    for idx, optim in enumerate(optimizers):
        _log(logger, logging.INFO, "Optimizer[%d]:", idx, stacklevel=2)
        _log(logger, logging.INFO, "%s", optim, stacklevel=2)
        try:
            all_params = [p for g in optim.param_groups for p in g.get("params", [])]
        except Exception:
            all_params = []
        module_summary = _summarize_param_modules(model, all_params)
        if module_summary:
            _log(logger, logging.INFO, "    modules: %s", module_summary, stacklevel=2)

    if scheduler is None:
        return

    if isinstance(scheduler, list):
        schedulers = scheduler
    else:
        schedulers = [scheduler]
    for idx, sch in enumerate(schedulers):
        _log(logger, logging.INFO, "Scheduler[%d]:", idx, stacklevel=2)
        _log(logger, logging.INFO, "%s", sch, stacklevel=2)


def _log_dataset(
    logger: logging.Logger,
    dataset,
    *,
    label: str,
    indent: str = "  ",
    depth: int = 0,
) -> None:
    from espnet3.components.data.dataset import CombinedDataset, DatasetWithTransform

    prefix = indent * (depth + 1)
    _log(
        logger,
        logging.INFO,
        "%s%s class: %s",
        indent * depth,
        label,
        _qualified_name(dataset),
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
            _log(logger, logging.INFO, "%scombined[%d]:", prefix, i, stacklevel=3)
            _log(
                logger,
                logging.INFO,
                "%stransform: %s",
                indent * (depth + 2),
                _callable_name(transform),
                stacklevel=3,
            )
            _log(
                logger,
                logging.INFO,
                "%spreprocessor: %s",
                indent * (depth + 2),
                _callable_name(preprocessor),
                stacklevel=3,
            )
            _log(
                logger,
                logging.INFO,
                "%sdataset class: %s",
                indent * (depth + 2),
                _qualified_name(child),
                stacklevel=3,
            )
            _log(
                logger,
                logging.INFO,
                "%slen: %s",
                indent * (depth + 3),
                len(child),
                stacklevel=3,
            )
    elif isinstance(dataset, DatasetWithTransform):
        _log(
            logger,
            logging.INFO,
            "%stransform: %s",
            indent * (depth + 1),
            _callable_name(dataset.transform),
            stacklevel=3,
        )
        _log(
            logger,
            logging.INFO,
            "%spreprocessor: %s",
            indent * (depth + 1),
            _callable_name(dataset.preprocessor),
            stacklevel=3,
        )
        child = dataset.dataset
        _log(
            logger,
            logging.INFO,
            "%sdataset class: %s",
            indent * (depth + 1),
            _qualified_name(child),
            stacklevel=3,
        )
        _log(
            logger,
            logging.INFO,
            "%slen: %s",
            indent * (depth + 2),
            len(child),
            stacklevel=3,
        )


def log_data_organizer(logger: logging.Logger, data_organizer) -> None:
    """Log dataset organizer and dataset details."""
    _log(
        logger,
        logging.INFO,
        "Data organizer: %s",
        _qualified_name(data_organizer),
        stacklevel=2,
    )

    train = getattr(data_organizer, "train", None)
    valid = getattr(data_organizer, "valid", None)
    test = getattr(data_organizer, "test", None)

    if train is None:
        _log(logger, logging.INFO, "train dataset: None", stacklevel=2)
    else:
        _log_dataset(logger, train, label="train")

    if valid is None:
        _log(logger, logging.INFO, "valid dataset: None", stacklevel=2)
    else:
        _log_dataset(logger, valid, label="valid")

    if not test:
        _log(logger, logging.INFO, "test datasets: None", stacklevel=2)
        return

    if isinstance(test, dict):
        _log(logger, logging.INFO, "test datasets: %d", len(test), stacklevel=2)
        for name, ds in test.items():
            _log_dataset(logger, ds, label=f"test[{name}]")
    else:
        _log_dataset(logger, test, label="test")


def log_dataloader(
    logger: logging.Logger,
    loader,
    *,
    label: str,
    sampler=None,
    batch_sampler=None,
    iter_factory=None,
    batches=None,
) -> None:
    """Log dataloader/iterator details including samplers and iter factories."""
    if hasattr(loader, "dataset") and hasattr(loader, "batch_size"):
        dataset = getattr(loader, "dataset", None)
        try:
            dataset_len = len(dataset) if dataset is not None else None
        except Exception:
            dataset_len = None
        dataset_name = _qualified_name(dataset) if dataset is not None else "None"
        if dataset_len is not None:
            dataset_desc = f"{dataset_name}(len={dataset_len})"
        else:
            dataset_desc = dataset_name

        sampler_obj = getattr(loader, "sampler", None)
        sampler_name = (
            _qualified_name(sampler_obj) if sampler_obj is not None else "None"
        )
        batch_sampler_obj = getattr(loader, "batch_sampler", None)
        batch_sampler_name = (
            _qualified_name(batch_sampler_obj)
            if batch_sampler_obj is not None
            else "None"
        )
        collate_fn = getattr(loader, "collate_fn", None)
        collate_name = _callable_name(collate_fn) if collate_fn is not None else "None"
        multiprocessing_ctx = getattr(loader, "multiprocessing_context", None)
        if multiprocessing_ctx is not None:
            multiprocessing_ctx = getattr(
                multiprocessing_ctx, "get_start_method", lambda: multiprocessing_ctx
            )()

        lines = [
            "[DataLoader]",
            f"  dataset           : {dataset_desc}",
            f"  batch_size        : {getattr(loader, 'batch_size', None)}",
            f"  shuffle           : {getattr(loader, 'shuffle', None)}",
            f"  sampler           : {sampler_name}",
            f"  batch_sampler     : {batch_sampler_name}",
            f"  num_workers       : {getattr(loader, 'num_workers', None)}",
            f"  drop_last         : {getattr(loader, 'drop_last', None)}",
            f"  pin_memory        : {getattr(loader, 'pin_memory', None)}",
            f"  persistent_workers: {getattr(loader, 'persistent_workers', None)}",
            f"  prefetch_factor   : {getattr(loader, 'prefetch_factor', None)}",
            f"  timeout           : {getattr(loader, 'timeout', None)}",
            f"  multiprocessing   : {multiprocessing_ctx}",
            f"  collate_fn        : {collate_name}",
        ]
        _log(
            logger,
            logging.INFO,
            "DataLoader[%s]:\n%s",
            label,
            "\n".join(lines),
            stacklevel=2,
        )
    else:
        _log(
            logger,
            logging.INFO,
            "DataLoader[%s] class: %s",
            label,
            _qualified_name(loader),
            stacklevel=2,
        )
        _log(logger, logging.INFO, "DataLoader[%s]:\n%s", label, loader, stacklevel=2)

    if sampler is not None:
        _log(
            logger,
            logging.INFO,
            "Sampler[%s] class: %s",
            label,
            _qualified_name(sampler),
            stacklevel=2,
        )
        _log(logger, logging.INFO, "Sampler[%s]: %s", label, sampler, stacklevel=2)
        _dump_attrs(
            logger,
            sampler,
            indent="  ",
            depth=0,
            max_depth=2,
            seen=set(),
        )

    if batch_sampler is not None:
        _log(
            logger,
            logging.INFO,
            "BatchSampler[%s] class: %s",
            label,
            _qualified_name(batch_sampler),
            stacklevel=2,
        )
        _log(
            logger,
            logging.INFO,
            "BatchSampler[%s]: %s",
            label,
            batch_sampler,
            stacklevel=2,
        )
        _dump_attrs(
            logger,
            batch_sampler,
            indent="  ",
            depth=0,
            max_depth=2,
            seen=set(),
        )

    if iter_factory is not None:
        _log(
            logger,
            logging.INFO,
            "IterFactory[%s]: %s",
            label,
            _qualified_name(iter_factory),
            stacklevel=2,
        )
        _dump_attrs(
            logger,
            iter_factory,
            indent="  ",
            depth=0,
            max_depth=3,
            seen=set(),
        )

    if batches is not None:
        if isinstance(batches, AbsSampler):
            _log(
                logger,
                logging.INFO,
                "IterBatches[%s] class: %s",
                label,
                _qualified_name(batches),
                stacklevel=2,
            )
            _log(
                logger,
                logging.INFO,
                "IterBatches[%s]: %s",
                label,
                batches,
                stacklevel=2,
            )
        try:
            batch_count = len(batches)
        except Exception:
            batch_count = None
        if batch_count is not None:
            _log(
                logger,
                logging.INFO,
                "IterBatches[%s]: %d batches",
                label,
                batch_count,
                stacklevel=2,
            )
            if batch_count:
                try:
                    first_batch = batches[0]
                    _log(
                        logger,
                        logging.INFO,
                        "IterBatches[%s]: first batch size=%s",
                        label,
                        len(first_batch),
                        stacklevel=2,
                    )
                except Exception:
                    pass
