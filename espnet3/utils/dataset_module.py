"""Helpers for dataset modules exposed under ``egs3.<name>.dataset``."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(config: Any) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return dict(OmegaConf.to_container(config, resolve=False))
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return {
            key: value
            for key, value in vars(config).items()
            if not key.startswith("_") and value is not None
        }
    raise TypeError(f"Unsupported dataset config type: {type(config)}")


def dataset_module_name(dataset_name: str) -> str:
    return f"egs3.{dataset_name}.dataset"


def load_dataset_module(dataset_name: str):
    return import_module(dataset_module_name(dataset_name))


def get_dataset_builder_class(dataset_name: str):
    module = load_dataset_module(dataset_name)
    return getattr(module, "DatasetBuilder")


def get_dataset_class(dataset_name: str):
    module = load_dataset_module(dataset_name)
    return getattr(module, "Dataset")


def get_dataset_root(dataset_name: str, recipe_dir: str | Path | None = None) -> Path:
    module = load_dataset_module(dataset_name)
    if hasattr(module, "get_dataset_root"):
        return Path(module.get_dataset_root(recipe_dir=recipe_dir))

    base_dir = Path(recipe_dir) if recipe_dir is not None else Path.cwd()
    return base_dir / "dataset" / dataset_name


def instantiate_dataset_reference(
    config: Mapping[str, Any] | DictConfig,
    recipe_dir: str | Path | None = None,
):
    plain = _to_plain_dict(config)
    dataset_name = plain.pop("dataset")
    plain.pop("name", None)
    plain.pop("transform", None)
    dataset_cls = get_dataset_class(dataset_name)
    if "recipe_dir" not in plain:
        plain["recipe_dir"] = recipe_dir
    return dataset_cls(**plain)


def ensure_dataset_reference_prepared(
    config: Mapping[str, Any] | DictConfig,
    recipe_dir: str | Path | None = None,
) -> None:
    plain = _to_plain_dict(config)
    dataset_name = plain.pop("dataset")
    plain.pop("name", None)
    plain.pop("transform", None)
    plain.pop("split", None)
    builder_cls = get_dataset_builder_class(dataset_name)
    builder = builder_cls()
    if "recipe_dir" not in plain:
        plain["recipe_dir"] = recipe_dir
    builder.build(**plain)
