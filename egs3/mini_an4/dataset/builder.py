"""Dataset builder for Mini AN4."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from omegaconf import OmegaConf

from espnet3.utils.dataset_builder import DatasetBuilder

from .prepare import create_dataset


def _load_config():
    resource = resources.files(__package__).joinpath("config.yaml")
    with resources.as_file(resource) as path:
        return OmegaConf.load(path)


def get_dataset_root(recipe_dir: str | Path | None = None) -> Path:
    cfg = _load_config()
    base = Path(recipe_dir) if recipe_dir is not None else Path.cwd()
    return base / "dataset" / str(cfg.dataset_name)


class MiniAN4Builder(DatasetBuilder):
    """Prepare Mini AN4 manifests and audio artifacts if missing."""

    def build(self, recipe_dir: str | Path | None = None, **kwargs) -> None:
        cfg = _load_config()
        dataset_dir = get_dataset_root(recipe_dir=recipe_dir)
        if self.is_prepared(dataset_dir):
            return

        archive_path = Path(__file__).resolve().parent / str(cfg.archive_path)
        create_dataset(
            dataset_dir=dataset_dir,
            archive_path=archive_path.resolve(),
        )

    @staticmethod
    def is_prepared(dataset_dir: str | Path) -> bool:
        cfg = _load_config()
        root = Path(dataset_dir)
        return all((root / relpath).is_file() for relpath in cfg.required_files)
