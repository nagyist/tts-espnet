from __future__ import annotations

import io
import logging as py_logging
from pathlib import Path

import torch
import torch.nn as nn

from espnet3.utils import logging_utils as elog


def _reset_logger(logger: py_logging.Logger, handlers, level, propagate: bool) -> None:
    logger.handlers = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_configure_logging_adds_console_and_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        logger = elog.configure_logging(log_dir=tmp_path, filename="run.log")
        logger.info("hello")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        stream_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.StreamHandler)
        ]

        assert (tmp_path / "run.log").exists()
        assert len(file_handlers) == 1
        assert len(stream_handlers) >= 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_is_idempotent(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        elog.configure_logging(log_dir=tmp_path, filename="run.log")
        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        file_handlers = [
            h for h in root.handlers if isinstance(h, py_logging.FileHandler)
        ]
        assert len(file_handlers) == 1
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_configure_logging_rotates_existing_file(tmp_path: Path):
    root = py_logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    old_propagate = root.propagate
    try:
        root.handlers = []
        root.propagate = False

        log_path = tmp_path / "run.log"
        log_path.write_text("old log\n", encoding="utf-8")

        elog.configure_logging(log_dir=tmp_path, filename="run.log")

        rotated = tmp_path / "run1.log"
        assert rotated.exists()
        assert rotated.read_text(encoding="utf-8") == "old log\n"
        assert log_path.exists()
    finally:
        _reset_logger(root, old_handlers, old_level, old_propagate)


def test_log_run_metadata_logs_command_and_git(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.logging")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    handler = py_logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(
        elog, "get_git_metadata", lambda cwd=None: {"commit": "abc", "branch": "main"}
    )

    try:
        elog.log_run_metadata(
            logger,
            argv=["run.py", "--arg", "1"],
            workdir=tmp_path,
            configs={"train": tmp_path / "train.yaml"},
        )
        out = stream.getvalue()
        assert "run.py --arg 1" in out
        assert str(tmp_path) in out
        assert "train.yaml" in out
        assert "commit=abc" in out and "branch=main" in out
        assert "Python:" in out
    finally:
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_run_metadata_writes_requirements(monkeypatch, tmp_path: Path):
    logger = py_logging.getLogger("espnet3.test.requirements")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    handler = py_logging.FileHandler(tmp_path / "run.log")
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False

    monkeypatch.setattr(elog, "_run_pip_freeze", lambda: "pkg==1.2.3")

    try:
        elog.log_run_metadata(logger, write_requirements=True)
        contents = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
        assert "pkg==1.2.3" in contents
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.tensor([idx], dtype=torch.float32)}

    def __len__(self):
        return 3


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def _capture_logger(name: str):
    logger = py_logging.getLogger(name)
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    handler = py_logging.StreamHandler(stream)
    logger.handlers = [handler]
    logger.setLevel(py_logging.INFO)
    logger.propagate = False
    return logger, stream, (old_handlers, old_level, old_propagate, handler)


def test_log_training_summary_includes_model_and_optimizer():
    logger, stream, cleanup = _capture_logger("espnet3.test.train_summary")
    old_handlers, old_level, old_propagate, handler = cleanup

    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    try:
        elog.log_training_summary(logger, model, optimizer=optimizer, scheduler=scheduler)
        out = stream.getvalue()
        assert "Model summary:" in out
        assert "Class Name: DummyModel" in out
        assert "Optimizer[0]:" in out
        assert "Scheduler[0]:" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_data_organizer_includes_datasets():
    logger, stream, cleanup = _capture_logger("espnet3.test.data_organizer")
    old_handlers, old_level, old_propagate, handler = cleanup

    class DummyOrganizer:
        def __init__(self):
            from espnet3.components.data.dataset import CombinedDataset

            self.train = CombinedDataset([DummyDataset()], [(lambda x: x, lambda x: x)])
            self.valid = CombinedDataset([DummyDataset()], [(lambda x: x, lambda x: x)])
            self.test = {}

    try:
        elog.log_data_organizer(logger, DummyOrganizer())
        out = stream.getvalue()
        assert "Data organizer:" in out
        assert "train class: espnet3.components.data.dataset.CombinedDataset" in out
        assert "valid class: espnet3.components.data.dataset.CombinedDataset" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_data_organizer_combined_variants():
    logger, stream, cleanup = _capture_logger("espnet3.test.data_organizer.variants")
    old_handlers, old_level, old_propagate, handler = cleanup

    def custom_transform(sample):
        return sample

    def other_transform(sample):
        return sample

    def custom_preprocessor(sample):
        return sample

    class DummyOrganizer:
        def __init__(self):
            from espnet3.components.data.dataset import CombinedDataset

            self.train = CombinedDataset(
                [DummyDataset(), DummyDataset()],
                [(custom_transform, custom_preprocessor), (other_transform, None)],
            )
            self.valid = CombinedDataset(
                [DummyDataset()],
                [(custom_transform, None)],
            )
            self.test = {}

    try:
        elog.log_data_organizer(logger, DummyOrganizer())
        out = stream.getvalue()
        assert "train class: espnet3.components.data.dataset.CombinedDataset" in out
        assert "valid class: espnet3.components.data.dataset.CombinedDataset" in out
        assert "combined[0]:" in out
        assert "combined[1]:" in out
        assert "transform: test.espnet3.utils.test_logging.custom_transform" in out
        assert "transform: test.espnet3.utils.test_logging.other_transform" in out
        assert "preprocessor: test.espnet3.utils.test_logging.custom_preprocessor" in out
        # None preprocessor becomes do_nothing_transform
        assert "preprocessor: espnet3.components.data.dataset.do_nothing_transform" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_dataloader_formats_human_readable():
    logger, stream, cleanup = _capture_logger("espnet3.test.dataloader")
    old_handlers, old_level, old_propagate, handler = cleanup

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2, num_workers=0)
    try:
        elog.log_dataloader(logger, loader, label="train")
        out = stream.getvalue()
        assert "DataLoader[train]:" in out
        assert "[DataLoader]" in out
        assert "batch_size" in out
        assert "num_workers" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)


def test_log_dataloader_iter_factory_includes_batch_sampler_repr():
    logger, stream, cleanup = _capture_logger("espnet3.test.dataloader.iter_factory")
    old_handlers, old_level, old_propagate, handler = cleanup

    from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
    from espnet2.samplers.build_batch_sampler import build_batch_sampler

    batches = build_batch_sampler(
        shape_files=["test_utils/espnet3/stats/stats_dummy"],
        type="unsorted",
        batch_size=2,
        batch_bins=4000000,
    )
    iter_factory = SequenceIterFactory(DummyDataset(), batches=batches, shuffle=False)
    iterator = iter_factory.build_iter(0, shuffle=False)

    try:
        elog.log_dataloader(
            logger,
            iterator,
            label="train",
            iter_factory=iter_factory,
            batches=batches,
        )
        out = stream.getvalue()
        assert "IterFactory[train]:" in out
        assert "IterBatches[train] class:" in out
        assert "UnsortedBatchSampler(" in out
    finally:
        handler.close()
        _reset_logger(logger, old_handlers, old_level, old_propagate)
