import logging
import re

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet3.components.callbacks.default_callbacks import TrainBatchMetricsLogger
from espnet3.components.modeling.lightning_module import ESPnetLightningModule
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return {"x": torch.tensor([idx], dtype=torch.float32)}

    def __len__(self):
        return 4


class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 4, output_dim: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Minimal loss + stats expected by ESPnetLightningModule._step
        y = self.linear(x)
        loss = y.mean()
        stats = {"loss": loss, "acc": torch.tensor(0.5)}
        weight = torch.tensor(float(x.shape[0]))
        return loss, stats, weight


def _build_from_yaml(path):
    cfg = OmegaConf.load(path)
    model = instantiate(cfg.model)
    model_cfg = OmegaConf.create(
        {
            "dataset": cfg.dataset,
            "dataloader": cfg.dataloader,
            "optimizer": cfg.optimizer,
            "scheduler": cfg.scheduler,
            "num_device": 1,
        }
    )
    trainer_cfg = OmegaConf.create(cfg.trainer)
    return model, model_cfg, trainer_cfg


def test_logging_from_yaml_module_trainer(tmp_path, caplog):
    yaml_path = "test_utils/espnet3/config/logging_sample.yaml"
    model, model_cfg, trainer_cfg = _build_from_yaml(yaml_path)
    model_cfg.dataloader.train.iter_factory = None
    model_cfg.dataloader.train.batch_size = 2
    model_cfg.dataloader.valid.iter_factory = None
    model_cfg.dataloader.valid.batch_size = 2
    with caplog.at_level(logging.INFO):
        lit = ESPnetLightningModule(model, model_cfg)
        wrapper = ESPnet3LightningTrainer(
            model=lit, config=trainer_cfg, exp_dir=str(tmp_path)
        )

        callback = next(
            cb
            for cb in wrapper.trainer.callbacks
            if isinstance(cb, TrainBatchMetricsLogger)
        )

        optim_bundle = lit.configure_optimizers()
        dummy_trainer = type(
            "DummyTrainer",
            (),
            {
                "current_epoch": 0,
                "callback_metrics": {
                    "train/loss": torch.tensor(1.5),
                    "train/acc": torch.tensor(0.5),
                },
                "optimizers": [optim_bundle["optimizer"]],
            },
        )()
        dummy_trainer.callback_metrics = {
            "train/loss": torch.tensor(1.5),
            "train/acc": torch.tensor(0.5),
        }

        lit.train_dataloader()
        lit.val_dataloader()

        batch = ("utt", {"x": torch.zeros(2, 4)})
        callback.on_train_batch_start(dummy_trainer, lit, batch, batch_idx=0)
        callback.on_before_backward(dummy_trainer, lit, torch.tensor(1.0))
        callback.on_after_backward(dummy_trainer, lit)
        callback.on_before_optimizer_step(
            dummy_trainer, lit, dummy_trainer.optimizers[0]
        )
        callback.on_after_optimizer_step(dummy_trainer, lit)
        callback.on_train_batch_end(
            dummy_trainer, lit, outputs=None, batch=batch, batch_idx=0
        )

    text = caplog.text
    assert "Data organizer:" in text
    assert "train class: espnet3.components.data.dataset.CombinedDataset" in text
    assert "valid class: espnet3.components.data.dataset.CombinedDataset" in text
    assert "DataLoader[train] class:" in text
    assert "DataLoader[valid] class:" in text
    assert "Model summary:" in text
    assert "Optimizer[0]:" in text
    assert "Scheduler[0]:" in text
    assert "1epoch:train:1-1batch" in text
    assert re.search(r"acc=0\.5\b", text)
    assert re.search(r"loss=1\.5\b", text)
    assert re.search(r"optim0_lr0=0\.01\b", text)
