import logging

import pytest
from omegaconf import OmegaConf

from espnet3.utils.run_utils import (
    apply_training_experiment_context,
    resolve_loaded_configs,
    validate_experiment_context,
)


def test_apply_training_experiment_context_inserts_missing_values(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create({"exp_tag": None})

    with caplog.at_level(logging.INFO):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Inserted inference_config.exp_tag from training_config" in caplog.text
    assert "Inserted inference_config.exp_dir from training_config" in caplog.text


def test_apply_training_experiment_context_warns_on_overwrite(caplog) -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/train_debug",
        }
    )
    inference = OmegaConf.create(
        {
            "exp_tag": "other_tag",
            "exp_dir": "./exp/other_tag",
        }
    )

    with caplog.at_level(logging.WARNING):
        apply_training_experiment_context(
            training_config=training,
            inference_config=inference,
            metrics_config=None,
            log=logging.getLogger("test.run_utils"),
        )

    assert inference.exp_tag == "train_debug"
    assert inference.exp_dir == "./exp/train_debug"
    assert "Overriding inference_config.exp_tag" in caplog.text
    assert "Overriding inference_config.exp_dir" in caplog.text


def test_apply_training_experiment_context_noop_without_training() -> None:
    inference = OmegaConf.create(
        {
            "exp_tag": "standalone_eval",
            "exp_dir": "./exp/standalone_eval",
        }
    )

    apply_training_experiment_context(
        training_config=None,
        inference_config=inference,
        metrics_config=None,
        log=logging.getLogger("test.run_utils"),
    )

    assert inference.exp_tag == "standalone_eval"
    assert inference.exp_dir == "./exp/standalone_eval"


def test_validate_experiment_context_accepts_standalone_inference() -> None:
    validate_experiment_context(
        training_config=None,
        inference_config=OmegaConf.create(
            {
                "exp_tag": "standalone_eval",
                "exp_dir": "./exp/standalone_eval",
            }
        ),
        metrics_config=None,
        stages_to_run=["infer"],
    )


def test_validate_experiment_context_requires_identity() -> None:
    with pytest.raises(ValueError, match="infer stage requires --training_config"):
        validate_experiment_context(
            training_config=None,
            inference_config=OmegaConf.create({"exp_tag": None}),
            metrics_config=None,
            stages_to_run=["infer"],
        )


def test_validate_experiment_context_accepts_training_backed_inference() -> None:
    validate_experiment_context(
        training_config=OmegaConf.create(
            {
                "exp_tag": "train_asr_rnn",
                "exp_dir": "./exp/train_asr_rnn",
            }
        ),
        inference_config=OmegaConf.create({"inference_dir": "${exp_dir}/inference"}),
        metrics_config=None,
        stages_to_run=["infer"],
    )


def test_resolve_loaded_configs_resolves_interpolations() -> None:
    training = OmegaConf.create(
        {
            "exp_tag": "train_debug",
            "exp_dir": "./exp/${exp_tag}",
        }
    )
    inference = OmegaConf.create({"inference_dir": "${exp_dir}/inference"})

    apply_training_experiment_context(
        training_config=training,
        inference_config=inference,
        metrics_config=None,
        log=logging.getLogger("test.run_utils"),
    )
    resolve_loaded_configs(training, inference)

    assert training.exp_dir == "./exp/train_debug"
    assert inference.inference_dir == "./exp/train_debug/inference"
