from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import config
from config import (
    CEILING_SEED,
    CEILING_STAGE,
    CONDITIONING_MODES,
    CONDITIONING_NONE,
    CROSSVAL_STAGE,
    MULTISEED_STAGE,
    SEEDS,
    STAGE_NAMES,
    STAGES,
    RuntimeConfig,
    stage_spec,
)
from run_pipeline import assert_command_matches_stage
from training import checkpoint_configuration_id


def _config(root: Path, stage: str, **overrides) -> RuntimeConfig:
    return RuntimeConfig(root, root, root, root, stage=stage, **overrides)


class StageTableTests(unittest.TestCase):
    def test_three_stages_are_fully_isolated(self) -> None:
        self.assertEqual(set(STAGE_NAMES), {CEILING_STAGE, MULTISEED_STAGE, CROSSVAL_STAGE})
        groups = {spec.experiment_group for spec in STAGES.values()}
        reports = {spec.report_dirname for spec in STAGES.values()}
        self.assertEqual(len(groups), len(STAGES))
        self.assertEqual(len(reports), len(STAGES))

    def test_stage_seed_pools(self) -> None:
        self.assertEqual(stage_spec(CEILING_STAGE).seeds, (CEILING_SEED,))
        self.assertEqual(stage_spec(MULTISEED_STAGE).seeds, SEEDS)
        self.assertEqual(stage_spec(CROSSVAL_STAGE).seeds, ())
        self.assertTrue(stage_spec(CEILING_STAGE).is_seed_driven)
        self.assertFalse(stage_spec(CROSSVAL_STAGE).is_seed_driven)

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            stage_spec("ceiling_test")

    def test_runtime_config_exposes_the_stage_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, spec in STAGES.items():
                built = _config(root, name)
                self.assertEqual(built.experiment_group, spec.experiment_group)
                self.assertEqual(built.to_dict()["experiment_group"], spec.experiment_group)
                self.assertEqual(built.to_dict()["stage"], name)

    def test_constant_learning_rate(self) -> None:
        # A decaying schedule would make a long ceiling run measure the schedule
        # expiring rather than the model saturating.
        self.assertEqual(config.SCHEDULER_GAMMA, 1.0)


class CheckpointNamespaceTests(unittest.TestCase):
    def test_stages_do_not_share_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for conditioning in CONDITIONING_MODES:
                identifiers = {
                    name: checkpoint_configuration_id(_config(root, name), conditioning)
                    for name in STAGE_NAMES
                }
                self.assertEqual(len(set(identifiers.values())), len(STAGE_NAMES))

    def test_epoch_count_does_not_change_the_namespace(self) -> None:
        # This is the property that lets a finished run be resumed for more epochs
        # into the same checkpoint directory.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ten = checkpoint_configuration_id(
                _config(root, CEILING_STAGE, epochs=10), CONDITIONING_NONE
            )
            fifteen = checkpoint_configuration_id(
                _config(root, CEILING_STAGE, epochs=15), CONDITIONING_NONE
            )
            self.assertEqual(ten, fifteen)

    def test_recipe_changes_still_change_the_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = checkpoint_configuration_id(
                _config(root, CEILING_STAGE), CONDITIONING_NONE
            )
            for overrides in (
                {"learning_rate": 5e-5},
                {"batch_size": 36},
                {"seed": 7},
            ):
                changed = checkpoint_configuration_id(
                    _config(root, CEILING_STAGE, **overrides), CONDITIONING_NONE
                )
                self.assertNotEqual(baseline, changed, overrides)


class CommandStageGuardTests(unittest.TestCase):
    """A command must not write into a stage that does not run it."""

    def test_seed_commands_require_a_seed_driven_stage(self) -> None:
        for command in ("seed", "multiseed", "multiseed-aggregate"):
            assert_command_matches_stage(command, MULTISEED_STAGE)
            with self.assertRaises(ValueError):
                assert_command_matches_stage(command, CROSSVAL_STAGE)

    def test_fold_commands_require_the_crossval_stage(self) -> None:
        for command in ("fold", "aggregate", "full", "claims", "summary", "all"):
            assert_command_matches_stage(command, CROSSVAL_STAGE)
            for stage in (CEILING_STAGE, MULTISEED_STAGE):
                with self.assertRaises(ValueError):
                    assert_command_matches_stage(command, stage)

    def test_ceiling_report_requires_a_single_seed_stage(self) -> None:
        assert_command_matches_stage("ceiling-report", CEILING_STAGE)
        with self.assertRaises(ValueError):
            assert_command_matches_stage("ceiling-report", MULTISEED_STAGE)


if __name__ == "__main__":
    unittest.main()
