from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import (
    CONDITIONED_VISION_DIM,
    CONDITIONING_MODES,
    CONDITIONING_NONE,
    CONDITIONING_PREFIX,
    CONDITIONING_TEXT,
    CONTEXT_WINDOW_SIZE,
    FULL_VALIDATION_SUBJECTS,
    PATIENT_FEATURE_DIM,
    PATIENT_PREFIX_DIM,
    PREDICTION_PROMPT,
    QWEN_HIDDEN_SIZE,
    RuntimeConfig,
    SEED,
    TARGET_FRAME_OFFSET,
    TARGET_WINDOW_SIZE,
    TASK_DESCRIPTIONS,
    UEFMA_LOSS_WEIGHT,
    VISION_FEATURE_DIM,
)
from data import (
    PatientNormalizer,
    StrokeRehabDataset,
    build_fold_assignments,
    canonical_subject_id,
    create_fold_assignments,
    fold_subjects,
)
from model import QwenMovementModel, compute_joint_loss, model_batch
from training import checkpoint_configuration_id, resolve_training_stop_status


class DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, QWEN_HIDDEN_SIZE)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def forward(self, inputs_embeds: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class CoreTests(unittest.TestCase):
    def test_canonical_subject(self) -> None:
        self.assertEqual(canonical_subject_id("S3"), "s03")
        self.assertEqual(canonical_subject_id("s32"), "s32")
        with self.assertRaises(ValueError):
            canonical_subject_id("patient3")
        self.assertEqual(FULL_VALIDATION_SUBJECTS, {"s03", "s19", "s27", "s32"})

    def test_fold_reproducibility_and_coverage(self) -> None:
        subjects = [f"s{index:02d}" for index in range(1, 33)]
        for n_folds, expected_size in ((4, 8), (8, 4)):
            first = build_fold_assignments(subjects, SEED, n_folds)
            second = build_fold_assignments(reversed(subjects), SEED, n_folds)
            self.assertTrue(first.equals(second))
            self.assertEqual(len(first), 32)
            self.assertTrue(first["subject_id"].is_unique)
            self.assertTrue(first.groupby("fold_id").size().eq(expected_size).all())
            for fold_id in range(n_folds):
                train, holdout = fold_subjects(first, fold_id)
                self.assertEqual((len(train), len(holdout)), (32 - expected_size, expected_size))
                self.assertFalse(set(train) & set(holdout))

    def test_tunable_fold_configuration_replaces_stale_assignments(self) -> None:
        subjects = tuple(f"s{index:02d}" for index in range(1, 33))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_config = RuntimeConfig(root, root, root, root, seed=1, n_folds=4)
            requested_config = RuntimeConfig(root, root, root, root, seed=SEED, n_folds=8)
            with patch("data.npz_subjects", return_value=subjects):
                create_fold_assignments(first_config)
                requested = create_fold_assignments(requested_config)

            payload = json.loads((root / "fold_assignments.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], SEED)
            self.assertEqual(payload["n_folds"], 8)
            persisted = pd.read_csv(root / "fold_assignments.csv")
            pd.testing.assert_frame_equal(persisted, requested, check_dtype=False)

    def test_safe_patient_encoding(self) -> None:
        normalizer = PatientNormalizer(50.0, 10.0, 20.0, 5.0, ("s01", "s02"))
        encoded = normalizer.encode("left", "Female", 60, "White/Caucasian", 15)
        self.assertEqual(encoded.shape, (PATIENT_FEATURE_DIM,))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertEqual(float(encoded[:2].sum()), 1.0)
        self.assertEqual(float(encoded[2:4].sum()), 1.0)
        self.assertEqual(float(encoded[4:10].sum()), 1.0)
        self.assertAlmostEqual(float(encoded[-2]), 1.0)
        self.assertAlmostEqual(float(encoded[-1]), -1.0)

    def test_patient_normalizer_uses_only_training_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path = root / "metadata.npz"
            np.savez(
                npz_path,
                subject_id=np.asarray(["s01", "s02", "s03"]),
                frame_index=np.asarray([0, 0, 0]),
                age=np.asarray([20.0, 40.0, 100.0]),
                time_since_stroke=np.asarray([2.0, 4.0, 10.0]),
            )
            normalizer = PatientNormalizer.fit(
                npz_path,
                ("s01", "s02"),
                root / "cache",
            )
            self.assertEqual(normalizer.fitted_subjects, ("s01", "s02"))
            self.assertAlmostEqual(normalizer.age_mean, 30.0)
            self.assertAlmostEqual(normalizer.time_mean, 3.0)
            self.assertAlmostEqual(normalizer.age_std, 10.0)
            self.assertAlmostEqual(normalizer.time_std, 1.0)

    def test_24_frame_context_alignment_and_zero_target_offset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row_count = 130
            npz_path = root / "windows.npz"
            np.savez(
                npz_path,
                vision_feat=np.zeros((row_count, VISION_FEATURE_DIM), dtype=np.float32),
                gesture_seq=np.zeros((row_count, TARGET_WINDOW_SIZE), dtype=np.int64),
                subject_id=np.asarray(["s01"] * row_count),
                UE_FMA_score=np.asarray([40.0] * row_count, dtype=np.float32),
                task_name=np.asarray(["tabletop"] * row_count),
                clip_id=np.asarray(["clip"] * row_count),
                frame_index=np.arange(row_count, dtype=np.int64),
            )
            dataset = StrokeRehabDataset(npz_path, root / "cache")
            context, target = dataset._window_indices(0)
            self.assertEqual(len(context), CONTEXT_WINDOW_SIZE)
            self.assertTrue(np.all(np.diff(context) == 5))
            self.assertEqual(target, int(context[-1]) + TARGET_FRAME_OFFSET)
            self.assertEqual(target, 115)
            self.assertEqual(dataset[0]["vision_feat"].shape, (24, VISION_FEATURE_DIM))

    def test_joint_loss_is_masked_and_normalized(self) -> None:
        logits = torch.zeros(2, 5, 5, requires_grad=True)
        targets = torch.tensor([[0, 1, 2, 3, 4], [0, 5, 2, 3, 4]])
        predicted = torch.tensor([33.0, 66.0], requires_grad=True)
        actual = torch.tensor([33.0, 0.0])
        losses = compute_joint_loss(logits, predicted, targets, actual)
        expected_mse = ((0.0**2) + (1.0**2)) / 2
        self.assertAlmostEqual(losses.primitive.item(), math.log(5), places=5)
        self.assertAlmostEqual(losses.ue_fma_normalized.item(), expected_mse, places=5)
        self.assertEqual(UEFMA_LOSS_WEIGHT, 0.1)
        self.assertAlmostEqual(
            losses.total.item(), math.log(5) + UEFMA_LOSS_WEIGHT * expected_mse, places=5
        )

    def test_base_and_prefix_shapes_and_parallel_gradients(self) -> None:
        batch_size = 2
        input_ids = torch.ones(batch_size, 3, dtype=torch.long)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        vision = torch.randn(batch_size, CONTEXT_WINDOW_SIZE, VISION_FEATURE_DIM)
        for conditioned in (False, True):
            model = QwenMovementModel(DummyBackbone(), QWEN_HIDDEN_SIZE, conditioned)
            patient = torch.randn(batch_size, PATIENT_FEATURE_DIM) if conditioned else None
            logits, fma = model(input_ids, attention, vision, patient)
            self.assertEqual(logits.shape, (batch_size, TARGET_WINDOW_SIZE, 5))
            self.assertEqual(fma.shape, (batch_size,))
            self.assertEqual(model.last_tensor_shapes["fused_visual_tokens"][-1], 1024)
            self.assertEqual(
                model.last_tensor_shapes["query_tokens"],
                (batch_size, TARGET_WINDOW_SIZE + 1, QWEN_HIDDEN_SIZE),
            )
            if conditioned:
                self.assertEqual(model.patient_encoder[1].out_features, PATIENT_PREFIX_DIM)
                self.assertEqual(model.feature_adapter[1].out_features, CONDITIONED_VISION_DIM)
                self.assertEqual(
                    model.last_tensor_shapes["patient_prefix"],
                    (batch_size, CONTEXT_WINDOW_SIZE, PATIENT_PREFIX_DIM),
                )
            else:
                self.assertIsNone(model.last_tensor_shapes["patient_prefix"])
            model.zero_grad(set_to_none=True)
            fma.sum().backward()
            self.assertFalse(any(
                parameter.grad is not None for parameter in model.primitive_head.parameters()
            ))
            self.assertTrue(any(
                parameter.grad is not None for parameter in model.regression_head.parameters()
            ))
            self.assertIsNotNone(model.fma_query.grad)

    def test_model_input_whitelist(self) -> None:
        batch = {
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "text_attention_mask": torch.ones(1, 1, dtype=torch.bool),
            "vision_feat": torch.zeros(1, CONTEXT_WINDOW_SIZE, 432),
            "patient_features": torch.zeros(1, 12),
            "ue_fma_score": torch.tensor([66.0]),
            "subject_id": ["s01"],
        }
        selected = model_batch(batch)
        self.assertEqual(
            set(selected),
            {"input_ids", "text_attention_mask", "vision_feat", "patient_features"},
        )
        self.assertNotIn("ue_fma_score", selected)
        self.assertNotIn("subject_id", selected)

    def test_training_stop_status(self) -> None:
        self.assertEqual(resolve_training_stop_status(5, 5, False), (False, "max_epochs"))
        self.assertEqual(resolve_training_stop_status(3, 5, True), (True, "early_stopping"))
        self.assertEqual(resolve_training_stop_status(3, 5, False), (False, "incomplete"))

    def test_checkpoint_namespace_captures_variant_and_fold_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = RuntimeConfig(root, root, root, root, n_folds=4)
            identifiers = {
                mode: checkpoint_configuration_id(config, mode)
                for mode in CONDITIONING_MODES
            }
            eight_fold = checkpoint_configuration_id(
                RuntimeConfig(root, root, root, root, n_folds=8), CONDITIONING_NONE
            )
            # "none" and "text" build the identical architecture, so only the hashed
            # conditioning string keeps their checkpoint directories apart.
            self.assertEqual(len(set(identifiers.values())), len(CONDITIONING_MODES))
            for mode, identifier in identifiers.items():
                self.assertIn(f"qwen35-0p8b-parallel-{mode}-", identifier)
            self.assertNotEqual(identifiers[CONDITIONING_NONE], eight_fold)
            with self.assertRaises(ValueError):
                checkpoint_configuration_id(config, "not_a_mode")

    def test_patient_text_prompt_matches_source_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row_count = 130
            npz_path = root / "windows.npz"
            np.savez(
                npz_path,
                vision_feat=np.zeros((row_count, VISION_FEATURE_DIM), dtype=np.float32),
                gesture_seq=np.zeros((row_count, TARGET_WINDOW_SIZE), dtype=np.int64),
                subject_id=np.asarray(["s01"] * row_count),
                UE_FMA_score=np.asarray([40.0] * row_count, dtype=np.float32),
                task_name=np.asarray(["tabletop"] * row_count),
                clip_id=np.asarray(["clip"] * row_count),
                frame_index=np.arange(row_count, dtype=np.int64),
                paretic_side=np.asarray(["right"] * row_count),
                gender=np.asarray(["Female"] * row_count),
                age=np.asarray([67.4] * row_count, dtype=np.float64),
                race=np.asarray(["Black/African-American"] * row_count),
                time_since_stroke=np.asarray([123.45] * row_count, dtype=np.float64),
            )
            description = TASK_DESCRIPTIONS["tabletop"]
            expected_patient = (
                "Patient information: paretic side is right; gender is Female; "
                "age is 67 years; race is Black/African-American; "
                "time since stroke is 123.5."
            )

            text_dataset = StrokeRehabDataset(
                npz_path, root / "cache", conditioning=CONDITIONING_TEXT
            )
            sample = text_dataset[0]
            self.assertEqual(
                sample["text"],
                f"{description}\n{expected_patient}\n{PREDICTION_PROMPT}",
            )
            # The text variant conditions through the prompt only.
            self.assertNotIn("patient_features", sample)

            base_dataset = StrokeRehabDataset(
                npz_path, root / "cache", conditioning=CONDITIONING_NONE
            )
            self.assertEqual(
                base_dataset[0]["text"], f"{description}\n{PREDICTION_PROMPT}"
            )
            self.assertNotIn("patient_features", base_dataset[0])

    def test_conditioning_rejects_mismatched_normalizer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            npz_path = root / "windows.npz"
            np.savez(npz_path, subject_id=np.asarray(["s01"]))
            normalizer = PatientNormalizer(50.0, 10.0, 20.0, 5.0, ("s01",))
            with self.assertRaises(ValueError):
                StrokeRehabDataset(
                    npz_path,
                    root / "cache",
                    conditioning=CONDITIONING_TEXT,
                    patient_normalizer=normalizer,
                )
            with self.assertRaises(ValueError):
                StrokeRehabDataset(
                    npz_path, root / "cache", conditioning=CONDITIONING_PREFIX
                )
            with self.assertRaises(ValueError):
                StrokeRehabDataset(npz_path, root / "cache", conditioning="binary")


if __name__ == "__main__":
    unittest.main()
