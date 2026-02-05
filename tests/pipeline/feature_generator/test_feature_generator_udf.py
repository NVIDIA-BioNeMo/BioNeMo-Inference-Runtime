# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys

import numpy as np
import pytest
import torch

from tensorrt_bionemo.pipeline.base import FeatureCollatorBase, FeatureGeneratorBase
from tensorrt_bionemo.pipeline.stages.feature_generator_stage import FeatureGeneratorUDF


class MockFeatureGenerator(FeatureGeneratorBase):

    def __init__(self, output_tensors: dict = None, enabled: bool = True, name: str = "mock"):
        super().__init__(config=None, name=name)
        self._output = output_tensors or {"generated_feature": torch.ones(10)}
        self._enabled = enabled

    def __call__(self, batch: dict, context: dict) -> dict:
        return self._output

    def is_enabled(self) -> bool:
        return self._enabled


class MockFeatureCollator(FeatureCollatorBase):

    def __init__(self, enabled: bool = True, name: str = "collator"):
        super().__init__(config=None, name=name)
        self._enabled = enabled

    def __call__(self, batch: dict, context: dict) -> dict:
        batch["collated"] = True
        return batch

    def is_enabled(self) -> bool:
        return self._enabled


class TestFeatureGeneratorUDFBasicProcessing:

    @pytest.fixture
    def feature_udf(self):
        return FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

    def test_init_with_defaults(self):
        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[],
        )
        assert udf.feature_generators == []
        assert udf.feature_collators == []
        assert udf.pre_init is None

    def test_init_with_feature_generators(self):
        generators = [MockFeatureGenerator()]
        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=generators,
        )
        assert len(udf.feature_generators) == 1

    def test_init_with_collators(self):
        collators = [MockFeatureCollator()]
        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[],
            feature_collators=collators,
        )
        assert len(udf.feature_collators) == 1


class TestFeatureGeneratorUDFTensorExtraction:

    @pytest.fixture
    def udf(self):
        return FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[],
        )

    def test_extract_tensors_from_torch_tensor(self, udf):
        row = {"tensor_field": torch.randn(5, 10)}
        result = udf._extract_tensors(row)

        assert "tensor_field" in result
        assert isinstance(result["tensor_field"], torch.Tensor)

    def test_extract_tensors_from_numpy_array(self, udf):
        row = {"numpy_field": np.array([1.0, 2.0, 3.0])}
        result = udf._extract_tensors(row)

        assert "numpy_field" in result
        assert isinstance(result["numpy_field"], torch.Tensor)

    def test_extract_tensors_from_numpy_scalar(self, udf):
        row = {"scalar_field": np.float32(3.14)}
        result = udf._extract_tensors(row)

        assert "scalar_field" in result
        assert isinstance(result["scalar_field"], torch.Tensor)

    def test_extract_tensors_ignores_non_numeric(self, udf):
        row = {
            "tensor": torch.ones(3),
            "string": "hello",
            "list": [1, 2, 3],
            "dict": {"a": 1},
        }
        result = udf._extract_tensors(row)

        assert "tensor" in result
        assert "string" not in result
        assert "list" not in result
        assert "dict" not in result


class TestFeatureGeneratorUDFFeatureGeneration:

    def test_single_feature_generator(self):
        output = {"feature_x": torch.randn(5, 10)}
        generator = MockFeatureGenerator(output_tensors=output, name="gen1")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"input_tensor": torch.ones(3), "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feature_x" in result
        assert result["feature_x"].shape == (5, 10)

    def test_multiple_feature_generators(self):
        gen1 = MockFeatureGenerator({"feat1": torch.ones(3)}, name="gen1")
        gen2 = MockFeatureGenerator({"feat2": torch.zeros(5)}, name="gen2")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[gen1, gen2],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feat1" in result
        assert "feat2" in result

    def test_disabled_generator_skipped(self):
        enabled_gen = MockFeatureGenerator({"enabled_feat": torch.ones(3)}, enabled=True, name="enabled")
        disabled_gen = MockFeatureGenerator({"disabled_feat": torch.zeros(3)}, enabled=False, name="disabled")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[enabled_gen, disabled_gen],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "enabled_feat" in result
        assert "disabled_feat" not in result

    def test_duplicate_generator_name_raises_error(self):
        gen1 = MockFeatureGenerator({"feat": torch.ones(3)}, name="same_name")
        gen2 = MockFeatureGenerator({"feat": torch.zeros(3)}, name="same_name")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[gen1, gen2],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        with pytest.raises(ValueError, match="already in the features dictionary"):
            asyncio.run(udf.udf_for_item(row))


class TestFeatureGeneratorUDFCollators:

    def test_collator_applied_when_enabled(self):
        generator = MockFeatureGenerator({"feat": torch.ones(3)}, name="gen")
        collator = MockFeatureCollator(enabled=True, name="collator")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            feature_collators=[collator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert result.get("collated") is True

    def test_collator_skipped_when_disabled(self):
        generator = MockFeatureGenerator({"feat": torch.ones(3)}, name="gen")
        collator = MockFeatureCollator(enabled=False, name="collator")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            feature_collators=[collator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "collated" not in result

    def test_multiple_collators_chained(self):
        class CollatorA(FeatureCollatorBase):
            def __init__(self):
                super().__init__(config=None, name="collator_a")

            def __call__(self, batch, context):
                batch["step_a"] = True
                return batch

            def is_enabled(self):
                return True

        class CollatorB(FeatureCollatorBase):
            def __init__(self):
                super().__init__(config=None, name="collator_b")

            def __call__(self, batch, context):
                batch["step_b"] = batch.get("step_a", False)
                return batch

            def is_enabled(self):
                return True

        generator = MockFeatureGenerator({"feat": torch.ones(3)}, name="gen")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            feature_collators=[CollatorA(), CollatorB()],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert result["step_a"] is True
        assert result["step_b"] is True


class TestFeatureGeneratorUDFPreInit:

    def test_pre_init_called(self):
        def mock_pre_init(context):
            context["initialized"] = True
            return context

        generator = MockFeatureGenerator({"feat": torch.ones(3)}, name="gen")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            pre_init=mock_pre_init,
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feat" in result

    def test_pre_init_none_allowed(self):
        generator = MockFeatureGenerator({"feat": torch.ones(3)}, name="gen")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            pre_init=None,
        )

        row = {"__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feat" in result


class TestFeatureGeneratorUDFBatchProcessing:

    def test_batch_processing_multiple_rows(self):
        generator = MockFeatureGenerator({"feat": torch.ones(5)}, name="gen")

        udf = FeatureGeneratorUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=[],
            update_row=False,
            feature_generators=[generator],
            features_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        async def run_batch():
            batch = {
                "input_data": [np.ones(3), np.ones(3), np.ones(3)],
                "__record_id": ["r1", "r2", "r3"],
            }
            results = []
            async for output in udf(batch):
                results.append(output)
            return results

        results = asyncio.run(run_batch())

        assert len(results) == 1
        assert len(results[0]["feat"]) == 3
