# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_bionemo.pipeline.base import ContextGeneratorBase, TransformBase
from tensorrt_bionemo.pipeline.stages.tokenizer_stage import TokenizerUDF


class MockContextGenerator(ContextGeneratorBase):

    def __init__(self, output_tensors: dict = None, required_kwargs: list = None):
        super().__init__()
        self._output = output_tensors or {"mock_feature": torch.ones(10)}
        self._required_kwargs = required_kwargs or []

    def __call__(self, **kwargs) -> dict[str, torch.Tensor]:
        return self._output


class MockTransform(TransformBase):

    def __init__(self, enabled: bool = True):
        super().__init__(config=None)
        self._enabled = enabled

    def __call__(self, batch: dict, context: dict = None) -> dict:
        batch["transformed"] = True
        return batch

    def is_enabled(self) -> bool:
        return self._enabled


class TestTokenizerUDFBasicProcessing:

    @pytest.fixture
    def tokenizer_udf(self):
        return TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            transform_funcs=[],
        )

    def test_init_with_defaults(self):
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={},
        )
        assert udf.context_generators == {}
        assert udf.transform_funcs == []

    def test_init_with_context_generators(self):
        generators = {"primary": MockContextGenerator()}
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators=generators,
        )
        assert "primary" in udf.context_generators

    def test_init_with_transform_funcs(self):
        transforms = [MockTransform()]
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={},
            transform_funcs=transforms,
        )
        assert len(udf.transform_funcs) == 1


class TestTokenizerUDFContextGeneration:

    def test_single_context_generator(self):
        output_tensors = {"feature_a": torch.randn(5, 10)}
        generator = MockContextGenerator(output_tensors=output_tensors)

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": generator},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"parsed": {"primary": {}}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feature_a" in result
        assert result["feature_a"].shape == (5, 10)

    def test_multiple_context_generators(self):
        gen1 = MockContextGenerator({"feat1": torch.ones(3)})
        gen2 = MockContextGenerator({"feat2": torch.zeros(5)})

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"gen1": gen1, "gen2": gen2},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feat1" in result
        assert "feat2" in result

    def test_context_generator_with_required_kwargs(self):
        class RequiredKwargsGenerator(ContextGeneratorBase):
            def __init__(self):
                super().__init__()
                self._required_kwargs = ["parsed"]

            def __call__(self, parsed=None) -> dict:
                return {"from_parsed": torch.tensor([len(str(parsed))])}

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": RequiredKwargsGenerator()},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        row = {"parsed": {"data": "test"}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "from_parsed" in result


class TestTokenizerUDFTransforms:

    def test_transform_applied_when_enabled(self):
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": MockContextGenerator({})},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            transform_funcs=[MockTransform(enabled=True)],
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert result.get("transformed") is True

    def test_transform_skipped_when_disabled(self):
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": MockContextGenerator({})},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            transform_funcs=[MockTransform(enabled=False)],
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "transformed" not in result

    def test_multiple_transforms_chained(self):
        class TransformA(TransformBase):
            def __init__(self):
                super().__init__(config=None)

            def __call__(self, batch, context=None):
                batch["step_a"] = True
                return batch

            def is_enabled(self):
                return True

        class TransformB(TransformBase):
            def __init__(self):
                super().__init__(config=None)

            def __call__(self, batch, context=None):
                batch["step_b"] = batch.get("step_a", False)
                return batch

            def is_enabled(self):
                return True

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": MockContextGenerator({})},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
            transform_funcs=[TransformA(), TransformB()],
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert result["step_a"] is True
        assert result["step_b"] is True


class TestTokenizerUDFContextMerger:

    def test_custom_context_merger(self):
        def custom_merger(context_dict):
            merged = {}
            for name, tensors in context_dict.items():
                for k, v in tensors.items():
                    merged[f"{name}_{k}"] = v
            return merged

        gen1 = MockContextGenerator({"feat": torch.ones(2)})
        gen2 = MockContextGenerator({"feat": torch.zeros(3)})

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"gen1": gen1, "gen2": gen2},
            context_merger_func=custom_merger,
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "gen1_feat" in result
        assert "gen2_feat" in result

    def test_default_dict_merger(self):
        from tensorrt_bionemo.pipeline.base import dict_context_merger

        gen = MockContextGenerator({"feature": torch.ones(5)})

        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": gen},
            context_merger_func=dict_context_merger,
        )

        row = {"parsed": {}, "__record_id": "test"}
        result = asyncio.run(udf.udf_for_item(row))

        assert "feature" in result


class TestTokenizerUDFBatchProcessing:

    def test_batch_processing_multiple_rows(self):
        udf = TokenizerUDF(
            compute_by_rows=True,
            drop_keys=None,
            expected_input_keys=["parsed"],
            update_row=True,
            context_generators={"primary": MockContextGenerator({"feat": torch.ones(3)})},
            context_merger_func=lambda x: {k: v for d in x.values() for k, v in d.items()},
        )

        async def run_batch():
            batch = {
                "parsed": [{}, {}, {}],
                "__record_id": ["r1", "r2", "r3"],
            }
            results = []
            async for output in udf(batch):
                results.append(output)
            return results

        results = asyncio.run(run_batch())

        assert len(results) == 1
        assert len(results[0]["feat"]) == 3

