import traceback
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

import numpy as np
import torch

from tensorrt_bionemo.pipeline.base import (FeatureCollatorBase,
                                            FeatureGeneratorBase,
                                            dict_context_merger)
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class FeatureGeneratorUDF(StatefulStageUDF):

    def __init__(
            self,
            compute_by_rows: bool,
            drop_keys: List[str],
            expected_input_keys: List[str],
            update_row: bool,
            feature_generators: list[FeatureGeneratorBase],
            features_merger_func: Optional[Callable] = dict_context_merger,
            feature_collators: Optional[list[FeatureCollatorBase]] = None,
            pre_init: Optional[Callable] = None):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys,
                         update_row)

        self.feature_generators = feature_generators
        self.features_merger_func = features_merger_func
        self.feature_collators = feature_collators or []
        self.pre_init = pre_init

    async def udf_for_rows(
            self, batch: List[Dict[str,
                                   Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Generate context tensors for each row in the batch. The final output should be a dictionary with the following keys:
            - __idx_in_batch: The index of the row in the batch.
            - __inference_error__: The error message if the generation failed.
            - parsed: The parsed object.
            - "context_0": torch.Tensor,
            - "context_1": torch.Tensor,
            - ...
            - "context_n": torch.Tensor,
        """
        for row in batch:
            context = {}
            features_dict = {}
            # Get only the tensors from the row.
            row_with_tensors = {}
            for k, v in row.items():
                if isinstance(v, np.ndarray):
                    row_with_tensors[k] = torch.from_numpy(v)
                elif isinstance(v, np.generic):
                    row_with_tensors[k] = torch.tensor(v)

            if self.pre_init is not None:
                context = self.pre_init(context=context)
            try:
                with torch.no_grad():
                    for generator in self.feature_generators:
                        if generator.is_enabled():
                            if generator.get_name() in features_dict:
                                raise ValueError(
                                    f"Feature generator {generator.get_name()} is already in the features dictionary."
                                )
                            features_dict[generator.get_name()] = generator(
                                row_with_tensors, context)
                    # Flatten the context dictionary
                    features_dict = self.features_merger_func(features_dict)

                    # Merge the features dictionary with the row_with_tensors dictionary
                    # This ensures that the original data is not lost.
                    row_with_tensors.update(features_dict)

                    for collator in self.feature_collators:
                        if collator.is_enabled():
                            row_with_tensors = collator(
                                row_with_tensors, context)

                    row_with_tensors["__inference_error__"] = {
                        "error_msg": None,
                        "traceback": None
                    }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                row_with_tensors["__inference_error__"] = {
                    "error_msg": error_msg,
                    "traceback": traceback.format_exc()
                }
            finally:
                row_with_tensors[self.IDX_IN_BATCH_COLUMN] = row[
                    self.IDX_IN_BATCH_COLUMN]
            yield row_with_tensors


class FeatureGeneratorStage(StatefulStage):
    """
    A stage that tokenizes the input.
    """

    fn: Type[StatefulStageUDF] = FeatureGeneratorUDF
    update_row: bool = False
