import traceback
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type

from tensorrt_bionemo.pipeline.base import (ContextGeneratorBase,
                                            TransformBase, dict_context_merger)
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class TokenizerUDF(StatefulStageUDF):

    def __init__(self,
                 compute_by_rows: bool,
                 drop_keys: List[str],
                 expected_input_keys: List[str],
                 update_row: bool,
                 context_generators: dict[str, ContextGeneratorBase],
                 context_merger_func: Optional[Callable] = dict_context_merger,
                 transform_funcs: list[TransformBase] = []):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys,
                         update_row)

        self.context_generators = context_generators
        self.context_merger_func = context_merger_func
        self.transform_funcs = transform_funcs

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
            context_dict = {}
            try:
                for name, generator in self.context_generators.items():
                    required_kwargs = generator.get_required_kwargs()
                    if required_kwargs:
                        required_kwargs_dict = {
                            k: row[k]
                            for k in required_kwargs
                        }
                        context_dict[name] = generator(**required_kwargs_dict)
                    else:
                        context_dict[name] = generator()
                # Flatten the context dictionary
                context_dict = self.context_merger_func(context_dict)
                for transform_func in self.transform_funcs:
                    if transform_func.is_enabled():
                        context_dict = transform_func(context_dict)
                context_dict["__inference_error__"] = {
                    "error_msg": None,
                    "traceback": None
                }
            except Exception as e:
                context_dict = {}
                error_msg = f"{type(e).__name__}: {str(e)}"
                context_dict["__inference_error__"] = {
                    "error_msg": error_msg,
                    "traceback": traceback.format_exc()
                }
            finally:
                context_dict[self.IDX_IN_BATCH_COLUMN] = row[
                    self.IDX_IN_BATCH_COLUMN]
            yield context_dict


class TokenizerStage(StatefulStage):
    """
    A stage that tokenizes the input.
    """

    fn: Type[StatefulStageUDF] = TokenizerUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "parsed":
            "A parsed record of the input. "
            "See tensorrt_bionemo.data.parsers.InputParsed for details."
        }
