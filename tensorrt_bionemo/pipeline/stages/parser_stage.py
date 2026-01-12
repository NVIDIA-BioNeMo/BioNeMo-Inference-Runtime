import traceback
from typing import Any, AsyncIterator, Dict, List, Type

from tensorrt_bionemo.data.parsers import (InputParsed, parse_a3m_content,
                                           parse_fasta_content,
                                           parse_mmcif_content, read_a3m,
                                           read_fasta, read_mmcif)
from tensorrt_bionemo.data.schemas import InputRequest
from tensorrt_bionemo.pipeline.stages.base import (StatefulStage,
                                                   StatefulStageUDF)


class ParserUDF(StatefulStageUDF):

    def __init__(self, compute_by_rows: bool, drop_keys: List[str],
                 expected_input_keys: List[str], update_row: bool):
        super().__init__(compute_by_rows, drop_keys, expected_input_keys,
                         update_row)

    def _parse_content(self, input_: InputRequest) -> InputParsed:
        input_parsed = parse_fasta_content(
            input_.fasta_file,
            is_description_formatted=input_.is_description_formatted)
        a3m_parsed = {}
        mmcif_parsed = {}
        for chain_id, a3m_files in input_.a3m_files.items():
            a3m_parsed[chain_id] = [
                parse_a3m_content(a3m_file) for a3m_file in a3m_files
            ]
        for chain_id, mmcif_files in input_.mmcif_files.items():
            mmcif_parsed[chain_id] = [
                parse_mmcif_content(mmcif_file) for mmcif_file in mmcif_files
            ]
        return InputParsed(primary=input_parsed,
                           msa=a3m_parsed,
                           template=mmcif_parsed)

    def _parse_file(self, input_: InputRequest) -> InputParsed:
        input_parsed = read_fasta(
            input_['fasta_file'],
            is_description_formatted=input_['is_description_formatted'])
        a3m_parsed = {}
        mmcif_parsed = {}
        for chain_id, a3m_files in input_['a3m_files'].items():
            a3m_parsed[chain_id] = [
                read_a3m(a3m_file) for a3m_file in a3m_files
            ]
        for chain_id, mmcif_files in input_['mmcif_files'].items():
            mmcif_parsed[chain_id] = [
                read_mmcif(mmcif_file) for mmcif_file in mmcif_files
            ]
        return InputParsed(primary=input_parsed,
                           msa=a3m_parsed,
                           template=mmcif_parsed)

    async def udf_for_rows(
            self, batch: List[Dict[str,
                                   Any]]) -> AsyncIterator[Dict[str, Any]]:
        results = []
        for row in batch:
            record: InputRequest = InputRequest(**
                                                row["record"])  # type: ignore
            try:
                if record['is_files']:
                    input_parsed = self._parse_file(record)
                else:
                    input_parsed = self._parse_content(record)
                results.append({
                    "parsed":
                    input_parsed,
                    "__inference_error__": {
                        "error_msg": None,
                        "traceback": None
                    },
                    self.IDX_IN_BATCH_COLUMN:
                    row[self.IDX_IN_BATCH_COLUMN]
                })
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                results.append({
                    "parsed":
                    None,
                    "__inference_error__": {
                        "error_msg": error_msg,
                        "traceback": traceback.format_exc()
                    },
                    self.IDX_IN_BATCH_COLUMN:
                    row[self.IDX_IN_BATCH_COLUMN]
                })
        assert len(batch) == len(results)
        for row, result in zip(batch, results):
            yield result


class ParserStage(StatefulStage):
    """
    A stage that parses the input.
    """

    fn: Type[StatefulStageUDF] = ParserUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "record":
            "A record of the input. "
            "See tensorrt_bionemo.data.schemas.InputRequest for details."
        }
