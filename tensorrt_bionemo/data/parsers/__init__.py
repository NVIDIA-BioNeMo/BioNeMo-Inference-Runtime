from typing import Optional

from .a3m import A3MParsed, parse_a3m_content, read_a3m
from .fasta import SequenceParsed, parse_fasta_content, read_fasta
from .mmcif import MmcifParsed, parse_mmcif_content, read_mmcif


class InputParsed(dict):

    def __init__(self,
                 primary: SequenceParsed,
                 msa: dict[str, list[A3MParsed]],
                 template: Optional[dict[str, list[MmcifParsed]]] = None):
        super().__init__(primary=primary, msa=msa, template={})


__all__ = [
    "parse_fasta_content",
    "read_fasta",
    "SequenceParsed",
    "parse_mmcif_content",
    "read_mmcif",
    "MmcifParsed",
    "parse_a3m_content",
    "read_a3m",
    "A3MParsed",
    "InputParsed",
]
