import re
from io import StringIO
from pathlib import Path
from typing import TextIO, Union

from Bio import SeqIO

from tensorrt_bionemo.data.schemas import EntityType, InputChain, Sequence

_alphabetical_order = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SequenceParsed(dict):

    def __init__(self, chains: list[InputChain]):
        super().__init__(chains=chains)


def parse_fasta_content(
    content: StringIO | TextIO,
    return_as_list: bool = False,
    is_description_formatted: bool = False
) -> Union[SequenceParsed, tuple[list[str], list[str]]]:
    """
    Read a fasta file from a string or text stream and return a list of sequences.
    If it is input chain, return SequenceParsed. Otherwise, if msa, return list of strings for better performance.
    """
    if isinstance(content, str):
        content = StringIO(content)
    fasta_sequences = SeqIO.parse(content, "fasta")
    # Regex to match format: CHAIN_ID|ENTITY_TYPE|MSA_ID
    pattern = r'^([^|]+)\|([^|]+)\|([^|]+)$'
    chains: list[InputChain] = []
    sequences: list[str] = []

    if return_as_list:
        seqs = []
        desps = []
        for fasta in fasta_sequences:
            seqs.append(str(fasta.seq))
            desps.append(fasta.description)
        return seqs, desps

    for i, fasta in enumerate(fasta_sequences):
        desp = fasta.description
        seq = fasta.seq

        match = None
        if is_description_formatted:
            match = re.match(pattern, desp)

        if match is not None:
            chain_id, entity_type, msa_id = match.groups()
            print(
                f"Parsed format - Chain: {chain_id}, Entity: {entity_type}, MSA: {msa_id}"
            )
        else:
            # print(f"Simple description format: {desp}")
            entity_type = EntityType.PROTEIN
            id_letter = _alphabetical_order[i % len(_alphabetical_order)]
            id_number = i // len(_alphabetical_order)
            if id_number == 0:
                id_ = id_letter
            else:
                id_ = f"{id_letter}{id_number}"

            input_chain = InputChain(chain_id=id_,
                                     entity_type=entity_type,
                                     sequence=Sequence(residues=str(seq),
                                                       description=desp))
            chains.append(input_chain)

    return SequenceParsed(chains=chains)


def read_fasta(file_path: str | Path,
               is_description_formatted: bool = False) -> SequenceParsed:
    with open(file_path) as source:
        return parse_fasta_content(
            source, is_description_formatted=is_description_formatted)
