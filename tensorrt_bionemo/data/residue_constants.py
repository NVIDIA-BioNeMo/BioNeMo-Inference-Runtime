from enum import Enum
from functools import lru_cache

from pydantic import BaseModel


class Model(Enum):
    OpenFold2 = "openfold2"
    Boltz = "boltz"


class ResType(BaseModel):
    name: str
    canonical_name: str

    def __eq__(self, other):
        if not isinstance(other, ResType):
            return False
        return self.name == other.name


class ResTypes:
    A = ResType(name="A", canonical_name="ALA")
    R = ResType(name="R", canonical_name="ARG")
    N = ResType(name="N", canonical_name="ASN")
    D = ResType(name="D", canonical_name="ASP")
    C = ResType(name="C", canonical_name="CYS")
    Q = ResType(name="Q", canonical_name="GLN")
    E = ResType(name="E", canonical_name="GLU")
    G = ResType(name="G", canonical_name="GLY")
    H = ResType(name="H", canonical_name="HIS")
    I = ResType(name="I", canonical_name="ILE")
    L = ResType(name="L", canonical_name="LEU")
    K = ResType(name="K", canonical_name="LYS")
    M = ResType(name="M", canonical_name="MET")
    F = ResType(name="F", canonical_name="PHE")
    P = ResType(name="P", canonical_name="PRO")
    S = ResType(name="S", canonical_name="SER")  # codespell:ignore
    T = ResType(name="T", canonical_name="THR")
    W = ResType(name="W", canonical_name="TRP")
    Y = ResType(name="Y", canonical_name="TYR")
    V = ResType(name="V", canonical_name="VAL")
    X = ResType(name="X", canonical_name="UNK")
    RA = ResType(name="RA", canonical_name="RA")  # RNA: Adenine
    RC = ResType(name="RC", canonical_name="RC")  # RNA: Cytosine
    RG = ResType(name="RG", canonical_name="RG")  # RNA: Guanine
    RU = ResType(name="RU", canonical_name="RU")  # RNA: Uracil
    RX = ResType(name="RX", canonical_name="RX")  # RNA: Unknown
    DA = ResType(name="DA", canonical_name="DA")  # DNA: Adenine
    DC = ResType(name="DC", canonical_name="DC")  # DNA: Cytosine
    DG = ResType(name="DG", canonical_name="DG")  # DNA: Guanine
    DT = ResType(name="DT", canonical_name="DT")  # DNA: Thymine
    DX = ResType(name="DX", canonical_name="DX")  # DNA: Unknown
    GAP = ResType(name="-", canonical_name="GAP")  # Gap
    PAD = ResType(name="<PAD>", canonical_name="PAD")  # Unknown

    @staticmethod
    def basic_20_residue_types() -> list[ResType]:
        return [
            ResTypes.A, ResTypes.R, ResTypes.N, ResTypes.D, ResTypes.C,
            ResTypes.Q, ResTypes.E, ResTypes.G, ResTypes.H, ResTypes.I,
            ResTypes.L, ResTypes.K, ResTypes.M, ResTypes.F, ResTypes.P,
            ResTypes.S, ResTypes.T, ResTypes.W, ResTypes.Y, ResTypes.V
        ]

    @staticmethod
    def rna_nucleotide_types() -> list[ResType]:
        return [ResType.RA, ResType.RC, ResType.RG, ResType.RU]

    @staticmethod
    def dna_nucleotide_types() -> list[ResType]:
        return [ResType.DA, ResType.DC, ResType.DG, ResType.DT]


@lru_cache
def get_all_residue_types(model: Model) -> list[ResType]:
    match model:
        case Model.OpenFold2:
            return ResTypes.basic_20_residue_types() + [ResTypes.X]
        case Model.Boltz:
            return [ResTypes.PAD, ResTypes.GAP] + \
                ResTypes.basic_20_residue_types() + [ResTypes.X] + \
                ResTypes.rna_nucleotide_types() + [ResTypes.RX] + \
                ResTypes.dna_nucleotide_types() + [ResTypes.DX]
        case _:
            raise ValueError(f"Invalid model: {model}")


@lru_cache
def get_res_types_index_map(model: Model) -> dict[ResType, int]:
    res_types = get_all_residue_types(model)
    return {res_type: i for i, res_type in enumerate(res_types)}
