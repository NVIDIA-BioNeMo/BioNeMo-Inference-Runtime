# From https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/data/const.py

####################################################################################################
# RESIDUES & TOKENS
####################################################################################################

CANONICAL_TOKENS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",  # codespell:ignore
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",  # unknown protein token
]

TOKENS = [
    "<pad>",
    "-",
    *CANONICAL_TOKENS,
    "A",
    "G",
    "C",
    "U",
    "N",  # unknown rna token
    "DA",
    "DG",
    "DC",
    "DT",
    "DN",  # unknown dna token
]

TOKENS_ID = {token: i for i, token in enumerate(TOKENS)}

POCKET_CONTACT_INFO = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET": 2,
    "BINDER": 3,
}
