# From https://github.com/jwohlwend/boltz/blob/v2.2.0/src/boltz/data/const.py

CHAIN_TYPES = [
    "PROTEIN",
    "DNA",
    "RNA",
    "NONPOLYMER",
]
CHAIN_TYPE_IDS = {chain: i for i, chain in enumerate(CHAIN_TYPES)}
NUM_CHAIN_TYPES = len(CHAIN_TYPES)

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
NUM_POCKET_CONTACT_INFO = len(POCKET_CONTACT_INFO)

CONTACT_CONDITIONING_INFO = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}

NUM_TOKENS = len(TOKENS)

# Methods
METHOD_TYPES_IDS = {
    "MD": 0,
    "X-RAY DIFFRACTION": 1,
    "ELECTRON MICROSCOPY": 2,
    "SOLUTION NMR": 3,
    "SOLID-STATE NMR": 4,
    "NEUTRON DIFFRACTION": 4,
    "ELECTRON CRYSTALLOGRAPHY": 4,
    "FIBER DIFFRACTION": 4,
    "POWDER DIFFRACTION": 4,
    "INFRARED SPECTROSCOPY": 4,
    "FLUORESCENCE TRANSFER": 4,
    "EPR": 4,
    "THEORETICAL MODEL": 4,
    "SOLUTION SCATTERING": 4,
    "OTHER": 4,
    "AFDB": 5,
    "BOLTZ-1": 6,
    "FUTURE1": 7,  # Placeholder for future supervision sources
    "FUTURE2": 8,
    "FUTURE3": 9,
    "FUTURE4": 10,
    "FUTURE5": 11,
}

METHOD_TYPES_IDS = {k.lower(): v for k, v in METHOD_TYPES_IDS.items()}
NUM_METHOD_TYPES = len(set(METHOD_TYPES_IDS.values()))

BOND_TYPES = [
    "OTHER",
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "AROMATIC",
    "COVALENT",
]
BOND_TYPE_IDS = {bond: i for i, bond in enumerate(BOND_TYPES)}
UNK_BOND_TYPE = "OTHER"
NUM_BOND_TYPES = len(BOND_TYPES)
