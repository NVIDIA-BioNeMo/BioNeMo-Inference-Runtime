import glob
import json

import torch

pdb_ids = [
    ele.split('/feats_')[1][:4] for ele in glob.glob(f"sample/feats*.pt")
]
seqlens = []
new_pdb_ids = []
# ids = ["8eil", "8c4d", "7qsj", "8cpk", "8are", "8owf", "7tpu", "7ylz", "8gpp", "8clz", "8k7x"]

ids = {}
for pdb_id in pdb_ids:
    feats_path = f"sample/feats_{pdb_id}.pt"
    batch = torch.load(feats_path, weights_only=False)
    for key, val in batch.items():
        if hasattr(val, "to"):
            if key in ["msa_mask"]:
                seqlen = val.shape[-1]
                ids[pdb_id] = seqlen
                print(f"Load pdb_id: {pdb_id} with seqlen: {seqlen}")

with open("sample/ids.json", "w") as f:
    json.dump(ids, f)
