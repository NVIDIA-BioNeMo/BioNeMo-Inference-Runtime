# Boltz-2

Boltz-2 repo: https://github.com/jwohlwend/boltz/tree/main

## Overview

## Build engines

For Pairformer and TokenTransformer modules, the building progress is similar to the Boltz-1 model in the Boltz1's example folder. Please see: [`examples/boltz1`](examples/boltz1). With the AffinityModule see below:

### Affinity Module - Convert and Split Weights

This will export for both of `torch` and `trt` backends.

```bash
$ export WORLD_SIZE=8 # world_size must be equal tp_size * dcp_size
$ export TP_SIZE=4
$ export DCP_SIZE=1 # hasn't supported for DCP_SIZE > 1
$ export DTYPE=float32
$ export AFFINITY_MODULE_NAME=affinity_module1 # affinity_module1 or affinity_module2
$ python convert_affinity_checkpoint.py \
    --tp_size ${TP_SIZE} \
    --dtype ${DTYPE} \
    --affinity_module_name ${AFFINITY_MODULE_NAME} \
    --output_dir affinity_module_ckpt_${TP_SIZE}_${DCP_SIZE}_${DTYPE} \
    --triangle_attn_backend CUEQUIV # VANILLA, TRIFAST, CUEQUIV
```

### Affinity Module - Build TensorRT engine(s)

Single worker build

```bash
# For float32
$ export BUILDER_FORCE_NUM_PROFILES=2
$ export MAX_SEQLEN=1536
$ export MIN_SEQLEN=16
$ export TP_SIZE=1
$ export DCP_SIZE=1
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-2 --module affinity_module \
    --checkpoint_dir token_transformer_ckpt_${TP_SIZE}_${DCP_SIZE} \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir affinity_module_${TP_SIZE}_${DCP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines

# For bfloat16
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-2 --module affinity_module \
    --checkpoint_dir affinity_module_ckpt_${TP_SIZE}_${DCP_SIZE} \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir affinity_module_${TP_SIZE}_${DCP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines \
    --weakly_dtype bfloat16
```

## Generate test samples

```
$ python ${_boltz2_dir_}/src/boltz/main.py predict --cache _cache_dir --use_msa_server ${_boltz2_dir_}/examples/prot.yaml
```

## Patching Dropout

```bash
$ vi ${_boltz2_dir_}/model/layers/dropout.py
```

Edit the file to:

```python
import torch
from torch import Tensor


def get_dropout_mask(
    dropout: float,
    z: Tensor,
    training: bool,
    columnwise: bool = False,
) -> Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    torch.Tensor
        The dropout mask

    """
    if not training:
        return 1.
    dropout = dropout * training
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand_like(v) > dropout
    d = d * 1.0 / (1.0 - dropout)
    return d

```
