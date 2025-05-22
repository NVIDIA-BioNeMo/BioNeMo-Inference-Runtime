# Boltz-2

Currently, Boltz-2 is private and access is restricted to authorized users only.

## Overview

Install boltz-2 from directory

```bash
$ cd ${_boltz2_dir_}
$ pip install -e .
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

The import boltz module will go error, so need to change the `${_boltz2_dir_}/model/layers/triangular_attention/primitives.py` in the site-packges directory. Replace:

```python
fa_is_installed = importlib.util.find_spec("flash_attn") is not None
if fa_is_installed:
    from flash_attn.bert_padding import unpad_input
    from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
```

by

```python
fa_is_installed = False
```
