# OpenFold3

## Overview

The converting process is quite similar to the Boltz1-converting process, please see the Boltz1's documentation [here](examples/boltz1/README.md)

## Usage

```python
    from tensorrt_bionemo.models.openfold3 import OpenFold3 as OpenFold3Opt, OpenFold3Config, OpenFold3AcceleratedModules
    from tensorrt_bionemo.models.helper import AcceleratedConfig
    from tensorrt_bionemo.runtime import BackendType, OnDemandContextMemoryManager
    config = OpenFold3Config.from_pretrained()
    manager = OnDemandContextMemoryManager()

    acc_m = OpenFold3AcceleratedModules(
        configs={
            "pairformer": AcceleratedConfig(
                checkpoint=pairformer_trt_engine_path.as_posix(),
                backend=BackendType.TRT,
                default=config.pairformer_config,
            ),
            "token_transformer": AcceleratedConfig(
                checkpoint=token_transformer_trt_engine_path.as_posix(),
                backend=BackendType.TRT,
                default=config.token_transformer_config,
            ),
        }
    )
    model = OpenFold3(model_config)
    model, _ = OpenFold3Opt.optimize(model, acc_m, manager)
```
