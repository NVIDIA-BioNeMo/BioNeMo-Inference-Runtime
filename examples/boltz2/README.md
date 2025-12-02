# Boltz-2

Boltz-2 repo: https://github.com/jwohlwend/boltz/tree/main

## Overview

## Build engines

For Pairformer and TokenTransformer modules, the building progress is similar to the Boltz-1 model in the Boltz1's example folder. Please see: [`examples/boltz1`](examples/boltz1). With the AffinityModule see below:

## Run demo

### Preparing data

```bash
$ pip install boltz==2.2.1
# set pin_memory=False in the data module: https://github.com/jwohlwend/boltz/blob/v2.2.1/src/boltz/data/module/inference.py#L271
$ boltz predict dataset --model boltz2 --cache .cache --out_dir data --use_msa_server --diffusion_samples 1 --recycling_steps 3 --sampling_steps 50 --override
```

### Run scripts

```bash
$ python run_demo.py --processed_dir data/boltz_results_dataset/processed --cache_dir .cache
```

Default backend is `torch`, you can change backend for each module by configurations, see details by `python run_demon.py --help`
