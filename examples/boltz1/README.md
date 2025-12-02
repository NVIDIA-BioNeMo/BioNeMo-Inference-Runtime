# Boltz-1

## Overview

## Usage

### Pairformer - Convert and Split Weights

This will export for the `trt` backend.

```bash
$ export PAIRFORMER_TYPE=structure # structure or confidence
$ export DTYPE=float32
$ python convert_pairformer_checkpoint.py \
    --pairformer_type ${PAIRFORMER_TYPE} \
    --dtype ${DTYPE}
    --output_dir ${PAIRFORMER_TYPE}_pairformer_ckpt \
    --triangle_attention_backend CUEQUIV # VANILLA, TRIFAST, CUEQUIV
```

Note: For the `CUEQUIV` triangle attention backend the flag `support_batch` default to True, this make the context memory of engine will be reduced by half.

### Pairformer - Build TensorRT engine(s)

Single worker build

```bash
$ export BUILDER_FORCE_NUM_PROFILES=2
$ export MAX_SEQLEN=2048
$ export MIN_SEQLEN=16
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module ${PAIRFORMER_TYPE}_pairformer \
    --checkpoint_dir ${PAIRFORMER_TYPE}_pairformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir engines/${PAIRFORMER_TYPE}_pairformer
```

For bfloat16 precision, add `--weakly_type bfloat16` to build:

```bash
$ export WEAKLY_DTYPE="bfloat16"
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module ${PAIRFORMER_TYPE}_pairformer \
    --checkpoint_dir ${PAIRFORMER_TYPE}_pairformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --weakly_dtype ${WEAKLY_DTYPE} \
    --output_dir engines/${PAIRFORMER_TYPE}_pairformer
```

### Token transformer - Convert and Split Weights

This will export for the `trt` backend.

```bash
$ export DTYPE=float32
$ export MULTIPLICITY=5
$ python convert_token_transformer_checkpoint.py \
    --dtype ${DTYPE} \
    --output_dir token_transformer_ckpt_${DTYPE} \
    --multiplicity ${MULTIPLICITY}
```

### Token transformer - Build TensorRT engine(s)

Single worker build

```bash
# For float32
$ export BUILDER_FORCE_NUM_PROFILES=2
$ export MAX_SEQLEN=2048
$ export MIN_SEQLEN=16
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module token_transformer \
    --checkpoint_dir token_transformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir engines/token_transformer

# For bfloat16 (recommend)
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module token_transformer \
    --checkpoint_dir token_transformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir engines/token_transformer \
    --weakly_dtype bfloat16
```

## Run demo

### Preparing data

```bash
$ pip install boltz==2.2.1
# set pin_memory=False in the data module: https://github.com/jwohlwend/boltz/blob/v2.2.1/src/boltz/data/module/inference.py#L271
$ boltz predict dataset --model boltz1 --cache .cache --out_dir data --use_msa_server --diffusion_samples 1 --recycling_steps 3 --sampling_steps 50 --override
```

### Run scripts

```bash
$ python run_demo.py --processed_dir data/boltz_results_dataset/processed --cache_dir .cache
```

Default backend is `torch`, you can change backend for each module by configurations, see details by `python run_demon.py --help`
