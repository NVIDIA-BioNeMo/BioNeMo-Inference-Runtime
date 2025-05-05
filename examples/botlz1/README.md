# Boltz-1

## Overview

## Usage

### Pairformer - Convert and Split Weights

```bash
$ export WORLD_SIZE=8 # world_size must be equal tp_size * dcp_size
$ export TP_SIZE=4
$ export DCP_SIZE=2
$ export PAIRFORMER_TYPE=structure # structure or confidence
$ python convert_pairformer_checkpoint.py \
    --tp_size ${TP_SIZE} \
    --dcp_size ${DCP_SIZE} \
    --pairformer_type ${PAIRFORMER_TYPE} \
    --output_dir ${PAIRFORMER_TYPE}_pairformer_ckpt_${TP_SIZE}_${DP_SIZE}
```

### Pairformer - Build TensorRT engine(s)

Single worker build

```bash
$ export BUILDER_FORCE_NUM_PROFILES=2
$ export MAX_SEQLEN=1024
$ export MIN_SEQLEN=64
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module ${PAIRFORMER_TYPE}_pairformer \
    --checkpoint_dir ${PAIRFORMER_TYPE}_pairformer_ckpt_${TP_SIZE}_${DP_SIZE} \
    --nccl_plugin float32 \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir ${PAIRFORMER_TYPE}_pairformer_${TP_SIZE}_${DP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines
```

Parallel build (recommend):

```bash
$ python tensorrt_bionemo/commands/build.py \
    --model boltz-1 --module ${PAIRFORMER_TYPE}_pairformer \
    --checkpoint_dir ${PAIRFORMER_TYPE}_pairformer_ckpt_${TP_SIZE}_${DP_SIZE} \
    --nccl_plugin float32 \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir ${PAIRFORMER_TYPE}_pairformer_${TP_SIZE}_${DP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines \
    --workers 8
```

For bfloat32 precision, add

### Benchmark Pairformer

```bash
$ cd benchmarks
$ mpirun -n ${WORLD_SIZE} python benchmark.py \
    -m pairformer \
    --engine_dir ${PAIRFORMER_TYPE}_pairformer_${TP_SIZE}_${DP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines \
    --csv # If you want to export to a csv file
```

## Run demo

### Preparing data

You have to use `botlz predict` to dump the input dict to model.

### Run scripts

```bash
$ pip install boltz==0.4.1 --no-deps
$ pip install pytorch_lightning==2.4.0 fairscale==0.4.13 mashumaro==3.14
```

The import boltz module will go error, so need to change the `boltz/model/layers/triangular_attention/primitives.py` in the site-packges directory. Replace:

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

And patch the function `get_dropout_mask` in `boltz/model/layers/drop_out.py`, this make a consistent prediction at the inference time:

```python
def get_dropout_mask(*args, **kwargs):
    return 1
```

Run the `run_demo.py` script:

```bash
$ export SAMPLE_DIR=sample
$ python run_demo.py --sample_dir ${SAMPLE_DIR} # run with torch original
$ mpirun -n ${WORLD_SIZE} \
    python run_demo.py --sample_dir ${SAMPLE_DIR} \
    --structure_pairformer_engines_dir _path_to_structure_pairformer_ \
    --confidence_pairformer_engines_dir _path_to_confidence_pairformer_
```
