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

### Benchmark Pairformer

```bash
$ cd benchmarks
$ mpirun -n ${WORLD_SIZE} python benchmark.py \
    -m pairformer \
    --engine_dir ${PAIRFORMER_TYPE}_pairformer_${TP_SIZE}_${DP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines \
    --csv # If you want to export to a csv file
```
