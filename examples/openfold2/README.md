# Getting Started

Setup the OpenFold2 inference as the original [guide](https://openfold.readthedocs.io/en/latest/Inference.html)

# Build engines

For Pairformer and TokenTransformer modules, the building progress is similar to the Boltz-1 model in the Boltz1's example folder. Please see: [`examples/boltz1`](examples/boltz1). With the AffinityModule see below:

## Evoformer

### Convert and Split Weights

This will export for both of `torch` and `trt` backends.

```bash
$ export WORLD_SIZE=8 # world_size must be equal tp_size * dcp_size
$ export TP_SIZE=4
$ export DCP_SIZE=1 # hasn't supported for DCP_SIZE > 1
$ export DTYPE=float32
# 'openfold2_finetuning_2', 'openfold2_finetuning_3', 'openfold2_finetuning_4', 'openfold2_finetuning_5','openfold2_no_templ_1', 'openfold2_no_templ_2',
# 'openfold2_no_templ_ptm_1', 'openfold2_ptm_1', 'openfold2_ptm_2'
$ export OPENFOLD2_MODEL_NAME=openfold2_ptm_1
$ python convert_evoformer_checkpoint.py \
    --tp_size ${TP_SIZE} \
    --dtype ${DTYPE} \
    --model_name ${OPENFOLD2_MODEL_NAME} \
    --output_dir evoformer_ckpt_${TP_SIZE}_${DCP_SIZE}_${DTYPE} \
    --triangle_attn_backend CUEQUIV # VANILLA, TRIFAST, CUEQUIV
```

For simplicity, with `world_size=1`. Run:

```bash
$ python convert_evoformer_checkpoint.py --model_name ${OPENFOLD2_MODEL_NAME} --output_dir evoformer_ckpt
```

### Evofromer Module - Build TensorRT engine(s)

Single worker build

```bash
# For float32
$ export BUILDER_FORCE_NUM_PROFILES=2
$ export MAX_SEQLEN=1536
$ export MIN_SEQLEN=16
# can use trtbnm-build instead
$ python tensorrt_bionemo/commands/build.py \
    --model openfold2 --module evoformer \
    --checkpoint_dir token_transformer_ckpt_${TP_SIZE}_${DCP_SIZE} \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir evoformer_ckpt_${TP_SIZE}_${DCP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines

# For bfloat16, use weakly-type
$ python tensorrt_bionemo/commands/build.py \
    --model openfold2 --module evoformer \
    --checkpoint_dir evoformer_ckpt_${TP_SIZE}_${DCP_SIZE} \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir evoformer_${TP_SIZE}_${DCP_SIZE}_${MIN_SEQLEN}_${MAX_SEQLEN}_engines \
    --weakly_dtype bfloat16
```

# Usage

```python
    from tensorrt_bionemo.models.helper import AcceleratedConfig
    from tensorrt_bionemo.models.openfold2.modeling import OpenFold2, OpenFold2AcceleratedModules
    from tensorrt_bionemo.runtime import BackendType, SharedContextMemoryManager
    manager = SharedContextMemoryManager()
    # manager = OnDemandContextMemoryManager(clear_cache_before_forward=True)
    evoformer_stack_config = OpenFold2Config.from_pretrained().evoformer_stack_config
    extra_msa_stack_config = OpenFold2Config.from_pretrained().extra_msa_stack_config
    manager = SharedContextMemoryManager()
    evoformer_stack_config.set_dtype("bfloat16")
    extra_msa_stack_config.set_dtype("bfloat16")
    extra_msa_stack_config.triangle_attn_backend = "CUEQUIV"
    extra_msa_stack_config.opm_chunk_size = 16
    extra_msa_stack_config.opm_mask_chunk_size = 256
    extra_msa_stack_config.padding_inputs = True # This padding make torch.compile() run faster
    acc_m = OpenFold2AcceleratedModules(
        {
            "evoformer": AcceleratedConfig(
                checkpoint="__path_to_engines_dir__",
                backend=BackendType.TRT,
                default=None,
                default=evoformer_stack_config
            ),
            "extra_msa_stack": AcceleratedConfig(
                checkpoint=None,
                backend=BackendType.TORCH,
                default=extra_msa_stack_config,
                compile=True, # This for torch.compile()
                warmup=True,  # Warmup for torch.compile()
            )
        }
    )
    # model is the torch original model
    model, _ = OpenFold2.optimize(model, acc_m, manager)
```
