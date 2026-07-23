# Getting Started

Setup the OpenFold2 inference as the original [guide](https://openfold.readthedocs.io/en/latest/Inference.html)

# Build engines

For Pairformer and TokenTransformer modules, the building progress is similar to the Boltz-1 model in the Boltz1's example folder. Please see: [`examples/boltz1`](examples/boltz1). With the AffinityModule see below:

## Evoformer

### Convert AlphaFold jax checkpoints

Need to install the OpenFold2 original code, and download AlphaFold2 JAX checkpoints. \[See\](# https://openfold.readthedocs.io/en/latest/Inference.html#download-model-parameters):
Run:

```bash
$ python jax_to_pt.py --jax_path params_model_1.npz --config_preset  model_1 --output_dir output
```

Configurations Refer Table:

```
Setting                         config_preset          AlphaFold params                    OpenFold params
---------------------------------------------------------------------------------------------------------------
With template, no ptm           model_1                params_model_1.npz                  finetuning_[2-5].pt
                                model_2                params_model_2.npz

With template, with ptm         model_1_ptm            params_model_1_ptm.npz              finetuning_ptm_[1-2].pt
                                model_2_ptm            params_model_2_ptm.npz

Without template, no ptm        model_3                params_model_3.npz                  finetuning_no_templ_[1-2].pt
                                model_4                params_model_4.npz
                                model_5                params_model_5.npz

Without template, with ptm      model_3_ptm            params_model_3_ptm.npz              finetuning_no_templ_ptm_1.pt
                                model_4_ptm            params_model_4_ptm.npz
                                model_5_ptm            params_model_5_ptm.npz
```

### Convert and Split Weights

This will export for both of `torch` and `trt` backends.

```bash
$ export DTYPE=float32
# 'openfold2_finetuning_2', 'openfold2_finetuning_3', 'openfold2_finetuning_4', 'openfold2_finetuning_5','openfold2_no_templ_1', 'openfold2_no_templ_2',
# 'openfold2_no_templ_ptm_1', 'openfold2_ptm_1', 'openfold2_ptm_2', 'alphafold2_1', 'alphafold2_2', 'alphafold2_3', 'alphafold2_4', alphafold2_5'
# 'alphafold2_multimer_1', 'alphafold2_multimer_2', 'alphafold2_multimer_3', 'alphafold2_multimer_4', 'alphafold2_multimer_5'
$ export OPENFOLD2_MODEL_NAME=openfold2_ptm_1
$ python convert_evoformer_checkpoint.py \
    --dtype ${DTYPE} \
    --model_name ${OPENFOLD2_MODEL_NAME} \
    --output_dir evoformer_ckpt\
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
$ export MAX_SEQLEN=1536
$ export MIN_SEQLEN=16
# please use tensorrt_bionemo/commands/build.py instead (if in development), trtbnm-build command may not in PATH.
$ trtbnm-build \
    --model ${OPENFOLD2_MODEL_NAME} --module evoformer \
    --checkpoint_dir evoformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir evoformer_ckpt_engines

# For bfloat16, use weakly-type (recommend)
$ trtbnm-build \
    --model ${OPENFOLD2_MODEL_NAME} --module evoformer \
    --checkpoint_dir evoformer_ckpt \
    --max_seqlen ${MAX_SEQLEN} \
    --min_seqlen ${MIN_SEQLEN} \
    --output_dir evoformer_engines \
    --weakly_dtype bfloat16
```

# Usage

Note that: the .pt checkpoints of multimer models aren't available on HuggingFace. It has to set to environment variables, i.e:

```bash
export ALPHAFOLD2_MULTIMER_1_CKPT=__checkpoint_dir__/alphafold_params/params_model_1_multimer_v3.pt
```

See all checkpoint environment variables in `hubs/local.py`.

Normal cases:

```python
    from tensorrt_bionemo.models.optimize_module_setter import AcceleratedConfig
    from tensorrt_bionemo.models.openfold2.modeling import OpenFold2, OpenFold2AcceleratedModules
    from tensorrt_bionemo.runtime import BackendType, OnDemandContextMemoryManager
    manager = OnDemandContextMemoryManager()

    model = OpenFold2(model_name=model_name)
    model.cuda()
    model.eval()
    acc_m = {
        "evoformer":
        AcceleratedConfig(
            checkpoint=evoformer_ckpt,
            backend=evoformer_backend,
            need_fallback=NeedFallbackEvoformer(evoformer_fallback_threshold),
        )
    }
    model = model.optimize(acc_m, manager)
```

Please take a look on the `run_demo.py` script to run with TRT-BNM (default torch-backend, fallback optional on trt-backend)

```bash
$ python run_demo.py \
    _fasta_dir_ \
    _template_cif_dir_ \
    --use_precomputed_alignments _msa_precomputed_dir_ \
    --model_name alphafold2_1 \
    --output_dir demo_casp14 \
    --skip_relaxation
# Add option to run with trt
    # --evoformer_backend trt \
    # --evoformer_ckpt _engines_path_
```

## Longer sequences ~ 3k

For longer sequences, it recommends to use the pytorch backend (or trt-backend with fallback threshold) and chunking configurations for the OuterProductMean module:

```python
config = OpenFold2.get_pretrained_config(model_name)
config.trunk.evoformer_stack.opm_chunk_size = 16
config.trunk.evoformer_stack.opm_mask_chunk_size = 128 # 64, 128, 256, 512
model = OpenFold2(config=config, model_name=model_name)
```

# Run with ray.

The `ray_pipeline.ipynb` demonstrate for running with ray map_batches stage, hide (pre/post)processing latencies.
