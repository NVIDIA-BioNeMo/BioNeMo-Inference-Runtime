import argparse
import json
import time
from pathlib import Path

# isort: off
from safetensors.torch import save_file
import torch
from tensorrt_llm_lite import logger

from tensorrt_bionemo.configs import BackendType
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz2 import PRETRAINED_CONFIG_REGISTRY
from tensorrt_bionemo.models.boltz2.convert import (convert_hf_pairformer,
                                                    convert_hf_pairformer_torch
                                                    )
# isort: on


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--dcp_size',
                        type=int,
                        default=1,
                        help='N-way data-context parallelism size')
    parser.add_argument('--is_affinity',
                        action='store_true',
                        default=False,
                        help='Whether to convert the affinity model')
    parser.add_argument('--max_transition_tp_size',
                        type=bool,
                        default=True,
                        help='Whether to use the max transition tp size')
    parser.add_argument(
        '--max_attention_pairwise_tp_size',
        type=bool,
        default=True,
        help='Whether to use the max attention pairwise tp size')
    parser.add_argument('--max_tri_mul_tp_size',
                        type=bool,
                        default=True,
                        help='Whether to use the max tri mul tp size')
    parser.add_argument('--triangle_attn_node_chunk_size',
                        type=int,
                        default=0,
                        help='The chunk size for the triangle attention node')
    parser.add_argument(
        '--triangle_attn_cueq_fallback_threshold',
        type=int,
        default=0,
        help=
        'Threshold to fall back from CUEQUIV to VANILLA for triangle attention node'
    )
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=1,
                        help='The max batch size for the pairformer')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['bfloat16', 'float32'])
    parser.add_argument('--output_dir',
                        type=Path,
                        default='pairformer_checkpoint',
                        help='The path to save the TensorRT-BNM checkpoint')

    parser.add_argument('--pairformer_type',
                        type=str,
                        default='structure',
                        choices=['structure', 'confidence'],
                        help='The type of pairformer to convert')
    parser.add_argument('--triangle_attention_backend',
                        type=str,
                        default='CUEQUIV',
                        choices=['VANILLA', 'CUEQUIV'],
                        help='The backend of triangle attention')
    parser.add_argument('--local_checkpoint',
                        type=Path,
                        default=None,
                        help='The path to the local checkpoint')
    parser.add_argument('--support_batch',
                        type=bool,
                        default=False,
                        help='Whether to support batch')
    parser.add_argument('--backend',
                        type=str,
                        default='trt',
                        choices=['all', 'trt', 'torch'],
                        help='The backend to convert')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')

    args = parser.parse_args()
    return args


def convert(worker_rank, world_size, configs, args):
    model_name = "boltz-2"
    if args.is_affinity:
        model_name = "boltz-2-affinity"
    # Dump for tensorrt config
    if args.backend == 'all' or args.backend == BackendType.TRT:
        (args.output_dir / f'{BackendType.TRT}').mkdir(parents=True,
                                                       exist_ok=True)
        with (args.output_dir /
              f'{BackendType.TRT}/config.json').open('w') as f:
            json.dump(configs[BackendType.TRT].model_dump(), f, indent=4)
    # Dump for torch config
    if args.backend == 'all' or args.backend == BackendType.TORCH:
        (args.output_dir / f'{BackendType.TORCH}').mkdir(parents=True,
                                                         exist_ok=True)
        with (args.output_dir /
              f'{BackendType.TORCH}/config.json').open('w') as f:
            json.dump(configs[BackendType.TORCH].model_dump(), f, indent=4)

    for rank in range(worker_rank, world_size, args.workers):
        mapping = Mapping(world_size=world_size,
                          tp_size=args.tp_size,
                          dcp_size=args.dcp_size,
                          rank=rank)
        if args.backend == 'all' or args.backend == BackendType.TRT:
            weights = convert_hf_pairformer(
                configs[BackendType.TRT],
                mapping,
                args.pairformer_type,
                local_checkpoint=args.local_checkpoint,
                model_name=model_name)
            save_file(
                weights,
                args.output_dir / f'{BackendType.TRT}/rank{rank}.safetensors')
        if args.backend == 'all' or args.backend == BackendType.TORCH:
            # Save the load_weights_fn and load_weights_fn_kwargs for the torch backend
            weights = convert_hf_pairformer_torch(
                config=configs[BackendType.TORCH],
                mapping=mapping,
                local_checkpoint=args.local_checkpoint,
                pairformer_type=args.pairformer_type,
                model_name=model_name)
            torch.save(weights,
                       args.output_dir / f'{BackendType.TORCH}/weights.pt')


def main():
    args = parse_arguments()
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz2_config = PRETRAINED_CONFIG_REGISTRY[SupMat.Boltz2]()
    if args.is_affinity:
        boltz2_config = PRETRAINED_CONFIG_REGISTRY[SupMat.Boltz2Affinity]()
    pairformer_config = boltz2_config.trunk.pairformer
    if args.pairformer_type == "confidence":
        pairformer_config = boltz2_config.confidence_module.pairformer

    if args.triangle_attention_backend == "CUEQUIV":
        args.support_batch = True

    config = {
        "max_batch_size":
        args.max_batch_size,
        "max_transition_tp_size":
        args.max_transition_tp_size,
        "max_attention_pairwise_tp_size":
        args.max_attention_pairwise_tp_size,
        "max_tri_mul_tp_size":
        args.max_tri_mul_tp_size,
        "triangle_attn_node_chunk_size":
        args.triangle_attn_node_chunk_size,
        "triangle_attn_cueq_fallback_threshold":
        args.triangle_attn_cueq_fallback_threshold,
        "no_update_s":
        False,
        "no_update_z":
        False,
        "backend":
        "trt",
        "token_s":
        pairformer_config.token_s,
        "token_z":
        pairformer_config.token_z,
        "pairwise_head_width":
        pairformer_config.pairwise_head_width,
        "pairwise_num_heads":
        pairformer_config.pairwise_num_heads,
        "num_blocks":
        pairformer_config.num_blocks,
        "num_heads":
        pairformer_config.num_heads,
        "dtype":
        args.dtype,
        "architecture":
        f"{args.pairformer_type}_pairformer",
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'dcp_size': args.dcp_size,
        },
        "disable_custom_all_reduce":
        args.max_transition_tp_size or args.max_attention_pairwise_tp_size
        or args.max_tri_mul_tp_size,
        "triangle_attention_backend":
        args.triangle_attention_backend,
        "post_layer_norm":
        pairformer_config.post_layer_norm,
        "version":
        "v2",
        "support_batch":
        args.support_batch,
        "trimul_high_precision":
        False
    }
    trt_pairformer_config = pairformer_config.model_copy(update=config)
    torch_pairformer_config = pairformer_config.model_copy(update=config)
    torch_pairformer_config.set_backend(BackendType.TORCH)

    configs = {
        BackendType.TRT: trt_pairformer_config,
        BackendType.TORCH: torch_pairformer_config
    }
    if args.workers == 1:
        convert(0, world_size, configs, args)
    else:
        if args.workers > world_size:
            args.workers = world_size
        logger.info(f'Convert checkpoint using {args.workers} workers.')
        import torch.multiprocessing as mp
        mp.spawn(convert,
                 nprocs=args.workers,
                 args=(world_size, configs, args))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
