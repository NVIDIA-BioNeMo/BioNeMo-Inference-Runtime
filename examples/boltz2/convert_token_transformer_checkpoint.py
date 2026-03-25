import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tensorrt_llm_lite import logger

from tensorrt_bionemo.configs import BackendType
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz2 import PRETRAINED_CONFIG_REGISTRY
from tensorrt_bionemo.models.boltz2.convert import (
    convert_hf_diffusion_transformer, convert_hf_diffusion_transformer_torch)


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
    parser.add_argument('--multiplicity',
                        type=int,
                        default=4,
                        help='The multiplicity for the token transformer')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['bfloat16', 'float32'])
    parser.add_argument('--output_dir',
                        type=Path,
                        default='token_transformer_checkpoint',
                        help='The path to save the TensorRT-BNM checkpoint')
    parser.add_argument('--pairwise_attn_backend',
                        type=str,
                        default='VANILLA',
                        choices=['VANILLA'],
                        help='The backend of pairwise attention')
    parser.add_argument('--local_checkpoint',
                        type=Path,
                        default=None,
                        help='The path to the local checkpoint')
    parser.add_argument(
        '--disable_custom_all_reduce',
        type=bool,
        default=True,
        help='Whether to disable custom all reduce - the multishot algorithm')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--backend',
                        type=str,
                        default='trt',
                        choices=['all', 'trt', 'torch'],
                        help='The backend to convert')
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
            weights = convert_hf_diffusion_transformer(
                configs[BackendType.TRT],
                mapping,
                local_checkpoint=args.local_checkpoint,
                model_name=model_name)
            save_file(
                weights,
                args.output_dir / f'{BackendType.TRT}/rank{rank}.safetensors')
        if args.backend == 'all' or args.backend == BackendType.TORCH:
            # Save the load_weights_fn and load_weights_fn_kwargs for the torch backend
            weights = convert_hf_diffusion_transformer_torch(
                config=configs[BackendType.TORCH],
                mapping=mapping,
                local_checkpoint=args.local_checkpoint,
                model_name=model_name)
            torch.save(weights,
                       args.output_dir / f'{BackendType.TORCH}/weights.pt')


def main():
    args = parse_arguments()
    assert args.dcp_size == 1, "Data-context parallelism is not supported for token transformer"
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz2_config = PRETRAINED_CONFIG_REGISTRY[SupMat.Boltz2]()
    if args.is_affinity:
        boltz2_config = PRETRAINED_CONFIG_REGISTRY[SupMat.Boltz2Affinity]()
    token_transformer_config = boltz2_config.structure_module.score_model.token_transformer

    config = {
        "multiplicity": args.multiplicity,
        "backend": "trt",
        "num_blocks": token_transformer_config.num_blocks,
        "num_heads": token_transformer_config.num_heads,
        "dim": token_transformer_config.dim,
        "dim_single_cond": token_transformer_config.dim_single_cond,
        "dim_pairwise": token_transformer_config.dim_pairwise,
        "dtype": args.dtype,
        "architecture": "token_transformer",
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'dcp_size': 1,
        },
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "pairwise_attn_backend": args.pairwise_attn_backend,
        "version": "v2"
    }
    trt_token_transformer_config = token_transformer_config.model_copy(
        update=config)
    torch_token_transformer_config = token_transformer_config.model_copy(
        update=config)
    torch_token_transformer_config.set_backend(BackendType.TORCH)

    configs = {
        BackendType.TRT: trt_token_transformer_config,
        BackendType.TORCH: torch_token_transformer_config
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
