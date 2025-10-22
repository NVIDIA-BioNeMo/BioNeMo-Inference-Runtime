import argparse
import json
import time
from pathlib import Path

import torch
import safetensors
from tensorrt_llm import logger

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.openfold3.configs import (OpenFold3Config,
                                                       TokenTransformerConfig)
from tensorrt_bionemo.models.openfold3.convert import \
    convert_hf_token_transformer, convert_hf_token_transformer_torch
from tensorrt_bionemo.runtime.backend import BackendType


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
    parser.add_argument(
        '--max_num_particles',
        type=int,
        default=1,
        help='The max number of particles for the token transformer')
    parser.add_argument(
        '--max_diffusion_samples',
        type=int,
        default=1,
        help='The max number of diffusion samples for the token transformer')
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
                        default='all',
                        choices=['all', 'trt', 'torch'],
                        help='The backend to convert')
    args = parser.parse_args()
    return args


def convert(worker_rank, world_size, configs, args):
    # Dump for tensorrt config
    if args.backend == 'all' or args.backend == BackendType.TRT:
        (args.output_dir / f'{BackendType.TRT}').mkdir(parents=True,
                                                       exist_ok=True)
        with (args.output_dir /
              f'{BackendType.TRT}/config.json').open('w') as f:
            json.dump(configs[BackendType.TRT].to_dict(), f, indent=4)
    # Dump for torch config
    if args.backend == 'all' or args.backend == BackendType.TORCH:
        (args.output_dir / f'{BackendType.TORCH}').mkdir(parents=True,
                                                         exist_ok=True)
        with (args.output_dir /
              f'{BackendType.TORCH}/config.json').open('w') as f:
            json.dump(configs[BackendType.TORCH].to_dict(), f, indent=4)

    for rank in range(worker_rank, world_size, args.workers):
        mapping = Mapping(world_size=world_size,
                          tp_size=args.tp_size,
                          dcp_size=args.dcp_size,
                          rank=rank)
        if args.backend == 'all' or args.backend == BackendType.TRT:
            weights = convert_hf_token_transformer(
                configs[BackendType.TRT],
                mapping,
                local_checkpoint=args.local_checkpoint)
            safetensors.torch.save_file(
                weights,
                args.output_dir / f'{BackendType.TRT}/rank{rank}.safetensors')
        
        if args.backend == 'all' or args.backend == BackendType.TORCH:
            # Save the load_weights_fn and load_weights_fn_kwargs for the torch backend
            weights = convert_hf_token_transformer_torch(
                config=configs[BackendType.TORCH],
                mapping=mapping,
                local_checkpoint=args.local_checkpoint)
            torch.save(weights,
                       args.output_dir / f'{BackendType.TORCH}/weights.pt')
        


def main():
    args = parse_arguments()
    assert args.dcp_size == 1, "Data-context parallelism is not supported for token transformer"
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz1_config = OpenFold3Config.from_pretrained(
        checkpoint_dir=args.local_checkpoint)
    token_transformer_config = boltz1_config.token_transformer_config

    config = {
        # "max_num_particles": args.max_num_particles,
        "max_num_particles": 1,
        "max_diffusion_samples": args.max_diffusion_samples,
        # "max_diffusion_samples": 1,
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
        "version": "v1"
    }
    trt_token_transformer_config = TokenTransformerConfig.from_dict(config)
    torch_token_transformer_config = TokenTransformerConfig.from_dict(config)
    torch_token_transformer_config.backend = BackendType.TORCH

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
        mp.spawn(convert, nprocs=args.workers, args=(world_size, configs, args))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
