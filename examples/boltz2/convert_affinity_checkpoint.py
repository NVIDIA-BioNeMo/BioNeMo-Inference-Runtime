import argparse
import json
import time
from pathlib import Path

import safetensors
import torch
from tensorrt_llm import logger

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz2.configs import (AffinityModuleConfig,
                                                    Boltz2Config)
from tensorrt_bionemo.models.boltz2.convert import (
    convert_hf_affinity_module, convert_hf_affinity_module_torch)
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
    parser.add_argument('--affinity_module_name',
                        type=str,
                        default='affinity_module1',
                        help='The name of the affinity module')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['bfloat16', 'float32'])
    parser.add_argument('--output_dir',
                        type=Path,
                        default='token_transformer_checkpoint',
                        help='The path to save the TensorRT-BNM checkpoint')
    parser.add_argument('--triangle_attn_backend',
                        type=str,
                        default='CUEQUIV',
                        choices=['VANILLA', 'TRIFAST', 'CUEQUIV'],
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
            weights = convert_hf_affinity_module(
                configs[BackendType.TRT],
                mapping,
                affinity_module_name=args.affinity_module_name,
                local_checkpoint=args.local_checkpoint)
            safetensors.torch.save_file(
                weights,
                args.output_dir / f'{BackendType.TRT}/rank{rank}.safetensors')
        if args.backend == 'all' or args.backend == BackendType.TORCH:
            # Save the load_weights_fn and load_weights_fn_kwargs for the torch backend
            weights = convert_hf_affinity_module_torch(
                local_checkpoint=args.local_checkpoint,
                world_size=world_size,
                rank=rank)
            torch.save(weights,
                       args.output_dir / f'{BackendType.TORCH}/weights.pt')


def main():
    args = parse_arguments()
    assert args.dcp_size == 1, "Data-context parallelism is not supported for token transformer"
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz2_config = Boltz2Config.from_pretrained(
        checkpoint_dir=args.local_checkpoint, is_affinity=True)
    affinity_module_configs = boltz2_config.affinity_module_configs.get(
        args.affinity_module_name, None)
    assert affinity_module_configs is not None, f"Affinity module {args.affinity_module_name} not found"

    config = {
        "token_s": affinity_module_configs.token_s,
        "token_z": affinity_module_configs.token_z,
        "num_dist_bins": affinity_module_configs.num_dist_bins,
        "max_dist": affinity_module_configs.max_dist,
        "pairformer_num_blocks": affinity_module_configs.pairformer_num_blocks,
        "pairwise_head_width": affinity_module_configs.pairwise_head_width,
        "pairwise_num_heads": affinity_module_configs.pairwise_num_heads,
        "max_batch_size": 1,
        "dtype": args.dtype,
        "architecture": "affinity_module",
        'mapping': {
            'world_size': world_size,
            'tp_size': args.tp_size,
            'dcp_size': 1,
        },
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "triangle_attn_backend": args.triangle_attn_backend,
        "backend": BackendType.TRT,
        "version": "v2"
    }
    trt_affinity_module_config = AffinityModuleConfig.from_dict(config)
    torch_affinity_module_config = AffinityModuleConfig.from_dict(config)
    torch_affinity_module_config.backend = BackendType.TORCH

    configs = {
        BackendType.TRT: trt_affinity_module_config,
        BackendType.TORCH: torch_affinity_module_config,
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
