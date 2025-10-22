import argparse
import copy
import json
import time
from pathlib import Path

import safetensors
import torch
from tensorrt_llm import logger

from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.openfold2.configs import (EvoformerStackConfig,
                                                       OpenFold2Config)
from tensorrt_bionemo.models.openfold2.convert import (
    convert_hf_evoformer, convert_hf_evoformer_torch)
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
    parser.add_argument('--is_multimer',
                        action='store_true',
                        default=False,
                        help='Whether to convert the multimer model')
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
                        default='evoformer_checkpoint',
                        help='The path to save the TensorRT-BNM checkpoint')

    parser.add_argument('--model_name',
                        type=str,
                        default='openfold2_ptm_1',
                        choices=[
                            'openfold2_finetuning_2', 'openfold2_finetuning_3',
                            'openfold2_finetuning_4', 'openfold2_finetuning_5',
                            'openfold2_no_templ_1', 'openfold2_no_templ_2',
                            'openfold2_no_templ_ptm_1', 'openfold2_ptm_1',
                            'openfold2_ptm_2'
                        ],
                        help='The name of the model to convert')
    parser.add_argument('--triangle_attn_backend',
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
                        default='all',
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
    model_name = args.model_name
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
            weights = convert_hf_evoformer(
                configs[BackendType.TRT],
                mapping,
                local_checkpoint=args.local_checkpoint,
                model_name=model_name)
            safetensors.torch.save_file(
                weights,
                args.output_dir / f'{BackendType.TRT}/rank{rank}.safetensors')
        if args.backend == 'all' or args.backend == BackendType.TORCH:
            # Save the load_weights_fn and load_weights_fn_kwargs for the torch backend
            weights = convert_hf_evoformer_torch(
                config=configs[BackendType.TORCH],
                local_checkpoint=args.local_checkpoint,
                model_name=model_name)
            torch.save(weights,
                       args.output_dir / f'{BackendType.TORCH}/weights.pt')


def main():
    args = parse_arguments()
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    openfold2_config = OpenFold2Config.from_pretrained(
        checkpoint_dir=args.local_checkpoint, is_multimer=args.is_multimer)
    evoformer_stack_config = openfold2_config.evoformer_stack_config

    if args.triangle_attn_backend == "CUEQUIV":
        args.support_batch = True

    config = copy.deepcopy(evoformer_stack_config.to_dict())
    config.update({
        "max_batch_size":
        args.max_batch_size,
        # "max_attention_pairwise_tp_size": args.max_attention_pairwise_tp_size,
        # "max_tri_mul_tp_size": args.max_tri_mul_tp_size,
        "dtype":
        args.dtype,
        "architecture":
        "evoformer",
        "mapping": {
            "world_size": world_size,
            "tp_size": args.tp_size,
            "dcp_size": args.dcp_size,
        },
        "disable_custom_all_reduce":
        args.max_attention_pairwise_tp_size or args.max_tri_mul_tp_size,
        "triangle_attn_backend":
        args.triangle_attn_backend,
        "support_batch":
        True,
        "backend":
        "trt",
    })
    trt_evoformer_config = EvoformerStackConfig.from_dict(config)
    torch_evoformer_config = copy.deepcopy(trt_evoformer_config)
    torch_evoformer_config.backend = "torch"

    configs = {
        BackendType.TRT: trt_evoformer_config,
        BackendType.TORCH: torch_evoformer_config
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
