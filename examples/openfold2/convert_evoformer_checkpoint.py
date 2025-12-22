import argparse
import copy
import json
import time
from pathlib import Path

import safetensors
from tensorrt_llm_lite import logger

from tensorrt_bionemo.configs import BackendType
from tensorrt_bionemo.hubs import FoldingSupportMatrix as SupMat
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.openfold2 import PRETRAINED_CONFIG_REGISTRY
from tensorrt_bionemo.models.openfold2.convert import convert_hf_evoformer


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
                        default=SupMat.OpenFold2_PTM1,
                        choices=[
                            SupMat.OpenFold2_FT2,
                            SupMat.OpenFold2_FT3,
                            SupMat.OpenFold2_FT4,
                            SupMat.OpenFold2_FT5,
                            SupMat.OpenFold2_NoTempl1,
                            SupMat.OpenFold2_NoTempl2,
                            SupMat.OpenFold2_NoTempl_PTM1,
                            SupMat.OpenFold2_PTM1,
                            SupMat.OpenFold2_PTM2,
                            SupMat.AlphaFold2_1,
                            SupMat.AlphaFold2_2,
                            SupMat.AlphaFold2_3,
                            SupMat.AlphaFold2_4,
                            SupMat.AlphaFold2_5,
                            SupMat.AlphaFold2_Multimer_1,
                            SupMat.AlphaFold2_Multimer_2,
                            SupMat.AlphaFold2_Multimer_3,
                            SupMat.AlphaFold2_Multimer_4,
                            SupMat.AlphaFold2_Multimer_5,
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
                        choices=['all', 'trt'],
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
            json.dump(configs[BackendType.TRT].model_dump(), f, indent=4)
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


def main():
    args = parse_arguments()
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    config = PRETRAINED_CONFIG_REGISTRY[args.model_name]()
    evoformer_stack_config = config.trunk.evoformer_stack

    if args.triangle_attn_backend == "CUEQUIV":
        args.support_batch = True

    if not config.is_multimer:
        n_seq = 516
        if not config.enable_template:
            n_seq = 512
    else:
        n_seq = 256

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
        "triangle_attention_backend":
        args.triangle_attn_backend,
        "support_batch":
        True,
        "backend":
        "trt",
        "n_seq":
        n_seq,
        "trimul_high_precision":
        False,
    })
    trt_evoformer_config = evoformer_stack_config.model_copy(update=config)

    configs = {
        BackendType.TRT: trt_evoformer_config,
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
