import argparse
import json
import time
from pathlib import Path

import safetensors
from tensorrt_llm import logger

from tensorrt_bionemo.confs.models.boltz2 import Boltz2Config
from tensorrt_bionemo.confs.modules.transformers import PairformerConfig
from tensorrt_bionemo.mapping import Mapping
from tensorrt_bionemo.models.boltz2.convert import convert_hf_pairformer


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
    parser.add_argument('--triangle_attn_backend',
                        type=str,
                        default='VANILLA',
                        choices=['VANILLA', 'TRIFAST'],
                        help='The backend of triangle attention')
    parser.add_argument('--local_checkpoint',
                        type=Path,
                        default=None,
                        help='The path to the local checkpoint')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')

    args = parser.parse_args()
    return args


def convert(worker_rank, world_size, config, args):
    for rank in range(worker_rank, world_size, args.workers):
        mapping = Mapping(world_size=world_size,
                          tp_size=args.tp_size,
                          dcp_size=args.dcp_size,
                          rank=rank)
        weights = convert_hf_pairformer(config, mapping, args.pairformer_type,
                                        args.local_checkpoint)
        safetensors.torch.save_file(weights,
                                    args.output_dir / f'rank{rank}.safetensors')


def main():
    args = parse_arguments()
    world_size = args.tp_size * args.dcp_size

    args.output_dir.mkdir(exist_ok=True, parents=True)

    tik = time.time()
    boltz2_config = Boltz2Config.from_pretrained(
        checkpoint_dir=args.local_checkpoint)
    pairformer_config = boltz2_config.structure_pairformer_backend_config
    if args.pairformer_type == "confidence":
        pairformer_config = boltz2_config.confidence_pairformer_backend_config

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
        "no_update_s":
        False,
        "no_update_z":
        False,
        "backend":
        "trt",
        "token_s":
        boltz2_config.token_s,
        "token_z":
        boltz2_config.token_z,
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
        "triangle_attn_backend":
        args.triangle_attn_backend,
        "post_layer_norm":
        pairformer_config.post_layer_norm,
        "version":
        "v2",
    }
    pairformer_config = PairformerConfig.from_dict(config)
    config = pairformer_config.to_dict()
    with (args.output_dir / 'config.json').open('w') as f:
        json.dump(config, f, indent=4)

    if args.workers == 1:
        convert(0, world_size, pairformer_config, args)
    else:
        if args.workers > world_size:
            args.workers = world_size
        logger.info(f'Convert checkpoint using {args.workers} workers.')
        import torch.multiprocessing as mp
        mp.spawn(convert,
                 nprocs=args.workers,
                 args=(world_size, pairformer_config, args))

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
