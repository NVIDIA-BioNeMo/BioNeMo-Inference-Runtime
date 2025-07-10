# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import copy
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Optional, Union

import torch
from tensorrt_llm._utils import (OMPI_COMM_TYPE_HOST, mpi_barrier, mpi_comm,
                                 mpi_rank, mpi_world_size)
from tensorrt_llm.logger import logger, severity_map
from tensorrt_llm.plugin import PluginConfig, add_plugin_argument

from tensorrt_bionemo import __version__
from tensorrt_bionemo._trt.builder import Engine, EngineConfig, build
from tensorrt_bionemo._trt.layers.affinity import AffinityModule
# TODO: Create a singleton for registering modules
from tensorrt_bionemo._trt.layers.transformers import (PairformerModule,
                                                       TokenTransformer)
from tensorrt_bionemo.configs import BuildModuleConfig, PretrainedModuleConfig
from tensorrt_bionemo.runtime.backend import BackendType

TRT_MODULES_MAPPING = {
    "boltz-1": {
        "structure_pairformer": PairformerModule,
        "confidence_pairformer": PairformerModule,
        "token_transformer": TokenTransformer,
    },
    "boltz-2": {
        "structure_pairformer": PairformerModule,
        "confidence_pairformer": PairformerModule,
        "token_transformer": TokenTransformer,
        "affinity_module": AffinityModule,
    }
}


def get_backend_names(directory_path: str) -> list[str]:
    """
    Returns a list of names of subdirectories within the specified directory.
    """
    backend_names = []
    try:
        # Get all entries in the directory
        entries = os.listdir(directory_path)

        # Iterate through entries and check if they are directories
        for entry in entries:
            full_path = os.path.join(directory_path, entry)
            if os.path.isdir(full_path):
                backend_names.append(entry)
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return backend_names


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        help="The directory path that contains TensorRT-BNM checkpoint.")
    parser.add_argument('--model',
                        type=str,
                        default=None,
                        help="The model name.")
    parser.add_argument('--module',
                        type=str,
                        default=None,
                        help="The module name.")
    parser.add_argument('--max_seqlen',
                        type=int,
                        default=128,
                        help="The maximum sequence length for the model.")
    parser.add_argument('--min_seqlen',
                        type=int,
                        default=64,
                        help="The minimum sequence length for the model.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engine_outputs',
        help=
        "The directory path to save the serialized engine files and engine config file."
    )
    parser.add_argument('--workers',
                        type=int,
                        default=1,
                        help="The number of workers for building in parallel.")
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=severity_map.keys(),
                        help="The logging level.")
    parser.add_argument('--enable_debug_output',
                        default=BuildModuleConfig.enable_debug_output,
                        action='store_true',
                        help="Enable debug output.")
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default=BuildModuleConfig.profiling_verbosity,
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        "The profiling verbosity for the generated TensorRT engine. Setting to detailed allows inspecting tactic choices and kernel parameters."
    )
    parser.add_argument(
        '--dry_run',
        default=BuildModuleConfig.dry_run,
        action='store_true',
        help=
        "Run through the build process except the actual Engine build for debugging."
    )
    parser.add_argument(
        '--input_timing_cache',
        type=str,
        default=BuildModuleConfig.input_timing_cache,
        help=
        "The file path to read the timing cache. This option is ignored if the file does not exist."
    )
    parser.add_argument('--output_timing_cache',
                        type=str,
                        default=BuildModuleConfig.output_timing_cache,
                        help="The file path to write the timing cache.")
    parser.add_argument('--monitor_memory',
                        default=False,
                        action='store_true',
                        help="Enable memory monitor during Engine build.")
    parser.add_argument('--norm_epsilon',
                        type=float,
                        default=1e-5,
                        help="The epsilon value for normalization.")
    parser.add_argument(
        '--mask_inf',
        type=float,
        default=1e9,
        help="The value to mask infinity in the attention mask.")
    parser.add_argument('--weakly_dtype',
                        type=str,
                        default=None,
                        choices=['float16', 'bfloat16', 'float32'],
                        help="The data type of the model.")
    parser.add_argument('--torch_dtype',
                        type=str,
                        default=None,
                        choices=['bfloat16', 'float32'],
                        help="The data type of the torch model.")
    logits_parser = parser.add_argument_group("Logits arguments")
    logits_parser.add_argument('--logits_dtype',
                               type=str,
                               default=None,
                               choices=['bfloat16', 'float32'],
                               help="The data type of logits.")

    plugin_config_parser = parser.add_argument_group("Plugin config arguments")
    add_plugin_argument(plugin_config_parser)
    return parser


def build_module(build_config: BuildModuleConfig,
                 rank: int = 0,
                 ckpt_dir: str = None,
                 module_config: Union[str, PretrainedModuleConfig] = None,
                 module_cls=None,
                 dry_run: bool = False,
                 weakly_dtype: str = None,
                 **kwargs) -> Union[Engine, BuildModuleConfig]:
    module_config = copy.deepcopy(module_config)
    module_config.update_from_dict(kwargs)

    module_config.architecture
    assert rank < module_config.mapping.world_size

    rank_config = copy.deepcopy(module_config)
    rank_config.set_rank(rank)

    # Patch rank config to build config
    build_config.module_config = rank_config
    assert module_cls is not None
    if ckpt_dir is None:
        module = module_cls(rank_config)
    else:
        module = module_cls.from_checkpoint(ckpt_dir, config=rank_config)

    return build(module, build_config)


def build_and_save(rank, gpu_id, ckpt_dir, build_config, output_dir, log_level,
                   module_config, module_cls, **kwargs):
    import tensorrt_bionemo  # load plugins
    torch.cuda.set_device(gpu_id)
    logger.set_level(log_level)
    if module_config.backend == BackendType.TRT:
        engine = build_module(build_config,
                              rank,
                              ckpt_dir,
                              module_config,
                              module_cls=module_cls,
                              **kwargs)
        assert engine is not None
        engine.save(output_dir)
    elif module_config.backend == BackendType.TORCH:
        # copy rank{rank}.pkl to output_dir
        if os.path.exists(os.path.join(ckpt_dir, f"rank{rank}.pkl")):
            shutil.copy(os.path.join(ckpt_dir, f"rank{rank}.pkl"),
                        os.path.join(output_dir, f"rank{rank}.pkl"))
        engine_config = EngineConfig(module_config, BuildModuleConfig(),
                                     __version__)
        engine = Engine(engine_config, None, None)
        engine.save(output_dir)
    else:
        raise ValueError(f"Backend {module_config.backend} is not supported")
    return True


def parallel_build(module_config: PretrainedModuleConfig,
                   ckpt_dir: Optional[str],
                   build_config: BuildModuleConfig,
                   output_dir: str,
                   workers: int = 1,
                   log_level: str = 'info',
                   module_cls=None,
                   **kwargs):

    world_size = module_config.mapping.world_size

    use_mpi = mpi_world_size() > 1

    if not use_mpi and workers == 1:
        for rank in range(world_size):
            passed = build_and_save(rank, rank % workers, ckpt_dir,
                                    build_config, output_dir, log_level,
                                    module_config, module_cls, **kwargs)
            assert passed, "Engine building failed, please check error log."
    elif not use_mpi:
        with ProcessPoolExecutor(mp_context=get_context('spawn'),
                                 max_workers=workers) as p:
            futures = [
                p.submit(build_and_save, rank, rank % workers, ckpt_dir,
                         build_config, output_dir, log_level, module_config,
                         module_cls, **kwargs) for rank in range(world_size)
            ]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(exceptions
                       ) == 0, "Engine building failed, please check error log."
    else:
        mpi_local_comm = mpi_comm().Split_type(split_type=OMPI_COMM_TYPE_HOST)
        mpi_local_rank = mpi_local_comm.Get_rank()
        node_gpu_count = torch.cuda.device_count()
        exceptions = []
        for engine_rank in range(world_size):
            if engine_rank % mpi_world_size() != mpi_rank():
                continue
            try:
                build_and_save(engine_rank, mpi_local_rank % node_gpu_count,
                               ckpt_dir, build_config, output_dir, log_level,
                               module_config, module_cls, **kwargs)
            except Exception as e:
                traceback.print_exc()
                exceptions.append(e)
        mpi_barrier()
        if len(exceptions) != 0:
            print("Engine building failed, please check error log.", flush=True)
            mpi_comm().Abort()


def main():
    parser = parse_arguments()
    args, unknown = parser.parse_known_args()

    if args.checkpoint_dir is None:
        raise ValueError("checkpoint_dir is required")

    logger.set_level(args.log_level)
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    workers = min(torch.cuda.device_count(), args.workers)
    plugin_config = PluginConfig.from_arguments(args)
    plugin_config.validate()

    kwargs = {
        'logits_dtype': args.logits_dtype,
        'norm_epsilon': args.norm_epsilon,
        'mask_inf': args.mask_inf,
    }
    ckpt_dir = args.checkpoint_dir
    backend_names = get_backend_names(ckpt_dir)
    for backend in backend_names:
        if not BackendType.is_supported(backend):
            logger.warning(f"Backend {backend} is not supported")
            continue
        backend_dir = os.path.join(ckpt_dir, backend)
        config_path = os.path.join(backend_dir, 'config.json')
        module_cls = TRT_MODULES_MAPPING[args.model][args.module]
        module_config = PretrainedModuleConfig.from_json_file(
            module_cls, config_path)

        output_dir = os.path.join(args.output_dir, backend)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Building module {args.module} for backend {backend}")
        if backend == BackendType.TRT:
            # TODO: remove this, make it a command line argument
            force_num_profiles_from_env = int(
                os.environ.get("BUILDER_FORCE_NUM_PROFILES", 0))
            if force_num_profiles_from_env is not None:
                logger.warning(
                    f"Overriding # of builder profiles <= {force_num_profiles_from_env}."
                )
            logger.info(
                f"Disable custom all reduce: {module_config.disable_custom_all_reduce}"
            )
            strongly_typed = True
            logger.info(
                f"Module config dtype: {module_config.dtype}, weakly_dtype: {args.weakly_dtype}"
            )
            if args.weakly_dtype is not None and args.weakly_dtype != module_config.dtype:
                strongly_typed = False
                logger.info(
                    f"Building weakly-typed engine with dtype {args.weakly_dtype}."
                )

            build_config_dict = {
                'strongly_typed': strongly_typed,
                'weakly_dtype': args.weakly_dtype,
                'force_num_profiles': force_num_profiles_from_env,
                'profiling_verbosity': args.profiling_verbosity,
                'enable_debug_output': args.enable_debug_output,
                'input_timing_cache': args.input_timing_cache,
                'output_timing_cache': args.output_timing_cache,
                'dry_run': args.dry_run,
                'monitor_memory': args.monitor_memory,
                'max_seqlen': args.max_seqlen,
                'min_seqlen': args.min_seqlen
            }
            build_config = module_cls.build_config_class.from_dict(
                build_config_dict, plugin_config=plugin_config)

            parallel_build(module_config, backend_dir, build_config, output_dir,
                           workers, args.log_level, module_cls, **kwargs)
        elif backend == BackendType.TORCH:
            if args.torch_dtype is not None:
                module_config.set_dtype(args.torch_dtype)
            build_config = None
            parallel_build(module_config, backend_dir, build_config, output_dir,
                           workers, args.log_level, module_cls, **kwargs)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == "__main__":
    main()
