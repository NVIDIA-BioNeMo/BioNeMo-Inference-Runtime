# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import copy
import time

import tensorrt as trt
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder, Engine, EngineConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

from tensorrt_bionemo._trt.module_utils import PretrainedModule
from tensorrt_bionemo.configs import BuildModuleConfig
from tensorrt_bionemo.version import __version__


def build(module: PretrainedModule, build_config: BuildModuleConfig = None):
    tic = time.time()

    build_config = copy.deepcopy(build_config)
    build_config.plugin_config.dtype = module.config.dtype

    module_config = module.config
    builder = Builder()
    if build_config.strongly_typed:
        precision = module.config.dtype
    else:
        precision = build_config.weakly_dtype
        # TODO: Refactor here
        module.config.dtype = precision  # change precision for weakly-typed mode
        module_config.dtype = precision

    builder_config = builder.create_builder_config(
        precision=precision,
        use_refit=False,  # TODO: add refit
        timing_cache=build_config.input_timing_cache,
        strongly_typed=build_config.strongly_typed,
        force_num_profiles=build_config.force_num_profiles,
        profiling_verbosity=build_config.profiling_verbosity,
        monitor_memory=build_config.monitor_memory,
    )
    # TODO: make to args
    # builder_config.trt_builder_config.max_aux_streams = 0
    builder_config.trt_builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 64 * (2**30))
    builder_config.trt_builder_config.builder_optimization_level = 5

    network = builder.create_network()
    network.trt_network.name = module_config.architecture
    network.plugin_config = build_config.plugin_config

    nccl_plugin = None
    if module.config.mapping.world_size > 1:
        if build_config.plugin_config.nccl_plugin is not None:
            nccl_plugin = build_config.plugin_config.nccl_plugin
        else:
            nccl_plugin = module.config.dtype
    network.plugin_config.set_nccl_plugin(nccl_plugin)

    with net_guard(network):
        prepare_input_args = {
            "opt_profiles": build_config.optimization_profiles,
            "has_attention": build_config.has_attention,
            "disable_custom_all_reduce": module_config.disable_custom_all_reduce
        }
        inputs = module.prepare_inputs(**prepare_input_args)
        outputs = module(**inputs)
        if not isinstance(outputs, tuple) and not isinstance(outputs, list):
            outputs = (outputs, )

        output_names = module.config.get_output_names()
        for output, output_name in zip(outputs, output_names):
            output.mark_output(output_name,
                               str_dtype_to_trt(module.config.dtype))

    if not build_config.strongly_typed:
        # Modify the network for weakly-typed mode
        if precision != "float32":
            builder_config.trt_builder_config.set_flag(trt.BuilderFlag.TF32)
        network = module.weakly_typed(network, precision)

    # Network -> Engine
    logger.info(
        f"Total time of constructing network from module object {time.time()-tic} seconds"
    )
    logger.info(f"Building Engine for rank {module_config.mapping.rank}")
    managed_weights = {} if network.plugin_config.manage_weights else None
    engine = None if build_config.dry_run else builder.build_engine(
        network, builder_config, managed_weights)

    engine_config = EngineConfig(module_config, build_config, __version__)

    if build_config.output_timing_cache is not None and module_config.mapping.rank == 0:
        ok = builder.save_timing_cache(builder_config,
                                       build_config.output_timing_cache)
        assert ok, "Failed to save timing cache."

    import psutil

    # Get the current process
    current_process = psutil.Process()
    # Get resource usage for the current process (self)
    rusage_s = current_process.memory_info()
    # Get resource usage for all child processes
    children = current_process.children(recursive=True)
    rusage_c = [child.memory_info() for child in children]
    logger.info(
        f"Build phase peak memory: {rusage_s.rss / 1024 / 1024:.2f} MB, children: {sum([ru.rss for ru in rusage_c]) / 1024 / 1024:.2f} MB"
    )

    return Engine(engine_config, engine, managed_weights)
