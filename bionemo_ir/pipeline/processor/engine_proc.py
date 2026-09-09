# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Callable
from typing import Any

from pydantic import Field

from bionemo_ir.data.utils import get_all_atom_types, get_all_residue_types
from bionemo_ir.pipeline.processor.base import Processor, ProcessorConfig, SerialProcessor, _ProcessorBase
from bionemo_ir.pipeline.processor.utils import build_cpu_stage_map_kwargs, get_available_gpu_count
from bionemo_ir.pipeline.stages import (
    FeatureGeneratorStage,
    FoldingEngineStage,
    ParserStage,
    TokenizerStage,
    WriterStage,
)
from bionemo_ir.pipeline.stages.base import StatefulStage
from bionemo_ir.pipeline.stages.configs import (
    EngineStageConfig,
    FeatureGeneratorStageConfig,
    ParallelismMode,
    ParserStageConfig,
    TokenizerStageConfig,
    WriterStageConfig,
    resolve_stage_config,
)
from bionemo_ir.registry import (
    get_default_runtime_args,
    get_feature_factory,
    get_model_class,
    get_tokenizer,
    load_metadata,
)


class EngineProcessorConfig(ProcessorConfig):
    model_source: str = Field(
        default=None,
        description="The model source to use for the processor.",
    )
    engine_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="The kwargs to pass to the engine. {'config': Optional[BaseConfig], 'accelerated_configs': Optional[Dict[str, AcceleratedConfig]]}",
    )
    parser_stage: Any = Field(
        default=True,
        description="Parser stage config (bool | dict | ParserStageConfig).",
    )
    tokenizer_stage: Any = Field(
        default=True,
        description="Tokenizer stage config (bool | dict | TokenizerStageConfig).",
    )
    feature_generator_stage: Any = Field(
        default=True,
        description="Feature generator stage config (bool | dict | FeatureGeneratorStageConfig).",
    )
    writer_stage: Any = Field(
        default=True,
        description="Writer stage config (bool | dict | WriterStageConfig).",
    )
    engine_stage: Any = Field(
        default=True,
        description="Engine stage config (bool | dict | EngineStageConfig). "
        "Controls folding engine replicas and GPU allocation.",
    )
    runtime_env: dict[str, Any] | None = Field(
        default=None,
        description="The runtime environment to use for the offline processing.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata (e.g. ccd_path, mol_dir) for context "
        "generators and feature generators. When None, the model factory's "
        "load_metadata (or a custom metadata_loader) is called automatically.",
    )
    metadata_loader: Callable[[], dict[str, Any]] | None = Field(
        default=None,
        description="Optional callable that returns a metadata dict. "
        "When set, overrides the model factory's default load_metadata. "
        "Ignored if metadata is provided explicitly.",
    )
    runtime_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime arguments passed to model.forward() alongside "
        "feed_dict. For Boltz models: recycling_steps, num_sampling_steps, "
        "diffusion_samples, steering_args, etc.",
    )

    max_pending_requests: int | None = Field(
        default=None,
        description="The maximum number of pending requests. If not specified, "
        "will use the default value from the backend engine.",
    )
    max_concurrent_batches: int = Field(
        default=8,
        description="The maximum number of concurrent batches in the engine. "
        "This is to overlap the batch processing to avoid the tail latency of "
        "each batch. The default value may not be optimal when the batch size "
        "or the batch processing latency is too small, but it should be good "
        "enough for batch size >= 32.",
    )

    @classmethod
    def create_default_replica_mode_config(
        cls,
        model_source: str,
        output_dir: str,
        output_format: str = "pdb",
        parser_stage_actors: int | None = None,
        tokenizer_stage_actors: int | None = None,
        tokenizer_stage_num_cpus: int | None = None,
        feature_generator_stage_actors: int | None = None,
        feature_generator_stage_num_cpus: int | None = None,
        engine_stage_num_gpus: int | None = None,
        engine_stage_num_cpus: int | None = None,
        engine_stage_memory: int | None = None,
        writer_stage_actors: int | None = None,
        writer_stage_num_cpus: int | None = None,
        should_continue_on_error: bool = True,
    ) -> "EngineProcessorConfig":
        """Build a processor config for replica mode (one engine per GPU) using all available GPUs."""
        num_gpus = get_available_gpu_count()
        return cls(
            model_source=model_source,
            executor_backend="ray",
            parser_stage=ParserStageConfig(compute=parser_stage_actors or num_gpus),
            tokenizer_stage=TokenizerStageConfig(
                compute=tokenizer_stage_actors or num_gpus, num_cpus=tokenizer_stage_num_cpus or 4
            ),
            feature_generator_stage=FeatureGeneratorStageConfig(
                num_cpus=feature_generator_stage_num_cpus or 8, compute=feature_generator_stage_actors or num_gpus
            ),
            writer_stage=WriterStageConfig(
                compute=writer_stage_actors or num_gpus,
                output_path=output_dir,
                format=output_format,
                num_cpus=writer_stage_num_cpus or 1,
            ),
            engine_stage=EngineStageConfig(
                parallelism_mode=ParallelismMode.REPLICA,
                compute=engine_stage_num_gpus or num_gpus,
                num_cpus=engine_stage_num_cpus or 4,
                memory=engine_stage_memory,
            ),
            should_continue_on_error=should_continue_on_error,
        )

    def get_model_pretrained_config(self):
        model_class = get_model_class(self.model_source)
        result = model_class.get_pretrained_config(self.model_source)
        if "config" in self.engine_kwargs:
            # Override the default pretrained config
            result = self.engine_kwargs["config"]
        return result


def _build_parser_stage(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> StatefulStage:
    parser_stage_cfg = resolve_stage_config(config.parser_stage, ParserStageConfig, processor_defaults)
    return ParserStage(
        fn_constructor_kwargs={},
        map_batches_kwargs=build_cpu_stage_map_kwargs(parser_stage_cfg),
        compute_by_rows=parser_stage_cfg.compute_by_rows,
        drop_keys=parser_stage_cfg.drop_keys,
    )


def _build_tokenizer_stage(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> StatefulStage:
    model_pretrained_config = config.get_model_pretrained_config()
    tokenizer_stage_cfg = resolve_stage_config(config.tokenizer_stage, TokenizerStageConfig, processor_defaults)
    feature_generator_stage_cfg = resolve_stage_config(
        config.feature_generator_stage, FeatureGeneratorStageConfig, processor_defaults
    )
    tokenizer = get_tokenizer(config.model_source)
    feature_factory = get_feature_factory(config.model_source)
    context_generators = {}
    for k, generator_spec in tokenizer.context_generator_specs.items():
        context_generators[k] = generator_spec.generator(
            config=model_pretrained_config,
            metadata=config.metadata,
        )
        context_generators[k].required_kwargs = generator_spec.required_kwargs

    transform_funcs = []
    for transform_spec in tokenizer.transform_specs:
        transform_cls = transform_spec.transform
        transform_funcs.append(transform_cls(config=model_pretrained_config, **transform_spec.kwargs))

    # Prefer tokenizer-stage init_context; fall back to feature-stage so a single
    # random_seed seeds both ETKDG (tokenizer) and augmentation (feature stage).
    init_context = tokenizer_stage_cfg.init_context
    if init_context is None:
        init_context = feature_generator_stage_cfg.init_context

    return TokenizerStage(
        fn_constructor_kwargs={
            "context_generators": context_generators,
            "context_merger_func": tokenizer.context_merger_func,
            "transform_funcs": transform_funcs,
            "pre_init": feature_factory.pre_init,
            "init_context": init_context,
        },
        map_batches_kwargs=build_cpu_stage_map_kwargs(tokenizer_stage_cfg),
        compute_by_rows=tokenizer_stage_cfg.compute_by_rows,
        drop_keys=tokenizer_stage_cfg.drop_keys,
    )


def _build_feature_generator_stage(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> StatefulStage:
    model_pretrained_config = config.get_model_pretrained_config()
    feature_generator_stage_cfg = resolve_stage_config(
        config.feature_generator_stage, FeatureGeneratorStageConfig, processor_defaults
    )

    feature_factory = get_feature_factory(config.model_source)
    feature_generators = []
    feature_collators = []
    # Initialize feature generators and collators

    for generator_spec in feature_factory.feature_generator_specs:
        generator = generator_spec.functor(
            config=model_pretrained_config, metadata=config.metadata, **generator_spec.kwargs
        )
        generator.name = generator_spec.name
        feature_generators.append(generator)
    for collator_spec in feature_factory.feature_collator_specs:
        collator = collator_spec.functor(
            config=model_pretrained_config, metadata=config.metadata, **collator_spec.kwargs
        )
        collator.name = collator_spec.name
        feature_collators.append(collator)
    return FeatureGeneratorStage(
        fn_constructor_kwargs={
            "feature_generators": feature_generators,
            "feature_collators": feature_collators,
            "features_merger_func": feature_factory.features_merger_func,
            "pre_init": feature_factory.pre_init,
            "init_context": feature_generator_stage_cfg.init_context,
        },
        # TODO: consider using GPU for feature factory
        map_batches_kwargs=build_cpu_stage_map_kwargs(feature_generator_stage_cfg),
        compute_by_rows=feature_generator_stage_cfg.compute_by_rows,
        # Drop parsed to save memory
        drop_keys=feature_generator_stage_cfg.drop_keys
        if feature_generator_stage_cfg.drop_keys is not None
        else ["parsed"],
    )


def _build_folding_engine_stage(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> StatefulStage:
    engine_stage_cfg = resolve_stage_config(config.engine_stage, EngineStageConfig, processor_defaults)

    fn_constructor_kwargs = {
        "model": config.model_source,
        "engine_kwargs": config.engine_kwargs,
        "max_pending_requests": config.max_pending_requests,
        "should_continue_on_error": config.should_continue_on_error,
        "parallelism_mode": engine_stage_cfg.parallelism_mode,
        "runtime_args": config.runtime_args or None,
    }

    if config.executor_backend == "ray":
        import ray

        available_gpus = get_available_gpu_count()
        num_gpus_per_replica = engine_stage_cfg.num_gpus

        compute = engine_stage_cfg.compute
        if compute is None:
            max_replicas = max(1, int(available_gpus / num_gpus_per_replica))
            compute = max_replicas
        if isinstance(compute, int):
            compute_range = (compute, compute)
        else:
            compute_range = compute

        max_replicas = compute_range[1]
        total_gpus_needed = max_replicas * num_gpus_per_replica
        if total_gpus_needed > available_gpus:
            raise ValueError(
                f"Engine stage requires up to {max_replicas} replicas * "
                f"{num_gpus_per_replica} GPUs ({total_gpus_needed} total), "
                f"but only {available_gpus} available."
            )

        map_batches_kwargs = {
            "zero_copy_batch": True,
            "compute": ray.data.ActorPoolStrategy(
                min_size=compute_range[0],
                max_size=compute_range[1],
            ),
            "max_concurrency": config.max_concurrent_batches,
            "accelerator_type": config.accelerator_type,
            "runtime_env": engine_stage_cfg.runtime_env or config.runtime_env,
            "num_gpus": engine_stage_cfg.num_gpus,
            "memory": engine_stage_cfg.memory,
        }
    else:
        map_batches_kwargs = {}

    return FoldingEngineStage(
        fn_constructor_kwargs=fn_constructor_kwargs,
        map_batches_kwargs=map_batches_kwargs,
        compute_by_rows=engine_stage_cfg.compute_by_rows,
        drop_keys=engine_stage_cfg.drop_keys,
    )


def _build_writer_stage(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> StatefulStage:
    writer_stage_cfg = resolve_stage_config(config.writer_stage, WriterStageConfig, processor_defaults)
    res_types = get_all_residue_types(config.model_source)
    res_type_mapping = dict(enumerate(res_types))
    atom_types = get_all_atom_types(config.model_source)
    atom_type_mapping = dict(enumerate(atom_types))
    mappings = {
        "res_type_mapping": res_type_mapping,
        "atom_type_mapping": atom_type_mapping,
    }
    return WriterStage(
        fn_constructor_kwargs={
            "mappings": mappings,
            "output_path": writer_stage_cfg.output_path,
            "format": writer_stage_cfg.format,
        },
        map_batches_kwargs=build_cpu_stage_map_kwargs(writer_stage_cfg),
        compute_by_rows=writer_stage_cfg.compute_by_rows,
        drop_keys=writer_stage_cfg.drop_keys,
    )


def _build_stages(config: EngineProcessorConfig, processor_defaults: dict[str, Any]) -> list[StatefulStage]:
    stages = []
    stages.append(_build_parser_stage(config, processor_defaults))
    stages.append(_build_tokenizer_stage(config, processor_defaults))
    stages.append(_build_feature_generator_stage(config, processor_defaults))
    stages.append(_build_folding_engine_stage(config, processor_defaults))
    stages.append(_build_writer_stage(config, processor_defaults))
    return stages


def _resolve_metadata(config: EngineProcessorConfig) -> None:
    """Auto-resolve metadata when not explicitly provided.

    Priority: explicit metadata dict > metadata_loader callable > factory default.
    """
    if config.metadata is not None:
        return
    if config.metadata_loader is not None:
        config.metadata = config.metadata_loader()
    else:
        resolved = load_metadata(config.model_source)
        if resolved:
            config.metadata = resolved


def _resolve_runtime_args(config: EngineProcessorConfig) -> None:
    """Merge registry defaults with user-supplied runtime_args.

    Registry defaults are used as the base; any keys explicitly provided
    by the user take precedence.
    """
    defaults = get_default_runtime_args(config.model_source)
    if defaults:
        merged = {**defaults, **config.runtime_args}
        config.runtime_args = merged


def build_processor(config: EngineProcessorConfig) -> _ProcessorBase:
    """Build a processor from the given config.

    Returns a :class:`Processor` (Ray-backed) when
    ``config.executor_backend == "ray"``, or a :class:`SerialProcessor`
    (in-process, no Ray) when ``config.executor_backend is None``.
    """
    _resolve_metadata(config)
    _resolve_runtime_args(config)

    processor_defaults = {
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "runtime_env": config.runtime_env,
        "model_source": config.model_source,
    }

    if config.executor_backend == "ray":
        import ray

        if not ray.is_initialized():
            ray.init(runtime_env=config.runtime_env, ignore_reinit_error=True)
        stages = _build_stages(config, processor_defaults)
        return Processor(config, stages)

    stages = _build_stages(config, processor_defaults)
    return SerialProcessor(config, stages)
