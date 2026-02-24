from typing import Any, Dict, List, Optional

import ray
from pydantic import Field

from tensorrt_bionemo.data.utils import (get_all_atom_types,
                                         get_all_residue_types)
from tensorrt_bionemo.pipeline.processor.base import Processor, ProcessorConfig
from tensorrt_bionemo.pipeline.processor.utils import (
    build_cpu_stage_map_kwargs, get_available_gpu_count)
from tensorrt_bionemo.pipeline.stages import (FeatureGeneratorStage,
                                              FoldingEngineStage, ParserStage,
                                              TokenizerStage, WriterStage)
from tensorrt_bionemo.pipeline.stages.base import StatefulStage
from tensorrt_bionemo.pipeline.stages.configs import (
    EngineStageConfig, FeatureGeneratorStageConfig, ParallelismMode,
    ParserStageConfig, TokenizerStageConfig, WriterStageConfig,
    resolve_stage_config)
from tensorrt_bionemo.registry import (get_feature_factory, get_model_class,
                                       get_tokenizer)


class EngineProcessorConfig(ProcessorConfig):
    model_source: str = Field(
        default=None,
        description="The model source to use for the processor.",
    )
    engine_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=
        "The kwargs to pass to the tensorrt-bionemo engine. {'config': Optional[BaseConfig], 'accelerated_configs': Optional[Dict[str, AcceleratedConfig]]}"
    )
    parser_stage: Any = Field(
        default=True,
        description="Parser stage config (bool | dict | ParserStageConfig).",
    )
    tokenizer_stage: Any = Field(
        default=True,
        description=
        "Tokenizer stage config (bool | dict | TokenizerStageConfig).",
    )
    feature_generator_stage: Any = Field(
        default=True,
        description=
        "Feature generator stage config (bool | dict | FeatureGeneratorStageConfig).",
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
    runtime_env: Optional[Dict[str, Any]] = Field(
        default=None,
        description=
        "The runtime environment to use for the offline processing.",
    )

    max_pending_requests: Optional[int] = Field(
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
    def create_replica_mode_config(
            cls,
            model_source: str,
            output_dir: str,
            output_format: str = "pdb",
            tokenizer_stage_num_cpus: int = 2,
            feature_generator_stage_num_cpus: int = 4,
            engine_stage_num_cpus: int = 4) -> "EngineProcessorConfig":
        """Build a processor config for replica mode (one engine per GPU) using all available GPUs."""
        num_gpus = get_available_gpu_count()
        return cls(model_source=model_source,
                   parser_stage=ParserStageConfig(compute=num_gpus),
                   tokenizer_stage=TokenizerStageConfig(
                       compute=num_gpus, num_cpus=tokenizer_stage_num_cpus),
                   feature_generator_stage=FeatureGeneratorStageConfig(
                       num_cpus=feature_generator_stage_num_cpus,
                       compute=num_gpus),
                   writer_stage=WriterStageConfig(compute=num_gpus,
                                                  output_path=output_dir,
                                                  format=output_format),
                   engine_stage=EngineStageConfig(
                       compute=num_gpus, num_cpus=engine_stage_num_cpus))

    def get_model_pretrained_config(self):
        model_class = get_model_class(self.model_source)
        result = model_class.get_pretrained_config(self.model_source)
        if "config" in self.engine_kwargs:
            # Override the default pretrained config
            result = self.engine_kwargs["config"]
        return result


def _build_parser_stage(config: EngineProcessorConfig,
                        processor_defaults: Dict[str, Any]) -> StatefulStage:
    parser_stage_cfg = resolve_stage_config(config.parser_stage,
                                            ParserStageConfig,
                                            processor_defaults)
    return ParserStage(
        fn_constructor_kwargs={},
        map_batches_kwargs=build_cpu_stage_map_kwargs(parser_stage_cfg),
        compute_by_rows=parser_stage_cfg.compute_by_rows,
        drop_keys=parser_stage_cfg.drop_keys,
    )


def _build_tokenizer_stage(
        config: EngineProcessorConfig,
        processor_defaults: Dict[str, Any]) -> StatefulStage:
    model_pretrained_config = config.get_model_pretrained_config()
    tokenizer_stage_cfg = resolve_stage_config(config.tokenizer_stage,
                                               TokenizerStageConfig,
                                               processor_defaults)
    tokenizer = get_tokenizer(config.model_source)
    context_generators = {}
    for k, generator_spec in tokenizer.context_generator_specs.items():
        context_generators[k] = generator_spec.generator(
            config=model_pretrained_config)
        context_generators[k].required_kwargs = generator_spec.required_kwargs

    transform_funcs = []
    for transform_spec in tokenizer.transform_specs:
        transform_cls = transform_spec.transform
        transform_funcs.append(
            transform_cls(config=model_pretrained_config,
                          **transform_spec.kwargs))

    return TokenizerStage(
        fn_constructor_kwargs={
            "context_generators": context_generators,
            "context_merger_func": tokenizer.context_merger_func,
            "transform_funcs": transform_funcs,
        },
        map_batches_kwargs=build_cpu_stage_map_kwargs(tokenizer_stage_cfg),
        compute_by_rows=tokenizer_stage_cfg.compute_by_rows,
        drop_keys=tokenizer_stage_cfg.drop_keys,
    )


def _build_feature_generator_stage(
        config: EngineProcessorConfig,
        processor_defaults: Dict[str, Any]) -> StatefulStage:
    model_pretrained_config = config.get_model_pretrained_config()
    feature_generator_stage_cfg = resolve_stage_config(
        config.feature_generator_stage, FeatureGeneratorStageConfig,
        processor_defaults)

    feature_factory = get_feature_factory(config.model_source)
    feature_generators = []
    feature_collators = []
    # Initialize feature generators and collators

    for generator_spec in feature_factory.feature_generator_specs:
        generator = generator_spec.functor(config=model_pretrained_config,
                                           **generator_spec.kwargs)
        generator.name = generator_spec.name
        feature_generators.append(generator)
    for collator_spec in feature_factory.feature_collator_specs:
        collator = collator_spec.functor(config=model_pretrained_config,
                                         **collator_spec.kwargs)
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
        map_batches_kwargs=build_cpu_stage_map_kwargs(
            feature_generator_stage_cfg),
        compute_by_rows=feature_generator_stage_cfg.compute_by_rows,
        # Drop parsed to save memory
        drop_keys=feature_generator_stage_cfg.drop_keys
        if feature_generator_stage_cfg.drop_keys is not None else ["parsed"],
    )


def _build_folding_engine_stage(
        config: EngineProcessorConfig,
        processor_defaults: Dict[str, Any]) -> StatefulStage:
    engine_stage_cfg = resolve_stage_config(config.engine_stage,
                                            EngineStageConfig,
                                            processor_defaults)

    if engine_stage_cfg.parallelism_mode == ParallelismMode.DISTRIBUTED:
        # Extension point: implement multi-GPU per engine (Tensor Parallel / Context Parallel).
        # E.g. set up process groups, pass Mapping into engine_kwargs, use placement groups or
        # multi-process runner per logical replica.
        raise NotImplementedError(
            "DISTRIBUTED mode (Tensor Parallel / Context Parallel) is not implemented yet. "
            "Extend _build_folding_engine_stage and the engine stage for multi-GPU per replica."
        )

    available_gpus = get_available_gpu_count()
    num_gpus_per_replica = engine_stage_cfg.num_gpus

    # Replica count: use explicit compute if set, otherwise one replica per available GPU.
    compute = engine_stage_cfg.compute
    if compute is None:
        # Auto: calculate max replicas based on available GPUs and GPU requirement per replica.
        max_replicas = max(1, int(available_gpus / num_gpus_per_replica))
        compute = max_replicas
    if isinstance(compute, int):
        compute_range = (compute, compute)
    else:
        compute_range = compute

    # Validate total GPU demand does not exceed available GPUs.
    max_replicas = compute_range[1]
    total_gpus_needed = max_replicas * num_gpus_per_replica
    if total_gpus_needed > available_gpus:
        raise ValueError(
            f"Engine stage requires up to {max_replicas} replicas × {num_gpus_per_replica} GPU(s) = {total_gpus_needed} GPUs, "
            f"but only {available_gpus} are available. Set compute <= {int(available_gpus // num_gpus_per_replica)} or "
            f"leave compute unset to use all available GPUs.")

    return FoldingEngineStage(
        fn_constructor_kwargs={
            "model": config.model_source,
            "engine_kwargs": config.engine_kwargs,
            "max_pending_requests": config.max_pending_requests,
            "should_continue_on_error": config.should_continue_on_error,
            "parallelism_mode": engine_stage_cfg.parallelism_mode,
        },
        map_batches_kwargs=dict(
            zero_copy_batch=True,
            compute=ray.data.ActorPoolStrategy(
                min_size=compute_range[0],
                max_size=compute_range[1],
            ),
            max_concurrency=config.max_concurrent_batches,
            accelerator_type=config.accelerator_type,
            runtime_env=engine_stage_cfg.runtime_env or config.runtime_env,
            num_gpus=engine_stage_cfg.num_gpus,
        ),
        compute_by_rows=engine_stage_cfg.compute_by_rows,
        drop_keys=engine_stage_cfg.drop_keys,
    )


def _build_writer_stage(config: EngineProcessorConfig,
                        processor_defaults: Dict[str, Any]) -> StatefulStage:
    writer_stage_cfg = resolve_stage_config(config.writer_stage,
                                            WriterStageConfig,
                                            processor_defaults)
    res_types = get_all_residue_types(config.model_source)
    res_type_mapping = {i: res_type for i, res_type in enumerate(res_types)}
    atom_types = get_all_atom_types(config.model_source)
    atom_type_mapping = {
        i: atom_type
        for i, atom_type in enumerate(atom_types)
    }
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


def _build_stages(config: EngineProcessorConfig,
                  processor_defaults: Dict[str, Any]) -> List[StatefulStage]:
    stages = []
    stages.append(_build_parser_stage(config, processor_defaults))
    stages.append(_build_tokenizer_stage(config, processor_defaults))
    stages.append(_build_feature_generator_stage(config, processor_defaults))
    stages.append(_build_folding_engine_stage(config, processor_defaults))
    stages.append(_build_writer_stage(config, processor_defaults))
    return stages


def build_processor(config: EngineProcessorConfig) -> Processor:
    if not ray.is_initialized():
        ray.init(runtime_env=config.runtime_env, ignore_reinit_error=True)

    processor_defaults = {
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "runtime_env": config.runtime_env,
        "model_source": config.model_source,
    }

    stages = _build_stages(config, processor_defaults)

    return Processor(config, stages)
