from typing import Any, Dict, List, Optional

import ray
from pydantic import Field

from tensorrt_bionemo.data.utils import (get_all_atom_types,
                                         get_all_residue_types)
from tensorrt_bionemo.pipeline.processor.base import Processor, ProcessorConfig
from tensorrt_bionemo.pipeline.processor.utils import \
    build_cpu_stage_map_kwargs
from tensorrt_bionemo.pipeline.stages import (FeatureGeneratorStage,
                                              FoldingEngineStage, ParserStage,
                                              TokenizerStage, WriterStage)
from tensorrt_bionemo.pipeline.stages.base import StatefulStage
from tensorrt_bionemo.pipeline.stages.configs import (
    FeatureGeneratorStageConfig, ParserStageConfig, TokenizerStageConfig,
    WriterStageConfig, resolve_stage_config)
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
    return FoldingEngineStage(
        fn_constructor_kwargs={
            "model": config.model_source,
            "engine_kwargs": config.engine_kwargs,
            "max_pending_requests": config.max_pending_requests,
            "should_continue_on_error": config.should_continue_on_error,
        },
        map_batches_kwargs=dict(
            zero_copy_batch=True,
            # The number of running replicas. This is a deprecated field, but
            # we need to set `max_tasks_in_flight_per_actor` through `compute`,
            # which initiates enough many overlapping UDF calls per actor, to
            # saturate `max_concurrency`.
            compute=ray.data.ActorPoolStrategy(
                min_size=config.get_concurrency(autoscaling_enabled=False)[0],
                max_size=config.get_concurrency(autoscaling_enabled=False)[1],
            ),
            # The number of running batches "per actor" in Ray Core level.
            # This is used to make sure we overlap batches to avoid the tail
            # latency of each batch.
            max_concurrency=config.max_concurrent_batches,
            accelerator_type=config.accelerator_type,
            runtime_env=config.runtime_env,
        ),
        compute_by_rows=True,
        drop_keys=None,
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
    ray.init(runtime_env=config.runtime_env, ignore_reinit_error=True)

    processor_defaults = {
        "batch_size": config.batch_size,
        "concurrency": config.concurrency,
        "runtime_env": config.runtime_env,
        "model_source": config.model_source,
    }

    stages = _build_stages(config, processor_defaults)

    return Processor(config, stages)
