"""Shared utility functions for processor builders."""
from typing import Any, Dict, Optional, Tuple, Union

import ray

from tensorrt_bionemo.pipeline.stages.configs import _StageConfigBase


def get_value_or_fallback(value: Any, fallback: Any) -> Any:
    """Return value if not None, otherwise return fallback."""
    return value if value is not None else fallback


def extract_resource_kwargs(
    runtime_env: Optional[Dict[str, Any]],
    num_cpus: Optional[float],
    memory: Optional[float],
) -> Dict[str, Any]:
    """Extract non-None resource kwargs for map_batches."""
    kwargs = {}
    if runtime_env is not None:
        kwargs["runtime_env"] = runtime_env
    if num_cpus is not None:
        kwargs["num_cpus"] = num_cpus
    if memory is not None:
        kwargs["memory"] = memory
    return kwargs


def normalize_cpu_stage_concurrency(
        concurrency: Optional[Union[int, Tuple[int, int]]]) -> ray.data.ActorPoolStrategy:
    """
    Normalize concurrency specification to ActorPoolStrategy for CPU stages.
    
    Args:
        concurrency: Concurrency specification:
            - None: Returns ActorPoolStrategy(1, 1) - single actor, no autoscaling
            - int n: Returns ActorPoolStrategy(1, n) - autoscale from 1 to n actors
            - tuple (min, max): Returns ActorPoolStrategy(min, max) - custom autoscaling range
    
    Returns:
        ray.data.ActorPoolStrategy configured based on the input specification
    """
    if concurrency is None:
        return ray.data.ActorPoolStrategy(min_size=1, max_size=1)
    if isinstance(concurrency, int):
        return ray.data.ActorPoolStrategy(min_size=1, max_size=concurrency)
    return ray.data.ActorPoolStrategy(min_size=concurrency[0], max_size=concurrency[1])


def build_cpu_stage_map_kwargs(
    stage_cfg: _StageConfigBase, ) -> Dict[str, Any]:
    """Build map_batches_kwargs for CPU stages."""
    concurrency = normalize_cpu_stage_concurrency(stage_cfg.compute)
    return dict(
        zero_copy_batch=True,
        compute=concurrency,
        batch_size=stage_cfg.batch_size,
        **extract_resource_kwargs(
            stage_cfg.runtime_env,
            stage_cfg.num_cpus,
            stage_cfg.memory,
        ),
    )


def get_available_gpu_count() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 1
