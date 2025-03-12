from dataclasses import dataclass, field
from typing import Generic, Optional, TypeVar

import transformers
from tensorrt_llm.mapping import Mapping

TConfig = TypeVar("TConfig", bound=transformers.PretrainedConfig)


@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    pretrained_config: Optional[TConfig] = None
    mapping: Mapping = field(default_factory=Mapping)
    skip_create_weights: bool = False

    attn_backend: str = 'VANILLA'

    @classmethod
    def from_pretrained(cls,
                        checkpoint_dir: str,
                        trust_remote_code=False,
                        **kwargs):
        pretrained_config = transformers.AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=trust_remote_code,
        )

        return cls(pretrained_config=pretrained_config, **kwargs)
