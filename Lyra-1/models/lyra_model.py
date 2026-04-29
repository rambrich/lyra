"""Core Lyra model definition.

Builds on top of a pretrained vision-language backbone and adds
Lyra-specific adaptations for multimodal understanding.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class LyraConfig:
    """Configuration class for the Lyra model."""

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        vision_encoder_name: str = "openai/clip-vit-large-patch14-336",
        hidden_size: int = 4096,
        vision_hidden_size: int = 1024,
        projection_dim: int = 4096,
        num_vision_tokens: int = 576,
        use_flash_attention: bool = True,
        freeze_vision_encoder: bool = True,
        freeze_base_model: bool = False,
        **kwargs,
    ):
        self.base_model_name = base_model_name
        self.vision_encoder_name = vision_encoder_name
        self.hidden_size = hidden_size
        self.vision_hidden_size = vision_hidden_size
        self.projection_dim = projection_dim
        self.num_vision_tokens = num_vision_tokens
        self.use_flash_attention = use_flash_attention
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_base_model = freeze_base_model

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LyraConfig":
        """Instantiate LyraConfig from a Python dictionary."""
        return cls(**config_dict)


class VisionProjection(nn.Module):
    """Projects vision encoder outputs into the language model's embedding space."""

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden_size, projection_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        x = self.linear1(vision_features)
        x = self.act(x)
        x = self.linear2(x)
        return x


class LyraModel(nn.Module):
    """Lyra multimodal language model.

    Combines a frozen (or fine-tuned) vision encoder with a causal language
    model via a learned projection layer.
    """

    def __init__(self, config: LyraConfig):
        super().__init__()
        self.config = config

        # Vision projection layer
        self.vision_projection = VisionProjection(
            vision_hidden_size=config.vision_hidden_size,
            projection_dim=config.projection_dim,
        )

        logger.info("LyraModel initialised (vision projection only; load backbone separately)")

    @classmethod
    def from_config(cls, config: Union[LyraConfig, dict]) -> "LyraModel":
        """Build a LyraModel from a LyraConfig or a plain dict."""
        if isinstance(config, dict):
            config = LyraConfig.from_dict(config)
        return cls(config)

    def get_vision_projection(self) -> VisionProjection:
        """Return the vision projection module (useful for targeted optimisers)."""
        return self.vision_projection

    def project_vision_features(
        self, vision_features: torch.Tensor
    ) -> torch.Tensor:
        """Project raw vision encoder outputs to the LM embedding dimension.

        Args:
            vision_features: Float tensor of shape (B, N, vision_hidden_size).

        Returns:
            Projected features of shape (B, N, projection_dim).
        """
        return self.vision_projection(vision_features)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        vision_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """Forward pass placeholder — full implementation couples the backbone.

        Subclasses or training scripts should attach the backbone and call
        its forward method after embedding injection.
        """
        if vision_features is not None:
            projected = self.project_vision_features(vision_features)
            return projected
        raise NotImplementedError(
            "Full forward pass requires a backbone. Use LyraModel.from_config() "
            "inside a training script that attaches the LLM backbone."
        )
