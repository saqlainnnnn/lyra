import torch
import torch.nn as nn


def initialize_modernbert(
    module: nn.Module,
    initializer_range: float = 0.02,
):
    """
    ModernBERT-style parameter initialization.

    Linear and embedding weights:
        truncated normal, std=initializer_range

    Biases:
        zero

    LayerNorm:
        weight=1
        bias=0
    """

    for name, parameter in module.named_parameters():

        if parameter.dim() >= 2:
            nn.init.trunc_normal_(
                parameter,
                mean=0.0,
                std=initializer_range,
                a=-2.0 * initializer_range,
                b=2.0 * initializer_range,
            )

        elif parameter.dim() == 1:
            if "norm" in name.lower():
                nn.init.ones_(parameter)
            else:
                nn.init.zeros_(parameter)