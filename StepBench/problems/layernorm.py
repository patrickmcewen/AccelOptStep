# Origin: /home/ubuntu/patrick/AbstractOpt/KernelBench/KernelBench/level1/40_LayerNorm.py
# Notes: Direct copy, identical code
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["features"], dims["dim1"], dims["dim2"])]

def get_init_inputs(dims):
    return [(dims["features"], dims["dim1"], dims["dim2"])]

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)