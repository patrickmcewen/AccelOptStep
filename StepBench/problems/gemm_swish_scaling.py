# Origin: KernelBench/level2/59_Matmul_Swish_Scaling.py
# Notes: Excellent match. Swish (x * sigmoid(x)) is identical to SiLU. Has an extra scaling factor but core GEMM+SiLU pattern matches exactly.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features, bias=False)
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.matmul(x)
        x = x * torch.sigmoid(x)  # Swish activation
        x = x * self.scaling_factor
        return x

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["in_features"])]

def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], dims["scaling_factor"]]

def compute_gold(dims):
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
