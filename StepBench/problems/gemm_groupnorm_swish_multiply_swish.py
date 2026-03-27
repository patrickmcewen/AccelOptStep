# Origin: KernelBench/level2/88_Gemm_GroupNorm_Swish_Multiply_Swish.py
# Notes: Closest FFN-like match in KernelBench; has Gemm + Norm + Swish + Multiply pattern resembling SwiGLU gating, though uses GroupNorm instead of RMSNorm and double-Swish instead of SwiGLU
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))

    def forward(self, x):
        # (batch_size, in_features) -> (batch_size, out_features)
        x = self.gemm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = self.group_norm(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * self.multiply_weight
        # (batch_size, out_features) -> (batch_size, out_features)
        x = x * torch.sigmoid(x)
        return x

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["in_features"])]

def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], dims["num_groups"], (dims["out_features"],)]

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
