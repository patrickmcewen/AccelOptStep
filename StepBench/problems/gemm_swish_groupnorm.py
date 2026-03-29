# Origin: /home/ubuntu/patrick/AbstractOpt/KernelBench/KernelBench/level2/37_Matmul_Swish_Sum_GroupNorm.py
# Notes: Direct copy, identical code
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.matmul(x)
        x = torch.sigmoid(x) * x  # Swish activation
        x = x + self.bias
        x = self.group_norm(x)
        return x

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["in_features"])]

def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], dims["num_groups"], (dims["out_features"],)]

def compute_gold(dims):
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)