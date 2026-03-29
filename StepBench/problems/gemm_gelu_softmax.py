# Origin: /home/ubuntu/patrick/AbstractOpt/KernelBench/KernelBench/level2/99_Matmul_GELU_Softmax.py
# Notes: Direct copy, identical code and dimensions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = torch.nn.functional.gelu(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["in_features"])]

def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"]]

def compute_gold(dims):
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)