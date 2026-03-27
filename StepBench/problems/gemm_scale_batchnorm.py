# Origin: KernelBench/level2/33_Gemm_Scale_BatchNorm.py
# Notes: Best available GEMM+Norm match. BatchNorm is the closest normalization to RMSNorm in KernelBench; no GEMM+RMSNorm file exists.
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GEMM (general matrix multiplication), applies scaling,
    and then batch normalization.
    """
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        x = x * self.scale
        x = self.bn(x)
        return x

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["in_features"])]

def get_init_inputs(dims):
    return [dims["in_features"], dims["out_features"], (dims["out_features"],)]

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
