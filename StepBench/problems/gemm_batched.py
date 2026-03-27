# Origin: /home/ubuntu/patrick/AbstractOpt/KernelBench/KernelBench/level1/3_Batched_matrix_multiplication.py
# Notes: Direct copy, identical code and dimensions
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    batch, M, K, N = dims["batch"], dims["M"], dims["K"], dims["N"]
    return [torch.randn(batch, M, K), torch.randn(batch, K, N)]

def get_init_inputs(dims):
    return []

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)