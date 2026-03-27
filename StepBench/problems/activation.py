# Origin: Consolidation of KernelBench level1/19 (ReLU), level1/21 (Sigmoid),
#         level1/23 (Softmax), level1/25 (Swish), level1/26 (GELU).
#         Activation selected via 'fn' parameter from bench_config.yaml.
import torch
import torch.nn as nn

SEED = 42

ACTIVATIONS = {
    "relu": torch.relu,
    "gelu": lambda x: torch.nn.functional.gelu(x),
    "sigmoid": torch.sigmoid,
    "swish": lambda x: x * torch.sigmoid(x),
    "softmax": lambda x: torch.softmax(x, dim=1),
}


class Model(nn.Module):
    def __init__(self, fn_name):
        super().__init__()
        self.fn = ACTIVATIONS[fn_name]

    def forward(self, x):
        return self.fn(x)


def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["dim"])]


def get_init_inputs(dims):
    return [dims["fn"]]


def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
