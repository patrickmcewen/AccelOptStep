# Origin: KernelBench/level3/2_ShallowWideMLP.py
# Notes: No MoE files exist in KernelBench. This shallow wide MLP is the nearest match to the simplified MoE (single-linear experts) — both perform multiple wide GEMM operations. Wider layers parallel the expert projection pattern.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            current_input_size = hidden_size

        layers.append(nn.Linear(current_input_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        return self.network(x)

SEED = 42

def get_inputs(dims):
    torch.manual_seed(SEED)
    return [torch.randn(dims["batch_size"], dims["input_size"])]

def get_init_inputs(dims):
    return [dims["input_size"], dims["hidden_layer_sizes"], dims["output_size"]]

def compute_gold(dims):
    torch.manual_seed(SEED)
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
