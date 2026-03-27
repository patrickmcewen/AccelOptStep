# Origin: KernelBench/level3/1_MLP.py
# Notes: No MoE files exist in KernelBench. This multi-layer MLP (Linear+ReLU stack) is the nearest match to MoE's SwiGLU expert architecture — both involve multiple sequential GEMMs with nonlinear activations.
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()

        layers = []
        current_input_size = input_size

        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size

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
    return [dims["input_size"], dims["layer_sizes"], dims["output_size"]]

def compute_gold(dims):
    model = Model(*get_init_inputs(dims))
    inputs = get_inputs(dims)
    return model(*inputs)
