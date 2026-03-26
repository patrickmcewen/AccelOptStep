import torch
import torch.nn as nn

batch = 4
seq_len = 2048
d_model = 2048
d_state = 16


class Model(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = d_state
        # Input-dependent discretization parameters
        self.proj_delta = nn.Linear(d_model, d_model, bias=False)
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        # Diagonal state matrix (log-space for stability)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        b, l, d = x.shape

        delta = torch.nn.functional.softplus(self.proj_delta(x))  # (b, l, d)
        B = self.proj_B(x)  # (b, l, d_state)
        C = self.proj_C(x)  # (b, l, d_state)
        A = -torch.exp(self.A_log)  # (d, d_state)

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (b, l, d, d_state)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (b, l, d, d_state)

        # Linear recurrence scan
        h = torch.zeros(b, d, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(l):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (b, d)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (b, l, d)
        return y + x * self.D


def get_inputs():
    return [torch.randn(batch, seq_len, d_model)]


def get_init_inputs():
    return [d_model, d_state]
