from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax


class GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn: torch.nn.Module, nn: Optional[torch.nn.Module] = None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(
        self, x: Tensor, batch: Optional[Tensor] = None, size: Optional[int] = None
    ) -> Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = gate * x

        return out, gate

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, " f"nn={self.nn})"
