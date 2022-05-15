import torch.nn as nn


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x
