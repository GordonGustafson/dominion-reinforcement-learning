import torch
import torch.nn as nn

class ConstantModule(nn.Module):
    def __init__(self, value):
        super(ConstantModule, self).__init__()
        self.constant = nn.Parameter(torch.tensor(value), requires_grad=False)

    def forward(self, *args, **kwargs):
        return self.constant