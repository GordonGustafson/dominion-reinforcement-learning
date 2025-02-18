import torch.nn as nn
from torch.nn.parameter import Parameter

from typing import Iterator


class SumModules(nn.Module):
    def __init__(self, modules):
        super(SumModules, self).__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, inputs):
        sum_components = [module.forward(inputs) for module in self.modules_list]
        return sum(sum_components)