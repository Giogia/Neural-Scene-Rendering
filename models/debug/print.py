import torch.nn as nn
# from models.debug.print import PrintLayer

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.size())
        return x