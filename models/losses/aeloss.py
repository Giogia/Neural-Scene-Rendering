import torch
import torch.nn as nn


class AutoencoderLoss(nn.Module):

    def __init__(self):
        super(AutoencoderLoss, self).__init__()

    def forward(self, output, lossweights):
        return sum([lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                    for k, v in output["losses"].items()])
