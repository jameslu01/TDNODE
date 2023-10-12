import torch
import torch.nn as nn
from prodict import Prodict
import utils


class Reducer(nn.Module):
    def __init__(self, dims: Prodict, device=torch.device("cpu")):
        super(Reducer, self).__init__()
        self.input_dim = dims.INPUT_DIM
        self.output_dim = dims.OUTPUT_DIM
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.SELU(),
            nn.Linear(self.input_dim, self.output_dim),
            nn.SELU(),
        )
        self.device = device
        utils.init_network_weights(self.net, std=0.001)

    def forward(self, data):
        net_out = self.net(data)
        return net_out.permute(1, 0, 2)
