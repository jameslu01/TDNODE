import torch
import torch.nn as nn
from prodict import Prodict


class ODEFunc(nn.Module):
    def __init__(self, dims: Prodict, device: str = torch.device("cpu")):
        """The Neural ODE decoder Module of TDNODE. Neural network function that considers as input
        a tumor state and p-dimensional parameter encoding. Produces the next tumor state at the
        next available time point.

        Parameters
        ----------
        dims : Prodict
            A dictionary of the dimensionalities of the component modules to be used during
            instantiation.
        device : str, optional
            The device on which to load the module, by default torch.device("cpu").
        """
        super(ODEFunc, self).__init__()

        self.input_dim = dims.INPUT_DIM
        self.output_dim = dims.OUTPUT_DIM
        self.hidden_dim = dims.HIDDEN_DIM
        self.latent_dim = dims.LATENT_DIM
        self.input_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.device = device
        self.block2 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.block3 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.block4 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.block5 = nn.Sequential(nn.SELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.block6 = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.end_block = nn.Sequential(nn.SELU(), nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, t: torch.Tensor, data: torch.Tensor):
        """_summary_

        Parameters
        ----------
        t : torch.Tensor
            A tensor of time measurements to be used during the solve process. Shape: L_T x 1, where
            L_T is the number of distinct time points in the batch.
        data : torch.Tensor
            The concatenated batch of initial condition and parameter encodings (only at the first
            call). Shape: B x (c + p), where B is the batch size, c is the dimensionality of the
            initial condition encoding, and p is the dimensionality of the parameter encoding.

        Returns
        -------
        torch.Tensor
            A tensor of c-dimensional predictions. Shape: B x L_T x c, where c is the
            dimensionality of the initial condition encoding.
        """
        x1 = self.input_net(data)
        x2 = self.block2(x1.clone())
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        x4 += x3
        x5 = self.block5(x4)
        x5 += x2
        x6 = self.block6(x5)
        x7 = self.end_block(x1)
        out = x6 + x7
        returned = torch.cat(
            [out, torch.zeros(out.shape[0], self.latent_dim, device=self.device)], dim=-1
        )
        return returned
