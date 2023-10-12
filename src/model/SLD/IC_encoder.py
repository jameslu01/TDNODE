import torch
import torch.nn as nn
from prodict import Prodict
import utils


class ICNet(nn.Module):
    def __init__(self, dims: Prodict, device=torch.device("cpu")) -> None:
        """The initial condition encoder module for TDNODE.

        Parameters
        ----------
        dims : Prodict
            A dictionary of dimension specifications to instantiate module components.
        device : str, optional
            The device on which to load the module, by default torch.device("cpu")
        """
        super(ICNet, self).__init__()
        self.input_dim = dims.INPUT_DIM
        self.output_dim = dims.OUTPUT_DIM
        self.hidden_dim = dims.HIDDEN_DIM
        self.rnn = nn.GRU(self.input_dim, self.hidden_dim).to(device)
        self.net = nn.Linear(self.hidden_dim, self.output_dim)
        utils.init_network_weights(self.net, std=0.001)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Forward function for TDNODE. Produces an encoding of the batch's initial condition.

        Parameters
        ----------
        data : torch.Tensor
            The baseline time series data for the batch. Shape: B x L_b x 2, where B is the batch
            size and L_b is the number of pre-treatment tumor size measurements of patient in the
            batch with the most pre-treatment tumor size measurements.

        Returns
        -------
        torch.Tensor
            A tensor containing the initial condition of the patients in the batch. Shape: B x c,
            where c is the dimensionality of the initial condition encoder.
        """
        (rnn_out, _) = self.rnn(data)
        return self.net(rnn_out[:, -1, :,])
