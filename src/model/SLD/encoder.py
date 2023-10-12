import torch
import torch.nn as nn
from prodict import Prodict
from utils import init_network_weights

class Encoder(nn.Module):
    def __init__(self, dims: Prodict, device: str = torch.device("cpu")) -> None:
        """Parameter encoder module for TDNODE. Given a paritioned time series of post-treatment
        measurements up to an arbitrarily defined observation window, produces a p-dimensional
        encoding representative of the observed dynamics in the time series.

        Parameters
        ----------
        dims : Prodict
            A dictionary specifying the dimensionalities of the component modules to activate.
        device : str, optional
            The device on which to load module components, by default torch.device("cpu")
        """
        super(Encoder, self).__init__()

        self.input_dim = dims.INPUT_DIM
        self.device = device

        self.preprocess_net = PreprocessNet(dims=dims.ENCODER_PREPROCESSOR, device=self.device)
        self.postprocess_net = PostprocessNet(dims=dims.ENCODER_POSTPROCESSOR, device=self.device)

        self.LSTM_output_dim = dims.LSTM_OUTPUT_DIM
        self.LSTM = nn.LSTM(dims.ENCODER_PREPROCESSOR.OUTPUT_DIM, self.LSTM_output_dim)
        self.key = nn.Linear(self.input_dim, self.LSTM_output_dim)
        self.value = nn.Linear(self.input_dim, self.LSTM_output_dim)
        self.att = nn.MultiheadAttention(embed_dim=self.LSTM_output_dim, num_heads=1)
        init_network_weights(self.key, std=0.001)
        init_network_weights(self.value, std=0.001)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward propagation operation for the parameter encoder module.

        Parameters
        ----------
        input : torch.Tensor
            A batch of post-treamtnet time series measurements. Shape: B x (L_pmax - 1) x 4, where B
            is the batch size and L_pmax is the maximum number of post-treatment tumor size
            measurements up to a predetermined observation window of patients in a batch.

        Returns
        -------
        torch.Tensor
            A batch of parameter encodings representing the tumor dynamics of each patient. Shape:
            B x p, where p is the predetermined dimensionality of the parameter encoding.
        """
        key = self.key(input)
        value = self.value(input)
        preprocess = self.preprocess_net(input)
        (query, _) = self.LSTM(preprocess)
        last1 = query[:, -1, :,]
        (att, _) = self.att(query, key, value)
        last2 = att[:, -1, :,]
        last = torch.cat([last1, last2], dim=-1)
        return self.postprocess_net(last)


class PreprocessNet(nn.Module):
    def __init__(self, dims: Prodict, device: str = torch.device("cpu")):
        """A helper module for the `Encoder` module of TDNODE.

        Parameters
        ----------
        dims : Prodict
            A dictionary containing the dimensionalities of component modules to use during
            instantiation.
        """
        super(PreprocessNet, self).__init__()

        self.input_dim = dims.INPUT_DIM
        self.hidden_dim = dims.HIDDEN_DIM
        self.output_dim = dims.OUTPUT_DIM

        self.starting_net = nn.Linear(self.input_dim, self.hidden_dim)
        self.lower_net = nn.Sequential(nn.SELU(), nn.Linear(self.hidden_dim, self.output_dim))

        self.upper_net = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        init_network_weights(self.starting_net, std=0.001)
        init_network_weights(self.lower_net, std=0.001)
        init_network_weights(self.upper_net, std=0.001)

    def forward(self, data: torch.Tensor):
        """The forward propagation operation of the `PreProcessNet` module.

        Parameters
        ----------
        data : torch.Tensor
            A batch of post-treamtnet time series measurements. Shape: B x (L_pmax - 1) x 4, where B
            is the batch size and L_pmax is the maximum number of post-treatment tumor size
            measurements up to a predetermined observation window of patients in a batch.


        Returns
        -------
        torch.Tensor
            An intermediate output that is used as input to an LSTM module, producing the query to
            be used in the attention module.
        """
        x = self.starting_net(data)
        x2 = self.upper_net(x)
        x3 = self.lower_net(x) + x2
        return data + x3


class PostprocessNet(nn.Module):
    def __init__(self, dims: Prodict, device: str = torch.device("cpu")) -> None:
        """A helper module for the TDNODE

        Parameters
        ----------
        dims : Prodict
            A dictionary describing the dimensionalites of the component modules to set during
            instantiation
        """
        super(PostprocessNet, self).__init__()

        self.input_dim = dims.INPUT_DIM
        self.hidden_dim = dims.HIDDEN_DIM
        self.output_dim = dims.OUTPUT_DIM
        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.block = nn.Sequential(
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        init_network_weights(self.lin1, std=0.001)
        init_network_weights(self.block, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward propagation function of `PostProcessNet`. Produces the final set of parameter
        encodings for the current batch.

        Parameters
        ----------
        x : torch.Tensor
            An intermediate data structure that is the concatenated output of the LSTM module and
            attention module.

        Returns
        -------
        torch.Tensor
            A batch of parameter encodings. Shape: B x p, where B is the batch size and p is the
            predetermined dimensionality of the parameter encoding.
        """
        layer_1_out = self.lin1(x)
        net_out = self.block(layer_1_out)
        return layer_1_out + net_out
