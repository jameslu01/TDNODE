from model.SLD.encoder import Encoder
from model.SLD.IC_encoder import ICNet
from model.SLD.reducer import Reducer
from model.SLD.ode_func import ODEFunc
from prodict import Prodict
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class TDNODE(nn.Module):
    def __init__(
        self,
        dims: Prodict,
        tol: float,
        use_cached_encoding: bool = False,
        device: str = torch.device("cpu"),
        **kwargs
    ) -> None:
        """Root Module for TDNODE. Encompasses four separate components: IC encoder, parameter
        encoder, neural ODE decoder, and reducer.

        Parameters
        ----------
        dims : Prodict
            A dictionary specifying the dimensions of each module to instantiate.
        tol : float
            The numerical tolerance, a hyperparameter to tune the tolerated numerical error in
            `torchdiffeq`'s `odesolve` function.
        use_cached_encoding : bool, optional
            Indicator for whether to refer to the class attribute `parameter_encoding` instead
            of generating a new encoding. Used when the outcome of a manually set encoding is
            desired, by default False.
        device : string, optional
            The device on which to instantiate the model, by default torch.device("cpu")
        """
        super(TDNODE, self).__init__()

        self.device = device
        self.IC_net = ICNet(dims=dims.INITIAL_CONDITION_ENCODER, device=self.device)
        self.encoder = Encoder(dims=dims.LATENT_ENCODER, device=self.device)

        self.ode_func = ODEFunc(dims=dims.ODE_SOLVER, device=self.device)
        self.classifier = Reducer(dims=dims.REDUCER, device=self.device)
        self.use_cached_encoding = use_cached_encoding
        self.encoding = torch.Tensor([])
        self.IC_dim = dims.INITIAL_CONDITION_ENCODER.OUTPUT_DIM
        self.tol = tol

    def forward(
        self,
        batch: torch.Tensor,
        baseline: torch.Tensor,
        times: torch.Tensor,
        pt_time_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Main forward function of TDNODE. Generates initial condition encoding, parameter encoding
        (if `self.use_cached_encoding` is set to `False`), N-dimensional latent representation of
        tumor dynamics, and returns a series of tumor dynamic predictions in the data space.

        Parameters
        ----------
        batch : torch.Tensor
            A batch of paritioned time series data. Shape: B x (L_pmax - 1) x 4, where B is the
            batch size and L_pmax is the number of observed post-treatment SLD measurements in the
            patient in the batch with the most post-treatment observations.
        baseline : torch.Tensor
            A batch of baseline time series data. Shape: B x L_bmax x 2, where L_b is the number of
            pre-treatment tumor size measurements in the patient in the batch with the most
            pre-treatment observations.
        times : torch.Tensor
            A single-dimension tensor of time values. Shape: L_T, where L_T is the number of
            distinct observations in the current batch.
        pt_time_scale : torch.Tensor
            A stacked tensor of scaling factors. Shape: B x 1.

        Returns
        -------
        torch.Tensor
            The series of predictions for the current batch. Shape: B x L_T.
        """
        self.IC_encoding = self.IC_net(baseline)
        if not self.use_cached_encoding:
            self.encoding = self.encoder(batch)
            self.encoding = self.encoding * pt_time_scale
        encoding = torch.cat([self.IC_encoding, self.encoding], dim=-1)
        sol = odeint(self.ode_func, encoding, times, rtol=self.tol, atol=self.tol)
        preds = self.classifier(sol[:, :, : self.IC_dim,])
        return preds.reshape(preds.shape[:-1])

    def get_encoding(self) -> torch.Tensor:
        """Fetches parameter encoding attribute of TDNODE.

        Returns
        -------
        torch.Tensor
            The parameter encoding: Shape: B x p, where p is the dimenstionality of the latent
            encoding.
        """
        return self.encoding

    def set_cached_encoding(self, encoding: torch.Tensor) -> None:
        """Sets the singular argument as the parameter encoding. Used in conjunction with
        `use_cached_encoding = True` to determine effects of encoding perturbations.

        Parameters
        ----------
        encoding : torch.Tensor
            The encoding to set as the parameter encoding for TDNODE during forward propagation.
            Shape: B x p.
        """
        self.encoding = encoding
