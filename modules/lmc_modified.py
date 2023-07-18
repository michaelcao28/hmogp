
from typing import Optional, Union
import torch
import gpytorch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints

from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.utils.interpolation import left_interp
from torch import LongTensor, Tensor

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy

def _select_lmc_coefficients(lmc_coefficients: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """
    Given a list of indices for ... x N datapoints,
      select the row from lmc_coefficient that corresponds to each datapoint
    lmc_coefficients: torch.Tensor ... x num_latents x ... x num_tasks
    indices: torch.Tesnor ... x N
    """
    
    batch_shape = torch.broadcast_shapes(lmc_coefficients.shape[:-1], indices.shape[:-1])

    # We will use the left_interp helper to do the indexing
    lmc_coefficients = lmc_coefficients.expand(*batch_shape, lmc_coefficients.shape[-1])[..., None]
    indices = indices.expand(*batch_shape, indices.shape[-1])[..., None]
    res = left_interp(
        indices,
        torch.ones(indices.shape, dtype=torch.long, device=indices.device),
        lmc_coefficients,
    ).squeeze(-1)
    return res


class LMCVariationalStrategy(_VariationalStrategy):

    def __init__(
        self,
        base_variational_strategy: _VariationalStrategy,
        num_tasks: int,
        num_latents: int = 1,
        latent_dim: int = -1,
        jitter_val: Optional[float] = None,
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        batch_shape = self.base_variational_strategy._variational_distribution.batch_shape
        
        # Added
        self.latent_dist = None

        # Check if no functions
        if latent_dim >= 0:
            raise RuntimeError(f"latent_dim must be a negative indexed batch dimension: got {latent_dim}.")
        if not (batch_shape[latent_dim] == num_latents or batch_shape[latent_dim] == 1):
            raise RuntimeError(
                f"Mismatch in num_latents: got a variational distribution of batch shape {batch_shape}, "
                f"expected the function dim {latent_dim} to be {num_latents}."
            )
        self.num_latents = num_latents
        self.latent_dim = latent_dim

        # Make the batch_shape
        self.batch_shape = list(batch_shape)
        del self.batch_shape[self.latent_dim]
        self.batch_shape = torch.Size(self.batch_shape)

        # LCM coefficients
        lmc_coefficients = torch.randn(*batch_shape, self.num_tasks)
        self.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))

        if jitter_val is None:
            self.jitter_val = settings.variational_cholesky_jitter.value(
                self.base_variational_strategy.inducing_points.dtype
            )
        else:
            self.jitter_val = jitter_val
            

    @property
    def prior_distribution(self) -> MultivariateNormal:
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self) -> MultivariateNormal:
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self) -> bool:
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self) -> Tensor:
        return super().kl_divergence().sum(dim=self.latent_dim)

    def __call__(
        self, x: Tensor, prior: bool = False, task_indices: Optional[LongTensor] = None, **kwargs
    ) -> Union[MultitaskMultivariateNormal, MultivariateNormal]:
        
        self.latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        num_batch = len(self.latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        if task_indices is None:
            num_dim = num_batch + len(self.latent_dist.event_shape)

            # Every data point will get an output for each task
            # Therefore, we will set up the lmc_coefficients shape for a matmul
            
            # lmc_coefficients: ... Q x num_lpf
            lmc_coefficients = self.lmc_coefficients.expand(*self.latent_dist.batch_shape, self.lmc_coefficients.size(-1))

            # latent_mean: ... x N x Q
            latent_mean = self.latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
            
            # mean: ... x N x num_lpf
            mean = latent_mean @ lmc_coefficients.permute(
                *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
            )

            # latent_covar: ... x Q x N x N  
            latent_covar = self.latent_dist.lazy_covariance_matrix
            
            # lmc_factor: ... x Q x num_lpf x num_lpf
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            
            # latent_kron: ... x Q x (N x num_lpf) x (N x num_lpf)
            latent_kron = KroneckerProductLinearOperator(latent_covar, lmc_factor)
            
            # covar: ... x (N x num_lpf) x (N x num_lpf) 
            covar = latent_kron.sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(self.jitter_val)

            # Done!
            function_dist = MultitaskMultivariateNormal(mean, covar)

        else:            
            lmc_coefficients = _select_lmc_coefficients(self.lmc_coefficients, task_indices)

            # Mean: ... x N
            mean = (self.latent_dist.mean * lmc_coefficients).sum(latent_dim)

            # Covar: ... x N x N
            latent_covar = self.latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = (latent_covar * lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(self.jitter_val)

            # Done!
            function_dist = MultivariateNormal(mean, covar)

        return function_dist