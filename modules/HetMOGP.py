import torch
import gpytorch
import pyro
import pyro.distributions as dist

from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.utils.interpolation import left_interp

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy

class LMCSeparableKernel(gpytorch.kernels.Kernel):
    
    def __init__(self, kernel_list, **kwargs):
        
        super(LMCSeparableKernel, self).__init__(**kwargs)
        self.kernel_modules = torch.nn.ModuleList(kernel_list)
        self.init_kwargs = kwargs
    
    def forward(self, x1, x2, **params):
        covar_x = self.kernel_modules[0].forward(x1, x2, **params)
        i=1
        for k in self.kernel_modules[1:]:
            add_covar_x = k.forward(x1, x2, **params)
            if isinstance(add_covar_x, DenseLinearOperator):
                add_covar_x = add_covar_x.tensor
            covar_x = torch.vstack([covar_x, add_covar_x])
            i += 1
        return covar_x

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
        base_variational_strategy,
        num_tasks,
        num_latents=1,
        latent_dim=-1,
        jitter_val=None,
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        batch_shape = self.base_variational_strategy._variational_distribution.batch_shape
        
        # Added latent dist
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
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=self.latent_dim)

    def __call__(self, x, task_indices=None, prior=False, **kwargs):
        
        self.latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        num_batch = len(self.latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim

        if task_indices is None:
            num_dim = num_batch + len(self.latent_dist.event_shape)

            # Every data point will get an output for each task
            # Therefore, we will set up the lmc_coefficients shape for a matmul
            lmc_coefficients = self.lmc_coefficients.expand(*self.latent_dist.batch_shape, self.lmc_coefficients.size(-1))

            # Mean: ... x N x num_tasks
            latent_mean = self.latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
            mean = latent_mean @ lmc_coefficients.permute(
                *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
            )

            # Covar: ... x (N x num_tasks) x (N x num_tasks)
            latent_covar = self.latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = KroneckerProductLinearOperator(latent_covar, lmc_factor).sum(latent_dim)
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

class HetMOGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, num_lpf, likelihoods, kernels, name_prefix="lmc_mogp", num_inducing=100, jitter_val=1e-06):
        
        self.name_prefix = name_prefix
        self.likelihoods = likelihoods
        self.num_latents = len(kernels)
        
        input_dim = 1
        if train_x.dim() > 1:
            input_dim = train_x.size(-1)
        
        # Let's use a different set of inducing points for each latent function
        
        # from original hetmogp.py
        # inducing_points = torch.linspace(0, 1, num_inducing).expand(self.num_latents, input_dim, num_inducing).permute(0,2,1)

        # from lmc_1D.ipynb
        inducing_points = torch.linspace(0, 1, num_inducing).expand(1, input_dim, num_inducing).permute(0,2,1)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([self.num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, 
                variational_distribution, # variational distribution over the inducing point values, q(u),
                learn_inducing_locations=True
            )
        
        variational_strategy = LMCVariationalStrategy(
            base_variational_strategy=base_variational_strategy,
            num_tasks=num_lpf,
            num_latents=self.num_latents,
            latent_dim=-1,
            jitter_val=jitter_val
        )
        
        super().__init__(variational_strategy=variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        # self.covar_module = kernel
        self.covar_module = LMCSeparableKernel(kernels, batch_shape=torch.Size([self.num_latents]))

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def compute_likelihoods(self, y, function_dist, likelihoods):
        
        assert y.size(-1) == len(likelihoods), f"Size of output y does not much size of likelihoods list. \
            Input size is ({y.size(-1)}, {len(likelihoods)})."
        
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            f = pyro.sample(self.name_prefix + ".f(x)", function_dist)
            
            j = 0
            for i, l in enumerate(likelihoods):
                if l == "HetGaussian":
                    pyro.sample(
                        self.name_prefix + ".y_" + str(i),
                        dist.Normal(f[..., j], f[...,j+1].exp()).to_event(1),
                        obs=y[...,i]
                    )
                    j += 2
                elif l == "Poisson":
                    pyro.sample(
                        self.name_prefix + ".y_" + str(i),
                        dist.Poisson(f[...,j].exp()).to_event(1),
                        obs=y[...,i]
                    )
                    j += 1
                elif l == "Bernoulli":
                    pyro.sample(
                        self.name_prefix + ".y_" + str(i),
                        dist.Bernoulli(logits=f[...,j]).to_event(1),
                        obs=y[...,i]
                    )
                    j += 1
                elif l == "Categorical":
                    n_categories = y[...,i].max().to(torch.long).item() + 1
                    pyro.sample(
                        self.name_prefix + ".y_" + str(i),
                        dist.Categorical(logits=f[...,j:(n_categories+1)]).to_event(1),
                        obs=y[...,i]
                    )
                    j += n_categories
                    
            assert f.size(-1) == j, f"Not all latent parameter functions (LPF) were used as inputs to likelihood distributions. \
                Total LPF is {f.size(-1)}, however only {j} of the LPFs were used as inputs."
    
    # @pyro.poutine.scale(scale=1.0/(train_y.numel()))
    def guide(self, x, y):
         # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(x)
        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
            # Sample from latent function distribution
            f = pyro.sample(self.name_prefix + ".f(x)", function_dist)
        
    # @pyro.poutine.scale(scale=1.0/(train_y.numel()))
    def model(self, x, y):
        pyro.module(self.name_prefix + ".gp", self)

        # Get p(f) - prior distribution of latent function
        function_dist = self.pyro_model(x)        
        self.compute_likelihoods(y, function_dist, self.likelihoods)
        