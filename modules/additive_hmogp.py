from typing import Optional, Union
import torch
import gpytorch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints

from linear_operator.operators.dense_linear_operator import DenseLinearOperator
from linear_operator.operators import KroneckerProductLinearOperator, RootLinearOperator
from linear_operator.utils.interpolation import left_interp

from gpytorch import settings
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy

from gpytorch.priors import Prior


from gpytorch.models.gp import GP
import numpy as np

import re

class _PyroMixin(object):
    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        # Inducing values q(u)
        with pyro.poutine.scale(scale=beta):
            variational_distribution = self.variational_strategy.variational_distribution
            variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
            pyro.sample(name_prefix + ".u", variational_distribution)

        # Draw samples from q(f)
        function_dist = self(input, prior=False)
        function_dist = pyro.distributions.Normal(loc=function_dist.mean, scale=function_dist.stddev).to_event(
            len(function_dist.event_shape) - 1
        )
        return function_dist.mask(False)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        # Inducing values p(u)
        with pyro.poutine.scale(scale=beta):
            prior_distribution = self.variational_strategy.prior_distribution
            prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
            u_samples = pyro.sample(name_prefix + ".u", prior_distribution)
        
        # Include term for GPyTorch priors
        log_prior = torch.tensor(0.0, dtype=u_samples.dtype, device=u_samples.device)
        for _, module, prior, closure, _ in self.named_priors():            
            log_prior.add_(prior.log_prob(closure(module)).sum())
        pyro.factor(name_prefix + ".log_prior", log_prior)

        # Include factor for added loss terms
        added_loss = torch.tensor(0.0, dtype=u_samples.dtype, device=u_samples.device)
        for added_loss_term in self.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
        pyro.factor(name_prefix + ".added_loss", added_loss)

        # Draw samples from p(f)
        function_dist = self(input, prior=True)
        function_dist = pyro.distributions.Normal(loc=function_dist.mean, scale=function_dist.stddev).to_event(
            len(function_dist.event_shape) - 1
        )
        return function_dist.mask(False)

class ApproximateGP(GP, _PyroMixin):
    r"""
    The base class for any Gaussian process latent function to be used in conjunction
    with approximate inference (typically stochastic variational inference).
    This base class can be used to implement most inducing point methods where the
    variational parameters are learned directly.

    :param ~gpytorch.variational._VariationalStrategy variational_strategy: The strategy that determines
        how the model marginalizes over the variational distribution (over inducing points)
        to produce the approximate posterior distribution (over data)

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Example:
        >>> class MyVariationalGP(gpytorch.models.PyroGP):
        >>>     def __init__(self, variational_strategy):
        >>>         super().__init__(variational_strategy)
        >>>         self.mean_module = gpytorch.means.ZeroMean()
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # variational_strategy = ...
        >>> model = MyVariationalGP(variational_strategy)
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>>
        >>> # optimization loop for variational parameters...
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the approximate GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(self, variational_strategy):
        super().__init__()
        self.variational_strategy = variational_strategy

    def forward(self, x):
        raise NotImplementedError

    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        r"""
        (For Pyro integration only). The component of a `pyro.guide` that
        corresponds to drawing samples from the latent GP function.

        :param torch.Tensor input: The inputs :math:`\mathbf X`.
        :param float beta: (default=1.) How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
            term by.
        :param str name_prefix: (default="") A name prefix to prepend to pyro sample sites.
        """
        return super().pyro_guide(input, beta=beta, name_prefix=name_prefix)

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        r"""
        (For Pyro integration only). The component of a `pyro.model` that
        corresponds to drawing samples from the latent GP function.

        :param torch.Tensor input: The inputs :math:`\mathbf X`.
        :param float beta: (default=1.) How much to scale the :math:`\text{KL} [ q(\mathbf f) \Vert p(\mathbf f) ]`
            term by.
        :param str name_prefix: (default="") A name prefix to prepend to pyro sample sites.
        :return: samples from :math:`q(\mathbf f)`
        :rtype: torch.Tensor
        """
        return super().pyro_model(input, beta=beta, name_prefix=name_prefix)

    def get_fantasy_model(self, inputs, targets, **kwargs):
        r"""
        Returns a new GP model that incorporates the specified inputs and targets as new training data using
        online variational conditioning (OVC).

        This function first casts the inducing points and variational parameters into pseudo-points before
        returning an equivalent ExactGP model with a specialized likelihood.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP

        Reference: "Conditioning Sparse Variational Gaussian Processes for Online Decision-Making,"
            Maddox, Stanton, Wilson, NeurIPS, '21
            https://papers.nips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html

        """
        return self.variational_strategy.get_fantasy_model(inputs=inputs, targets=targets, **kwargs)

    def __call__(self, inputs, prior=False, **kwargs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        return self.variational_strategy(inputs, prior=prior, **kwargs)

class LMCSeparableKernel(gpytorch.kernels.Kernel):
    # Stores each vector in covar_x to plot ILFs / LPFs
    # Allows for additive kernels
    
    def __init__(self, kernel_list, active_dims=None, **kwargs):
        
        super(LMCSeparableKernel, self).__init__(**kwargs)
        self.kernel_modules = torch.nn.ModuleList(kernel_list)
        self.active_dims_ = torch.asarray(active_dims, dtype=torch.long)
    
    def forward(self, x1, x2, **params):
        
        x1_, x2_ = x1, x2
        i = 0
        for d in self.active_dims_:
            x1_ = x1[...,d]
            if x2 is not None:
                x2_ = x2[...,d]
            if i == 0:
                covar_x = self.kernel_modules[i].forward(x1_, x2_, **params)  
                if not isinstance(covar_x, torch.Tensor):
                    covar_x = covar_x.to_dense()               
            else:
                add_covar_x = self.kernel_modules[i].forward(x1_, x2_, **params)
                if not isinstance(add_covar_x, torch.Tensor):
                    add_covar_x = add_covar_x.to_dense()
                covar_x = torch.concatenate([covar_x, add_covar_x])
                # covar_x = torch.vstack([covar_x, add_covar_x])
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
        base_variational_strategy: _VariationalStrategy,
        num_tasks: int,
        num_latents: int = 1,
        latent_dim: int = -1,
        independent_outputs: bool = False,
        jitter_val: Optional[float] = None,
        lmc_coefficients_prior: Optional[Prior] = None
    ):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.num_tasks = num_tasks
        batch_shape = self.base_variational_strategy._variational_distribution.batch_shape
        
        # Added
        self.ind_latent_fn = None
        self.independent_outputs = independent_outputs

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
        if not self.independent_outputs:
            lmc_coefficients = torch.randn(*batch_shape, self.num_tasks)
            self.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))
            # Set prior on lmc_coefficients
            if lmc_coefficients_prior:
                self.register_prior("lmc_coefficients_prior", lmc_coefficients_prior, "lmc_coefficients")
        else:
            self.lmc_coefficients = torch.eye(self.num_tasks)

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

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=self.latent_dim)

    def __call__(
        self, x: torch.Tensor, prior: bool = False, task_indices: Optional[torch.LongTensor] = None, **kwargs
    ) -> Union[MultitaskMultivariateNormal, MultivariateNormal]:
        
        latent_dist = self.base_variational_strategy(x, prior=prior, **kwargs)
        num_batch = len(latent_dist.batch_shape)
        latent_dim = num_batch + self.latent_dim
        
        # Very odd way to get the latent mean. Weren't able to find another way
        if "AddBackward0" in str(latent_dist.mean.grad_fn):
            self.ind_latent_fn = latent_dist

        if task_indices is None:      
                  
            num_dim = num_batch + len(latent_dist.event_shape)

            # Every data point will get an output for each task
            # Therefore, we will set up the lmc_coefficients shape for a matmul
            
            if self.independent_outputs:
                self.lmc_coefficients = torch.eye(self.num_tasks, device=x.device)
            
            # lmc_coefficients: ... Q x num_lpf
            lmc_coefficients = self.lmc_coefficients.expand(*latent_dist.batch_shape, self.lmc_coefficients.size(-1))

            # latent_mean: ... x N x Q
            latent_mean = latent_dist.mean.permute(*range(0, latent_dim), *range(latent_dim + 1, num_dim), latent_dim)
            
            # mean: ... x N x num_lpf
            mean = latent_mean @ lmc_coefficients.permute(
                *range(0, latent_dim), *range(latent_dim + 1, num_dim - 1), latent_dim, -1
            )

            # latent_covar: ... x Q x N x N  
            latent_covar = latent_dist.lazy_covariance_matrix
            
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
            mean = (latent_dist.mean * lmc_coefficients).sum(latent_dim)

            # Covar: ... x N x N
            latent_covar = latent_dist.lazy_covariance_matrix
            lmc_factor = RootLinearOperator(lmc_coefficients.unsqueeze(-1))
            covar = (latent_covar * lmc_factor).sum(latent_dim)
            # Add a bit of jitter to make the covar PD
            covar = covar.add_jitter(self.jitter_val)

            # Done!
            function_dist = MultivariateNormal(mean, covar)

        return function_dist

class HetMOGP(ApproximateGP):
    
    def __init__(self, train_x, num_lpf, likelihoods, kernels, active_dims=None, name_prefix="hmogp", num_inducing=100, jitter_val=1e-06, lmc_coefficients_prior: Optional[Prior] = None, independent_outputs=False, **kwargs):
        
        self.name_prefix = name_prefix
        self.likelihoods = likelihoods
        self.num_lpf = num_lpf
        self.device = train_x.device
        self.independent_outputs = independent_outputs
        
        if self.independent_outputs:
            assert active_dims is None, "active_dims must be empty when independent_outputs=True."
            self.num_latents = num_lpf
        else:
            self.num_latents = len(kernels)
        
        self.elbo_norm = train_x.size(0) #* len(self.likelihoods) scales ELBO for performance
        self.has_ordinal = "OrderedLogistic" in self.likelihoods
        self.has_dirichlet = "Dirichlet" in self.likelihoods
        self.has_hetdirichlet = "HetDirichlet" in self.likelihoods
        likelihoods_str = ' '.join(self.likelihoods)
        self.n_alldirichlet = len(re.findall("Dirichlet", likelihoods_str))
        self.n_dirichlet = self.likelihoods.count("Dirichlet")
        
        if self.has_ordinal:
            self.n_classes = kwargs["n_classes"]
        if self.n_alldirichlet > 0:
            self.n_compositions = kwargs["n_compositions"]
            if self.n_alldirichlet == 1:
                assert isinstance(self.n_compositions,int), "n_compositions must be an integer!"
            else:
                assert isinstance(self.n_compositions,list), "n_compositions must be a list!"               

        if train_x.dim() == 1:
            train_x = train_x.unsqueeze(-1)
        
        if active_dims is not None:
            assert len(active_dims) == len(kernels), f"Size of active_dims does not much size of kernel list. len(active_dims) = {len(active_dims)}, len(kernels) = {len(kernels)}."
        
        # Let's use a different set of inducing points for each latent function
        inducing_points = train_x[torch.randint(0, train_x.size(0), size=(num_inducing,)),...].unsqueeze(0)

        # We have to mark the MeanFieldVariationalDistribution as batch
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
            num_tasks=self.num_lpf,
            num_latents=self.num_latents,
            latent_dim=-1,
            jitter_val=jitter_val,
            lmc_coefficients_prior=lmc_coefficients_prior,
            independent_outputs = self.independent_outputs,
        )
        
        super().__init__(variational_strategy=variational_strategy)
        
        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.num_latents]))
        if not independent_outputs:
            self.covar_module = LMCSeparableKernel(kernels, active_dims=active_dims, batch_shape=torch.Size([self.num_latents]))
        else:
            self.covar_module = kernels

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 
    
    def guide(self, x, y, **kwargs):
         # Get q(f) - variational (guide) distribution of latent function
        elbo_norm = x.size(0) # scale for better training performance
        with pyro.poutine.scale(scale=1.0/self.elbo_norm):
            function_dist = self.pyro_guide(x)
            
            # prior for cutpoints of ordinal
            if self.has_ordinal:
                mu_init = torch.tensor(0.).expand(self.n_classes-1)
                sd_init = torch.tensor(5.).expand(self.n_classes-1)
                mu_cutpoints = pyro.param("mu_cutpoints", mu_init)
                sd_cutpoints = pyro.param("sd_cutpoints", sd_init, constraint=constraints.positive)
                pyro.sample(
                        self.name_prefix + ".cutoff_pts",
                        dist.TransformedDistribution(
                            dist.Normal(mu_cutpoints, sd_cutpoints),
                            dist.transforms.OrderedTransform()
                        ),
                    )
            if self.has_dirichlet:
                alpha_init = torch.tensor(5., device=self.device).expand([self.n_dirichlet])
                beta_init = torch.tensor(.2, device=self.device).expand([self.n_dirichlet])
                alpha = pyro.param("alpha", alpha_init, constraint=constraints.positive)
                beta = pyro.param("beta", beta_init, constraint=constraints.positive)
                pyro.sample(
                    self.name_prefix + ".precision",
                    dist.Gamma(alpha, beta).to_event(1)
                )
            
            with pyro.plate(self.name_prefix + ".data_plate", dim=-1, device=self.device):           
                # Sample from latent function distribution
                f = pyro.sample(self.name_prefix + ".f(x)", function_dist)
        
    def model(self, x, y, **kwargs):
        elbo_norm = x.size(0) # scale for better training performance
        with pyro.poutine.scale(scale=1.0/self.elbo_norm):
            pyro.module(self.name_prefix + ".gp", self)
            
            # prior for cutpoints of ordinal
            if self.has_ordinal:
                cutoff_pts = pyro.sample(
                        self.name_prefix + ".cutoff_pts",
                        dist.TransformedDistribution(
                            dist.Normal(0., 5.).expand([self.n_classes - 1]), 
                            dist.transforms.OrderedTransform()
                        ),
                    )
            if self.has_dirichlet:
                alpha_init = torch.tensor(5., device=self.device).expand([self.n_dirichlet])
                beta_init = torch.tensor(.2, device=self.device).expand([self.n_dirichlet])
                precision = pyro.sample(
                    self.name_prefix + ".precision",
                    dist.Gamma(alpha_init, beta_init).to_event(1)
                )
                
            # Get p(f) - prior distribution of latent function
            function_dist = self.pyro_model(x)        
            
            y_input = y is not None            
            # Begin computing the log likelihood
            with pyro.plate(self.name_prefix + ".data_plate", dim=-1, device=self.device):
                # Sample from latent function distribution
                f = pyro.sample(self.name_prefix + ".f(x)", function_dist)
                j = 0
                d = 0
                all_dirichlet_counter = 0
                dirichlet_counter = 0
                for i, likelihood in enumerate(self.likelihoods):                    
                    y_ = None
                    iter_d = 1
                    iter_j = 1
                    if y_input:             
                        y_ = y[...,d]
                    
                    if likelihood == "Normal":
                        sd = kwargs["sd"]
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Normal(f[..., j], sd[j]).to_event(1),
                            obs=y_
                        )
                        iter_j = 1
                    elif likelihood == "HetNormal":
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Normal(f[..., j], f[...,j+1].exp()).to_event(1),
                            obs=y_
                        )
                        iter_j = 2
                    elif likelihood == "Poisson":
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Poisson(f[...,j].exp()).to_event(1),
                            obs=y_
                        )
                    elif likelihood == "Bernoulli":
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Bernoulli(logits=f[...,j]).to_event(1),
                            obs=y_
                        )
                    elif likelihood == "Categorical":
                        if y_input:
                            n_categories = y_.max().to(torch.long).item() + 1
                        else:
                            n_categories = kwargs["n_categories"]
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Categorical(logits=f[...,j:(n_categories)]).to_event(1),
                            obs=y_
                        )
                        iter_j = n_categories
                    elif likelihood == "OrderedLogistic":
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.OrderedLogistic(f[...,j], cutoff_pts).to_event(1), 
                            obs=y_
                        )
                    elif likelihood == "Dirichlet":                        
                        if y_input:
                            n_comp = self.n_compositions
                            if isinstance(n_comp, list):
                                n_comp = n_comp[all_dirichlet_counter]
                            y_ = y[...,d:(d + n_comp)]
                        else:
                            n_comp = kwargs["n_compositions"]
                            if isinstance(n_comp, list):
                                n_comp = n_comp[all_dirichlet_counter]
                        mu = f[...,j:(j + n_comp)].softmax(-1)
                        # Add [...,None] in precision to broadcast elementwise multiplication even with batches
                        concentration = mu * precision[...,dirichlet_counter,None]
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Dirichlet(concentration).to_event(1),
                            obs=y_
                        )
                        iter_d = n_comp
                        iter_j = n_comp
                        dirichlet_counter += 1
                        all_dirichlet_counter += 1
                    elif likelihood == "HetDirichlet":                        
                        if y_input:
                            n_comp = self.n_compositions
                            if isinstance(n_comp, list):
                                n_comp = n_comp[all_dirichlet_counter]
                            y_ = y[...,d:(d + n_comp)]
                        else:
                            n_comp = kwargs["n_compositions"]
                            if isinstance(n_comp, list):
                                n_comp = n_comp[all_dirichlet_counter]
                        mu = f[...,j:(j + n_comp)].softmax(-1)
                        # Add [...,None] in precision to broadcast elementwise multiplication even with batches
                        # precision = f[...,(j + n_comp):(j + 2*n_comp)].exp()
                        precision = f[...,(j + n_comp):(j + n_comp + 1)].exp()
                        concentration = mu * precision
                        pyro.sample(
                            self.name_prefix + ".y_" + str(i),
                            dist.Dirichlet(concentration).to_event(1),
                            obs=y_
                        )
                        iter_d = n_comp
                        iter_j = n_comp + 1
                        all_dirichlet_counter += 1
                    else:
                        raise ValueError(f"No such likelihood available. The given likelihood is {likelihood}")
                    
                    d += iter_d
                    j += iter_j
                    
                assert f.size(-1) == j, f"Not all latent parameter functions (LPF) were used as inputs to likelihood distributions. Total LPF is {f.size(-1)}, however {j} of the LPFs were used as inputs."
                
                if y_input:
                    assert y.size(-1) == d, f"Size of output y does not match event_shape of the likelihood. Size of y:{y.size(-1)}, whereas event_shape of likelihood is {d})."
        