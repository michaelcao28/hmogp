from torch.nn import Module as TModule
import torch
import gpytorch
from gpytorch.priors import Prior

class UniformPrior(Prior, torch.distributions.Uniform):
  """
  Uniform prior
  """

  def __init__(self, low, high, validate_args=None, transform=None):
    if not isinstance(low, torch.Tensor):
      low = torch.tensor(low, dtype=torch.float)
    if not isinstance(high, torch.Tensor):
      high = torch.tensor(high, dtype=torch.float)
    TModule.__init__(self)
    self.low = low
    self.high = high
    torch.distributions.Uniform.__init__(self, low, high, validate_args=validate_args)
    del self.low
    del self.high
    self.register_buffer("low", low)
    self.register_buffer("high", high)
    self._transform = transform 

  def expand(self, batch_shape):
    batch_shape = torch.Size(batch_shape)
    return UniformPrior(self.low.expand(batch_shape), self.high.expand(batch_shape))

class BetaPrior(Prior, torch.distributions.Beta):
  """
  Beta prior
  """

  def __init__(self, alpha0, beta0, validate_args=None, transform=None):
    if not isinstance(alpha0, torch.Tensor):
      alpha0 = torch.tensor(alpha0)
    if not isinstance(beta0, torch.Tensor):
      beta0 = torch.tensor(beta0)
    TModule.__init__(self)
    self.alpha0 = alpha0
    self.beta0 = beta0
    torch.distributions.Beta.__init__(self, alpha0, beta0, validate_args=validate_args)
    del self.alpha0
    del self.beta0
    self.register_buffer("alpha0", alpha0)
    self.register_buffer("beta0", beta0)
    self._transform = transform 

  def expand(self, batch_shape):
    batch_shape = torch.Size(batch_shape)
    return BetaPrior(self.alpha0.expand(batch_shape), self.beta0.expand(batch_shape))