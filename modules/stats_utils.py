import numpy as np
import pyro.distributions as dist
import torch

def compute_log_likelihood(
    f_samples,
    likelihoods,
    y,
    **kwargs
):
    
    n_samples, n_data, n_lpf = f_samples.size()
    q_outcomes = y.size(-1)
    
    assert n_data == y.size(0), f"f.size(1) and y.size(0) must be equal size. Received sizes for f and y ({n_data, y.size(0)})."
    
    log_likelihood = np.empty((n_samples, n_data, len(likelihoods)))
    average_nlpd = np.empty(len(likelihoods))
    
    for s, f in enumerate(f_samples):
        j = 0
        d = 0
        all_dirichlet_counter = 0
        for l, likelihood in enumerate(likelihoods):                    
            y_ = None
            iter_d = 1
            iter_j = 1       
            y_ = y[...,d]
            
            if likelihood == "HetNormal":
                likelihood_fn = dist.Normal(f[..., j], f[...,j+1].exp())
                iter_j = 2
            elif likelihood == "Poisson":
                likelihood_fn = dist.Poisson(f[...,j].exp())
            elif likelihood == "Bernoulli":
                likelihood_fn = dist.Bernoulli(logits=f[...,j])
            elif likelihood == "Categorical":
                n_categories = y_.max().to(torch.long).item() + 1
                likelihood_fn = dist.Categorical(logits=f[...,j:(n_categories)])
                iter_j = n_categories
            elif likelihood == "OrderedLogistic":
                n_classes = y_.max().to(torch.long).item() + 1
                cutoff_pts = kwargs["cutoff_pts"]
                assert cutoff_pts is not None
                likelihood_fn = dist.OrderedLogistic(f[...,j], cutoff_pts)
            elif likelihood == "HetDirichlet":                        
                n_comp = kwargs["n_compositions"]
                if isinstance(n_comp, list):
                    n_comp = n_comp[all_dirichlet_counter]
                y_ = y[...,d:(d + n_comp)]
                mu = f[...,j:(j + n_comp)].softmax(-1)
                precision = f[...,(j + n_comp):(j + n_comp + 1)].exp()
                concentration = mu * precision
                likelihood_fn = dist.Dirichlet(concentration)
                iter_d = n_comp
                iter_j = n_comp + 1
                all_dirichlet_counter += 1
            
            log_likelihood_val = likelihood_fn.log_prob(y_)
            log_likelihood[s,...,l] = log_likelihood_val
            nan_value = log_likelihood_val.isnan().sum().item()
            
            assert nan_value == 0, f"nan values found in log_likelihood."
            
            d += iter_d
            j += iter_j
                            
        assert n_lpf == j, f"Not all latent parameter functions (LPF) were used as inputs to likelihood distributions. Total LPF is {n_lpf}, however {j} of the LPFs were used as inputs."
        
        assert q_outcomes == d, f"Size of output y does not match event_shape of the likelihood. Size of y:{q_outcomes}, whereas event_shape of likelihood is {d})."
    
    return log_likelihood

def compute_nlpd(log_likelihood_samples, test_y):
    ll = np.array(log_likelihood_samples)
    nlpd = -(ll.sum(1).mean(0) / test_y.size(0))
    return nlpd
