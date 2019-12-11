import numpy as np
import pandas as pd
import warnings
import scipy.optimize
import scipy.stats as st

def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out

def diff_means(x, y): 
    """Compute the difference in means, means / std."""
    diff_means = (np.mean(x) - np.mean(y))**2 / (np.std(x) - np.std(y))
    return diff_means

def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

def draw_perm_reps(x, y, stat_fun, size=1):
    """Generate array of permuation replicates."""
    return np.array([stat_fun(*draw_perm_sample(x, y)) for _ in range(size)])

def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = diff_means(x_perm, y_perm)

    return out

def double_exponential_cdf(beta_1, beta_2, t):
    # initialize constant to multiply with
    cnt = beta_1 * beta_2/ (beta_2 - beta_1 )
    cdf= cnt * ((1 / beta_1) * (1 - np.exp(-beta_1 * t)) - (1 / beta_2) * (1 - np.exp(-beta_2*t)))
    return cdf

def log_like_iid_gamma_log_params(log_params, n):
    """
    Log likelihood for i.i.d. Gamma measurements with
    input being logarithm of parameters.
    
    Parameters
    ----------
    log_params: array
        Logarithm of the parameters alpha and beta.
    n: array
        Array of counts.
    
    Returns
    -------
    output: float
        log-likelihood.
    """
    log_alpha, log_beta = log_params
    
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)
    
    return np.sum(st.gamma.logpdf(n, alpha, loc=0, scale=1/beta))

def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    NBinom measurements, parametrized by alpha, b=1/beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda log_params, n: -log_like_iid_gamma_log_params(log_params, n),
          x0=np.array([2, 1/300]),
          args=(n,),
          method='L-BFGS-B',
    )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)

def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])

def draw_gamma(alpha, beta, size=1):
    return rg.gamma(alpha, 1/beta, size=size)

def ecdf(x, data):
    """Give the value of an ECDF at arbitrary points x."""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]

def log_sum_exp_trick(betas): 
    """
    Compute log-sum-exp-trick for numerical robustness of 
    log(beta2 - beta 1), i.e. ensure that the difference is always positive. 
    """
    m = np.max(betas)
    x = np.min(betas)
    
    lset = np.log(m) + np.log(1 - (x / m))
    
    return lset

def log_sum_of_exp(betas,t): 
    """
    Compute log sum of exponentials for the double exponential model. 
    """
    m = np.max(betas)
    x = np.min(betas)
    
    log_sum_exp = -x*t + np.log(1 - np.exp((x - m)*t))
    
    return log_sum_exp

def log_like_iid_double_poisson_log_params(log_params, t):
    """
    Log likelihood for i.i.d. double Poisson measurements with
    input being logarithm of parameters.
    
    Parameters
    ----------
    log_params: array
        Logarithm of the parameters beta1, and beta 2.
    t: array
        Array of times to catastrophe.
    
    Returns
    -------
    output: float
        log-likelihood.
    """
    log_beta1, log_beta2 = log_params
    
    n = len(t)
    beta1 = np.exp(log_beta1)
    beta2 = np.exp(log_beta2)
    
    # Collapse to gamma distribution if b1 == b2
    if np.isclose(beta1, beta2):
        return np.sum(st.gamma.logpdf(t, a= 2, loc=0, scale=1/beta1))
    
    # Make a list of betas
    betas = [beta1, beta2]
    
    # Constant for the log_likelihood function
    cnt = n*(np.log(beta1) + np.log(beta2) - log_sum_exp_trick(betas))

    # Compute log likelihood
    log_likelihood = cnt + np.sum(log_sum_of_exp(betas,t))    
    
    return log_likelihood

def mle_iid_double_poisson_log_params(t):
    """Perform MLE for parameters for the double exponential model."""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        res = scipy.optimize.minimize(
            fun=lambda log_params, n: -log_like_iid_double_poisson_log_params(log_params, t),
            x0=np.array([np.log(1/200), np.log(1/300)]),
            args=(t,),
            method='Powell',
            
            # Log bounds 
            #bounds=((-9, -9), (-2,-2))
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)
        
def draw_double_exponential(beta1, beta2, size):
    d = np.empty(size)
    for i in range(size):
        d[i] = rg.exponential(1/beta1) + rg.exponential(1/beta2)
    
    return d

def draw_double_exponential_ecdfs(beta1, beta2, size):
    return rg.exponential(1/beta1, size) + rg.exponential(1/beta2, size)

def ecdf_double_exponential(x, data):
    """"""
    y = np.arange(len(data) + 1) / len(data)
    return y[np.searchsorted(np.sort(data), x, side="right")]
