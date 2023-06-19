"""
Summary of the Fong & Holmes (2021) implementation on Conformal Bayesian
Computation. Specifically, this module deals with the sparse regression
(and its uncertainty quantification) assessed both for the `sklearn`:
- Boston house prices' dataset (boston)
- Diabetes dataset (diabetes)

The following 4 methods are implemented:
- Bayesian inference (bayes)
- Conformal bayes (cb)
- Split conformal prediction (split)
- Full conformal prediction (full)

A brief summary of the models can be found below:
- 'bayes': This method uses the posterior samples of the model parameters to
compute the likelihood of the test point belonging to each class. It uses the
likelihoods to construct a prediction interval.
- 'split': This method fits a LASSO logistic regression model to the first
half of the training data, then computes the residuals on the second half
of the training data. It uses the residuals to define a threshold for the
prediction intervals, which are constructed using the logistic regression
model on the testing data.
- 'full': This method fits a logistic regression model to the combined
training and testing data, then computes the rank of the test point in
the combined data set. It uses the rank to define a threshold for the
prediction intervals, which are constructed using the logistic
regression model on the test point.
- 'cb': uses the posterior distribution from MCMC sampling to define
conformal prediction intervals.


The execution chain is as it follows:
- The data is loaded using `load_train_test_sparse_regression`
- The Bayesian inference (bayes) is performed through the function
`run_sparse_regression_mcmc` (which requires the MCMC computations
defined at `fit_mcmc_laplace`).
- Then, using the former results, the Conformal Bayesian method
(along the 2 other baselines) is applied using the function
`run_sparse_regression_conformal`:
  - The 'split' and 'full' conformal baselines are defined in the functions
`conformal_split` and `conformal_full`, respectively.
  - The 'cb' method is implemented in the function `compute_cb_region_IS`
  and it also uses the own `logistic_loglikelihood` clause.
- Once launched the main function `run_sparse_regression_conformal`, it
loads the training and testing data and the posterior samples of the model
parameters (computed because `run_sparse_regression_mcmc` was run first).
- Then it iterates through a specified number of repetitions, applying each
of the four methods to compute prediction intervals and coverage probabilities
for each test point.
- For each repetition, the function applies the split method, the full method,
the Bayesian method, and the conformal Bayes method to compute the prediction
intervals and coverage probabilities. It also records the computation time
for each method.
- At the end of the function, these results are saved to various files,
including the coverage probabilities, the lengths of the prediction
intervals, and the computation times for each method.

"""


import pymc3 as pm
from os import makedirs
from jax.ops import index_update
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
import time
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from jax.scipy.stats import norm
from sklearn.linear_model import LassoCV, Lasso


# #############################################################################
# LOAD DATA (sparse regression)
# #############################################################################

def load_train_test_sparse_regression(train_frac, dataset, seed):
    # Load dataset
    if dataset == "diabetes":
        x, y = load_diabetes(return_X_y=True)
    elif dataset == "boston":
        x, y = load_boston(return_X_y=True)
    else:
        print('Invalid dataset')
        return

    n = np.shape(x)[0]
    d = np.shape(x)[1]

    # Standardize beforehand (for validity)
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    y = (y - np.mean(y)) / np.std(y)

    # Train test split
    ind_train, ind_test = train_test_split(np.arange(n),
                                           train_size=int(train_frac * n),
                                           random_state=seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.linspace(np.min(y_train) - 2, np.max(y_train) + 2, 100)

    return x_train, y_train, x_test, y_test, y_plot, n, d


# #############################################################################
# BAYESIAN INFERENCE (MCMC)
# #############################################################################

# Laplace prior PyMC3 model
def fit_mcmc_laplace(y, x, B, seed=100, misspec: bool = False):
    with pm.Model() as _:
        p = np.shape(x)[1]
        # Laplace
        b = pm.Gamma('b', alpha=1, beta=1)
        beta = pm.Laplace('beta', mu=0, b=b, shape=p)
        intercept = pm.Flat('intercept')
        if misspec:
            sigma = pm.HalfNormal("sigma", sigma=0.02)  # misspec prior
        else:
            sigma = pm.HalfNormal("sigma", sigma=1)
        obs = pm.Normal('obs', mu=pm.math.dot(x, beta) + intercept,
                        sigma=sigma, observed=y)

        trace = pm.sample(B, random_seed=seed, chains=4)
    beta_post = trace['beta']
    intercept_post = trace['intercept'].reshape(-1, 1)
    sigma_post = trace['sigma'].reshape(-1, 1)
    b_post = trace['b'].reshape(-1, 1)
    print(np.mean(sigma_post))  # check misspec.

    return beta_post, intercept_post, b_post, sigma_post


# #############################################################################
# APPLICATION OF THE BAYESIAN INFERENCE (MCMC)

# Repeat 50 mcmc runs for different train test splits
def run_sparse_regression_mcmc(dataset, misspec: bool = False):
    # Repeat over 50 reps
    rep = 50
    train_frac = 0.7
    B = 2000

    # Initialize
    x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_regression(
        train_frac, dataset, 100)

    beta_post = np.zeros((rep, 4 * B, d))
    intercept_post = np.zeros((rep, 4 * B, 1))
    b_post = np.zeros((rep, 4 * B, 1))
    sigma_post = np.zeros((rep, 4 * B, 1))
    times = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j
        x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_regression(
            train_frac, dataset, seed)
        start = time.time()
        beta_post[j], intercept_post[j], b_post[j], sigma_post[
            j] = fit_mcmc_laplace(y, x, B, seed, misspec)
        end = time.time()
        times[j] = (end - start)

    # Save posterior samples
    if misspec:
        suffix = dataset
    else:
        suffix = dataset + "_misspec"

    print("{}: {} ({})".format(suffix, np.mean(times),
                               np.std(times) / np.sqrt(rep)))

    np.save("samples/beta_post_sparsereg_{}".format(suffix), beta_post)
    np.save("samples/intercept_post_sparsereg_{}".format(suffix),
            intercept_post)
    np.save("samples/b_post_sparsereg_{}".format(suffix), b_post)
    np.save("samples/sigma_post_sparsereg_{}".format(suffix), sigma_post)
    np.save("samples/times_sparsereg_{}".format(suffix), times)


# #############################################################################
# CONFORMAL PREDICTION
# #############################################################################

# Lasso split method baseline
def conformal_split(y, x, x_test, alpha, y_plot, seed=100):
    n = np.shape(y)[0]
    n_test = np.shape(x_test)[0]
    # Fit lasso to training set
    ls = LassoCV(cv=5, random_state=seed)
    n_train = int(n / 2)
    ls.fit(x[0:n_train], y[0:n_train])
    # Predict lasso on validation set
    y_pred_val = ls.predict(x[n_train:])
    resid = np.abs(y_pred_val - y[n_train:])
    k = int(np.ceil((n / 2 + 1) * (1 - alpha)))
    d = np.sort(resid)[k - 1]
    # Compute split conformal interval
    band_split = np.zeros((n_test, 2))
    y_pred_test = ls.predict(x_test)  # predict lasso on test
    band_split[:, 0] = y_pred_test - d
    band_split[:, 1] = y_pred_test + d
    return band_split


# Lasso full method baseline
def conformal_full(y, x, x_test, alpha, y_plot, C, seed=100):
    n = np.shape(y)[0]
    rank_full = np.zeros(np.shape(y_plot)[0])
    for i in range(np.shape(y_plot)[0]):
        y_new = y_plot[i]
        x_aug = np.concatenate((x, x_test), axis=0)
        y_aug = np.append(y, y_new)
        ls = Lasso(alpha=C, random_state=seed)
        ls.fit(x_aug, y_aug)
        y_pred_val = ls.predict(x_aug)
        resid = np.abs(y_pred_val - y_aug)
        rank_full[i] = np.sum(resid >= resid[-1]) / (n + 1)
    region_full = rank_full > alpha
    return region_full


# #############################################################################
# CONFORMAL BAYESIAN
# #############################################################################

# CONFORMAL FROM MCMC SAMPLES (JAX IMPLEMENTATION)

# Compute bayesian central 1-alpha credible interval from MCMC samples
@jit
def compute_bayes_band_MCMC(alpha, y_plot, cdf_pred):
    cdf_pred = jnp.mean(cdf_pred, axis=1)

    band_bayes = np.zeros(2)
    band_bayes = index_update(band_bayes, 0, y_plot[
        jnp.argmin(jnp.abs(cdf_pred - alpha / 2))])
    band_bayes = index_update(band_bayes, 1, y_plot[
        jnp.argmin(jnp.abs(cdf_pred - (1 - alpha / 2)))])
    return band_bayes


# compute rank (un-normalized by n+1)
def compute_rank_IS(logp_samp_n, logwjk):
    # n = jnp.shape(logp_samp_n)[1]  # logp_samp_n is B x n
    # n_plot = jnp.shape(logwjk)[0]
    # rank_cp = jnp.zeros(n_plot)

    # compute importance sampling weights and normalizing
    wjk = jnp.exp(logwjk)
    Zjk = jnp.sum(wjk, axis=1).reshape(-1, 1)

    # compute predictives for y_i,x_i and y_new,x_n+1
    p_cp = jnp.dot(wjk / Zjk, jnp.exp(logp_samp_n))
    p_new = jnp.sum(wjk ** 2, axis=1).reshape(-1, 1) / Zjk

    # compute nonconformity score and sort
    pred_tot = jnp.concatenate((p_cp, p_new), axis=1)
    rank_cp = np.sum(pred_tot <= pred_tot[:, -1].reshape(-1, 1), axis=1)
    return rank_cp


# compute region of grid which is in confidence set
@jit
def compute_cb_region_IS(alpha, logp_samp_n,
                         logwjk):  # assumes they are connected
    n = jnp.shape(logp_samp_n)[1]  # logp_samp_n is B x n
    rank_cp = compute_rank_IS(logp_samp_n, logwjk)
    region_true = rank_cp > alpha * (n + 1)
    return region_true


# #############################################################################
# MAIN PUBLIC FUNCTION (application of CONFORMAL BAYESIAN and the 3 baselines)
# #############################################################################

def run_sparse_regression_conformal(dataset, misspec: bool = False):
    # Compute intervals
    # Initialize
    train_frac = 0.7
    x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_regression(
        train_frac, dataset, 100)

    # Load posterior samples
    if misspec:
        suffix = dataset
    else:
        suffix = dataset + "_misspec"

    beta_post = jnp.load("samples/beta_post_sparsereg_{}.npy".format(suffix))
    intercept_post = jnp.load(
        "samples/intercept_post_sparsereg_{}.npy".format(suffix))
    sigma_post = jnp.load("samples/sigma_post_sparsereg_{}.npy".format(suffix))

    # Initialize
    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep, n_test))
    coverage_cb_exact = np.zeros((rep, n_test))  # avoiding grid effects
    coverage_bayes = np.zeros((rep, n_test))
    coverage_split = np.zeros((rep, n_test))
    coverage_full = np.zeros((rep, n_test))

    length_cb = np.zeros((rep, n_test))
    length_bayes = np.zeros((rep, n_test))
    length_split = np.zeros((rep, n_test))
    length_full = np.zeros((rep, n_test))

    band_bayes = np.zeros((rep, n_test, 2))
    region_cb = np.zeros((rep, n_test, np.shape(y_plot)[0]))
    region_full = np.zeros((rep, n_test, np.shape(y_plot)[0]))
    band_split = np.zeros((rep, n_test, 2))

    times_bayes = np.zeros(rep)
    times_cb = np.zeros(rep)
    times_split = np.zeros(rep)
    times_full = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j
        # load dataset
        x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_regression(
            train_frac, dataset, seed)
        dy = y_plot[1] - y_plot[0]

        # split method
        start = time.time()
        band_split[j] = conformal_split(y, x, x_test, alpha, y_plot, seed)
        coverage_split[j] = (y_test >= band_split[j, :, 0]) & (
                    y_test <= band_split[j, :, 1])
        length_split[j] = np.abs(band_split[j, :, 0] - band_split[j, :, 1])
        end = time.time()
        times_split[j] = end - start

        # full method
        start = time.time()
        C = 0.004
        for i in (range(n_test)):
            region_full[j, i] = conformal_full(y, x, x_test[i:i + 1], alpha,
                                               y_plot, C, seed)
            coverage_full[j, i] = region_full[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_full[j, i] = np.sum(region_full[j, i]) * dy
        end = time.time()
        times_full[j] = end - start

        # Bayes
        start = time.time()

        @jit  # normal cdf from posterior samples
        def normal_likelihood_cdf(_y, _x):
            return norm.cdf(
                _y, loc=jnp.dot(beta_post[j],
                                _x.transpose()) + intercept_post[j],
                scale=sigma_post[j])  # compute likelihood samples

        # Precompute cdfs
        cdf_test = normal_likelihood_cdf(y_plot.reshape(-1, 1, 1), x_test)

        for i in (range(n_test)):
            band_bayes[j, i] = compute_bayes_band_MCMC(
                alpha, y_plot, cdf_test[:, :, i])
            coverage_bayes[j, i] = (y_test[i] >= band_bayes[j, i, 0]) & (
                        y_test[i] <= band_bayes[j, i, 1])
            length_bayes[j, i] = np.abs(
                band_bayes[j, i, 1] - band_bayes[j, i, 0])
        end = time.time()
        times_bayes[j] = end - start

        # Conformal Bayes
        start = time.time()

        @jit  # normal loglik from posterior samples
        def normal_loglikelihood(_y, _x):
            return norm.logpdf(
                _y,
                loc=jnp.dot(beta_post[j], _x.transpose()) + intercept_post[j],
                scale=sigma_post[j])  # compute likelihood samples

        logp_samp_n = normal_loglikelihood(y, x)
        logwjk = normal_loglikelihood(y_plot.reshape(-1, 1, 1), x_test)
        logwjk_test = normal_loglikelihood(y_test, x_test).reshape(1, -1,
                                                                   n_test)

        for i in (range(n_test)):
            region_cb[j, i] = compute_cb_region_IS(
                alpha, logp_samp_n, logwjk[:, :, i])
            coverage_cb[j, i] = region_cb[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]  # grid coverage
            length_cb[j, i] = np.sum(region_cb[j, i]) * dy
        end = time.time()
        times_cb[j] = end - start

        # compute exact coverage to avoid grid effects
        for i in (range(n_test)):
            coverage_cb_exact[j, i] = compute_cb_region_IS(
                alpha, logp_samp_n, logwjk_test[:, :, i])  # exact coverage

    # #Save regions (need to update)
    np.save("results/region_cb_sparsereg_{}".format(suffix), region_cb)
    np.save("results/band_bayes_sparsereg_{}".format(suffix), band_bayes)
    np.save("results/band_split_sparsereg_{}".format(suffix), band_split)
    np.save("results/region_full_sparsereg_{}".format(suffix), band_split)

    np.save("results/coverage_cb_sparsereg_{}".format(suffix), coverage_cb)
    np.save("results/coverage_cb_exact_sparsereg_{}".format(suffix),
            coverage_cb_exact)
    np.save("results/coverage_bayes_sparsereg_{}".format(suffix),
            coverage_bayes)
    np.save("results/coverage_split_sparsereg_{}".format(suffix),
            coverage_split)
    np.save("results/coverage_full_sparsereg_{}".format(suffix), coverage_full)

    np.save("results/length_cb_sparsereg_{}".format(suffix), length_cb)
    np.save("results/length_bayes_sparsereg_{}".format(suffix), length_bayes)
    np.save("results/length_split_sparsereg_{}".format(suffix), length_split)
    np.save("results/length_full_sparsereg_{}".format(suffix), length_full)

    np.save("results/times_cb_sparsereg_{}".format(suffix), times_cb)
    np.save("results/times_bayes_sparsereg_{}".format(suffix), times_bayes)
    np.save("results/times_split_sparsereg_{}".format(suffix), times_split)
    np.save("results/times_full_sparsereg_{}".format(suffix), times_full)


if __name__ == '__main__':
    __dataset: str = 'diabetes'  # 'boston'
    makedirs('samples', exist_ok=True)
    makedirs('results', exist_ok=True)
    # run MCMC
    run_sparse_regression_mcmc('diabetes', misspec=False)
    run_sparse_regression_mcmc('diabetes', misspec=True)

    # run Conformal Bayes
    run_sparse_regression_conformal('diabetes', misspec=False)
    run_sparse_regression_conformal('diabetes', misspec=True)
