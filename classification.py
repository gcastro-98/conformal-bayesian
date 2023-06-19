"""
Summary of the Fong & Holmes (2021) implementation on Conformal Bayesian
Computation. Specifically, this module deals with the sparse classification
(and its uncertainty quantification) assessed both for the:
- UCI ML Breast Cancer Wisconsin dataset (breast)
- Little et al. (2008) Parkisons dataset (parkinsons)

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
- The data is loaded using `load_train_test_sparse_classification`
- The Bayesian inference (bayes) is performed through the function
`run_sparse_classification_mcmc` (which requires the MCMC computations
defined at `fit_mcmc_laplace`).
- Then, using the former results, the Conformal Bayesian method
(along the 2 other baselines) is applied using the function
`run_sparse_classification_conformal`:
  - The 'split' and 'full' conformal baselines are defined in the functions
`conformal_split` and `conformal_full`, respectively.
  - The 'cb' method is implemented in the function `compute_cb_region_IS`
  and it also uses the own `logistic_loglikelihood` clause.
- Once launched the main function `run_sparse_classification_conformal`, it
loads the training and testing data and the posterior samples of the model
parameters (computed because `run_sparse_classification_mcmc` was run first).
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
import os
import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import jax.numpy as jnp
from jax import jit
import jax.scipy as jsp
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import pymc3 as pm


# #############################################################################
# LOAD DATA (sparse classification)
# #############################################################################

def load_train_test_sparse_classification(train_frac, dataset, seed):
    # Load dataset
    if dataset == "breast":
        x, y = load_breast_cancer(return_X_y=True)
    elif dataset == "parkinsons":
        data = pd.read_csv('data/parkinsons.data')
        data[data == '?'] = np.nan
        data.dropna(axis=0, inplace=True)
        y = data['status'].values  # convert strings to integer
        x = data.drop(columns=['name', 'status']).values
    else:
        print('Invalid dataset')
        return

    n = np.shape(x)[0]
    d = np.shape(x)[1]

    # Standardize beforehand (for validity)
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    # Train test split
    ind_train, ind_test = train_test_split(np.arange(n),
                                           train_size=int(train_frac * n),
                                           random_state=seed)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    y_plot = np.array([0, 1])

    return x_train, y_train, x_test, y_test, y_plot, n, d


# #############################################################################
# BAYESIAN INFERENCE (MCMC)
# #############################################################################

# Laplace prior PyMC3 model
def fit_mcmc_laplace(y, x, B, seed=100):
    with pm.Model() as _:  # as model
        # p = np.shape(x)[1]
        # # Laplace
        # b = pm.Gamma('b', alpha=1, beta=1)
        # beta = pm.Laplace('beta', mu=0, b=b, shape=p)
        # intercept = pm.Flat('intercept')
        # obs = pm.Bernoulli(
        #     'obs', logit_p=pm.math.dot(x, beta) + intercept, observed=y)
        trace = pm.sample(B, random_seed=seed, chains=4)

    beta_post = trace['beta']
    intercept_post = trace['intercept'].reshape(-1, 1)
    b_post = trace['b'].reshape(-1, 1)

    return beta_post, intercept_post, b_post


# #############################################################################
# APPLICATION OF THE BAYESIAN INFERENCE (MCMC)

# repeat 50 mcmc runs for different train test splits
def run_sparse_classification_mcmc(dataset):
    # Repeat over 50 reps
    rep = 50
    train_frac = 0.7
    B = 2000

    # Initialize
    x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_classification(
        train_frac, dataset, 100)

    beta_post = np.zeros((rep, 4 * B, d))
    intercept_post = np.zeros((rep, 4 * B, 1))
    b_post = np.zeros((rep, 4 * B, 1))
    times = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j
        x, y, x_test, y_test, y_plot, n, d = \
            load_train_test_sparse_classification(train_frac, dataset, seed)
        start = time.time()
        beta_post[j], intercept_post[j], b_post[j] = fit_mcmc_laplace(y, x, B,
                                                                      seed)
        end = time.time()
        times[j] = (end - start)

    print("{}: {} ({})".format(dataset, np.mean(times),
                               np.std(times) / np.sqrt(rep)))

    # Save posterior samples
    np.save("samples/beta_post_sparseclass_{}".format(dataset), beta_post)
    np.save("samples/intercept_post_sparseclass_{}".format(dataset),
            intercept_post)
    np.save("samples/b_post_sparseclass_{}".format(dataset), b_post)
    np.save("samples/times_sparseclass_{}".format(dataset), times)


# #############################################################################
# CONFORMAL PREDICTION
# #############################################################################

# Split method baseline
def conformal_split(alpha, y, x, x_test, seed=100):
    n = np.shape(y)[0]
    #  n_test = np.shape(x_test)[0]
    # Fit lasso to training set
    n_train = int(n / 2)
    ls = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5,
                              random_state=seed)
    ls.fit(x[0:n_train], y[0:n_train])
    resid = ls.predict_proba(x[n_train:])[:, 1]
    resid[y[n_train:] == 0] = 1 - resid[y[n_train:] == 0]
    resid = -np.log(
        np.clip(resid, 1e-6, 1 - 1e-6))  # clip for numerical stability
    k = int(np.ceil((n / 2 + 1) * (1 - alpha)))
    d = np.sort(resid)[k - 1]

    logp_test = -np.log(np.clip(ls.predict_proba(x_test), 1e-6, 1 - 1e-6))
    region_split = logp_test <= d

    return region_split


# Full method baseline
def conformal_full(alpha, y, x, x_test, C, seed=100):
    n = np.shape(y)[0]
    rank_cp = np.zeros(2)
    for y_new in (0, 1):
        x_aug = np.concatenate((x, x_test), axis=0)
        y_aug = np.append(y, y_new)
        ls = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                random_state=seed)
        ls.fit(x_aug, y_aug)
        resid = ls.predict_proba(x_aug)[:, 1]
        resid[y_aug == 0] = 1 - resid[y_aug == 0]
        resid = -np.log(resid)
        rank_cp[y_new] = np.sum(resid >= resid[-1]) / (n + 1)
    region_full = rank_cp > alpha
    return region_full


# #############################################################################
# CONFORMAL BAYESIAN
# #############################################################################

# CONFORMAL FROM MCMC SAMPLES (JAX IMPLEMENTATION)

# compute rank (un-normalized by n+1)

@jit
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

    # compute non-conformity score and sort
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

def run_sparse_classification_conformal(dataset):
    # Compute intervals
    # Load posterior samples
    beta_post = jnp.load(
        "samples/beta_post_sparseclass_{}.npy".format(dataset))
    intercept_post = jnp.load(
        "samples/intercept_post_sparseclass_{}.npy".format(dataset))

    # Initialize
    train_frac = 0.7
    x, y, x_test, y_test, y_plot, n, d = load_train_test_sparse_classification(
        train_frac, dataset, 100)

    alpha = 0.2
    rep = np.shape(beta_post)[0]
    n_test = np.shape(x_test)[0]

    coverage_cb = np.zeros((rep, n_test))
    coverage_bayes = np.zeros((rep, n_test))
    coverage_split = np.zeros((rep, n_test))
    coverage_full = np.zeros((rep, n_test))

    length_cb = np.zeros((rep, n_test))
    length_bayes = np.zeros((rep, n_test))
    length_split = np.zeros((rep, n_test))
    length_full = np.zeros((rep, n_test))

    p_bayes = np.zeros((rep, n_test))
    region_bayes = np.zeros((rep, n_test, 2))
    region_cb = np.zeros((rep, n_test, 2))
    region_split = np.zeros((rep, n_test, 2))
    region_full = np.zeros((rep, n_test, 2))

    times_bayes = np.zeros(rep)
    times_cb = np.zeros(rep)
    times_split = np.zeros(rep)
    times_full = np.zeros(rep)

    for j in tqdm(range(rep)):
        seed = 100 + j

        # load data
        x, y, x_test, y_test, y_plot, n, d = \
            load_train_test_sparse_classification(train_frac, dataset, seed)

        # Split conformal method
        start = time.time()
        region_split[j] = conformal_split(alpha, y, x, x_test, seed)
        for i in (range(n_test)):
            coverage_split[j, i] = region_split[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_split[j, i] = np.sum(region_split[j, i])
        end = time.time()
        times_split[j] = end - start

        # Full conformal method
        start = time.time()
        C = 1.
        for i in (range(n_test)):
            region_full[j, i] = conformal_full(alpha, y, x, x_test[i:i + 1], C,
                                               seed)
            coverage_full[j, i] = region_full[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_full[j, i] = np.sum(region_full[j, i])
        end = time.time()
        times_full[j] = end - start

        # ###########################
        @jit
        def logistic_loglikelihood(_y, _x):
            eta = (jnp.dot(beta_post[j], _x.transpose()) + intercept_post[j])
            B = np.shape(eta)[0]
            _n = np.shape(eta)[1]
            eta = eta.reshape(B, _n, 1)
            temp0 = np.zeros((B, _n, 1))
            logp = -jsp.special.logsumexp(
                jnp.concatenate((temp0, -eta), axis=2),
                axis=2)  # numerically stable
            log1p = -jsp.special.logsumexp(
                jnp.concatenate((temp0, eta), axis=2), axis=2)
            return _y * logp + (1 - _y) * log1p  # compute likelihood samples

        # ###########################

        # Bayes
        start = time.time()
        for i in (range(n_test)):
            p_bayes[j, i] = jnp.mean(
                jnp.exp(logistic_loglikelihood(1, x_test[i:i + 1])))
            # Compute region from p_bayes
            if p_bayes[j, i] > (1 - alpha):  # only y = 1
                region_bayes[j, i] = np.array([0, 1])
            elif (1 - p_bayes[j, i]) > (1 - alpha):  # only y = 0
                region_bayes[j, i] = np.array([1, 0])
            else:
                region_bayes[j, i] = np.array([1, 1])
            coverage_bayes[j, i] = region_bayes[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_bayes[j, i] = np.sum(region_bayes[j, i])
        end = time.time()
        times_bayes[j] = end - start

        # Conformal Bayes
        start = time.time()
        logp_samp_n = logistic_loglikelihood(y, x)
        logwjk = logistic_loglikelihood(y_plot.reshape(-1, 1, 1), x_test)
        for i in (range(n_test)):
            region_cb[j, i] = compute_cb_region_IS(
                alpha, logp_samp_n, logwjk[:, :, i])
            coverage_cb[j, i] = region_cb[
                j, i, np.argmin(np.abs(y_test[i] - y_plot))]
            length_cb[j, i] = np.sum(region_cb[j, i])
        end = time.time()
        times_cb[j] = end - start

    # Save regions (need to update)
    np.save("results/p_bayes_sparseclass_{}".format(dataset), p_bayes)
    np.save("results/region_bayes_sparseclass_{}".format(dataset),
            region_bayes)
    np.save("results/region_cb_sparseclass_{}".format(dataset), region_cb)
    np.save("results/region_split_sparseclass_{}".format(dataset),
            region_split)
    np.save("results/region_full_sparseclass_{}".format(dataset), region_full)

    np.save("results/coverage_bayes_sparseclass_{}".format(dataset),
            coverage_bayes)
    np.save("results/coverage_cb_sparseclass_{}".format(dataset), coverage_cb)
    np.save("results/coverage_split_sparseclass_{}".format(dataset),
            coverage_split)
    np.save("results/coverage_full_sparseclass_{}".format(dataset),
            coverage_full)

    np.save("results/length_bayes_sparseclass_{}".format(dataset),
            length_bayes)
    np.save("results/length_cb_sparseclass_{}".format(dataset), length_cb)
    np.save("results/length_split_sparseclass_{}".format(dataset),
            length_split)
    np.save("results/length_full_sparseclass_{}".format(dataset), length_full)

    np.save("results/times_bayes_sparseclass_{}".format(dataset), times_bayes)
    np.save("results/times_cb_sparseclass_{}".format(dataset), times_cb)
    np.save("results/times_split_sparseclass_{}".format(dataset), times_split)
    np.save("results/times_full_sparseclass_{}".format(dataset), times_full)


if __name__ == '__main__':
    __dataset = 'breast'  # 'parkinsons'
    os.makedirs('samples', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    run_sparse_classification_mcmc(__dataset)
    run_sparse_classification_conformal(__dataset)
