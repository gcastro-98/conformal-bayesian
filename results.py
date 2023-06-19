import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

examples = ['sparsereg_diabetes', 'sparsereg_diabetes_misspec',
            'sparsereg_boston', 'sparsereg_boston_misspec',
            'sparseclass_breast', 'sparseclass_parkinsons']

methods_tot = ['bayes', 'cb', 'split', 'full']


def report_mcmc_times() -> None:
    for example in examples:
        suffix = example
        times = np.load("samples/times_{}.npy".format(suffix))
        rep = np.shape(times)[0]
        print("{} MCMC time: {:.3f} ({:.3f})".format(
            suffix, np.mean(times), np.std(times)/np.sqrt(rep)))
    print()


def report_results() -> None:
    for example in examples:
        methods = methods_tot

        print('EXAMPLE: {}'.format(example))
        for method in methods:
            suffix = method + '_' + example

            # Coverage (take mean over test values)

            coverage = np.mean(np.load(
                "results/coverage_{}.npy".format(suffix)), axis=1)
            rep = np.shape(coverage)[0]
            mean = np.mean(coverage)
            se = np.std(coverage)/np.sqrt(rep)
            print("{} coverage is {:.3f} ({:.3f})".format(
                method, mean, se))

            # Return exact coverage if cb
            if method == 'cb' and (example not in ['sparseclass_breast',
                                                   'sparseclass_parkinsons']):
                suffix_ex = method + '_exact_' + example
                coverage = np.mean(np.load("results/coverage_{}.npy".format(
                    suffix_ex)), axis=1)  # take mean over test values
                rep = np.shape(coverage)[0]
                mean = np.mean(coverage)
                se = np.std(coverage)/np.sqrt(rep)
                print("{} exact coverage is {:.3f} ({:.3f})".format(
                    method, mean, se))
        print()

        for method in methods:
            suffix = method + '_' + example
            # Length
            length = np.mean(np.load(
                "results/length_{}.npy".format(suffix)), axis=1)
            rep = np.shape(length)[0]
            mean = np.mean(length)
            se = np.std(length)/np.sqrt(rep)
            print("{} length is {:.2f} ({:.2f})".format(method, mean, se))
        print()

        for method in methods:
            suffix = method + '_' + example
            # Length
            times = np.load("results/times_{}.npy".format(suffix))
            rep = np.shape(times)[0]
            mean = np.mean(times)
            se = np.std(times)/np.sqrt(rep)
            print("{} times is {:.3f} ({:.3f})".format(method, mean, se))
        print()


def report_missclassification_rates() -> None:
    for example in examples:
        methods = methods_tot
        # print misclassification/both/empty
        if example in ['sparseclass_breast', 'sparseclass_parkinsons']:
            for method in methods[0:2]:
                suffix = method + '_' + example
                coverage = np.load("results/coverage_{}.npy".format(suffix))
                length = np.load("results/length_{}.npy".format(suffix))
                rep = np.shape(coverage)[0]
                n_tot = np.sum(length == 1, axis=1)
                n_misclass = np.sum(
                    np.logical_and(length == 1, coverage == 0), axis=1)
                misclass_rate = n_misclass/n_tot
                both_rate = np.mean(length == 2, axis=1)
                empty_rate = np.mean(length == 0, axis=1)

                print('{} misclassification rate is {:.3f} ({:.3f})'.format(
                    method, np.mean(misclass_rate),
                    np.std(misclass_rate)/np.sqrt(rep)))
                print('{} both rate is {:.3f} ({:.3f})'.format(
                    method, np.mean(both_rate),
                    np.std(both_rate)/np.sqrt(rep)))
                print('{} empty rate is {:.3f} ({:.3f})'.format(
                    method, np.mean(empty_rate),
                    np.std(empty_rate)/np.sqrt(rep)))


def plot_sparse_classification_results() -> None:
    _ = plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    dataset = 'breast'
    length_cb = np.load("results/length_cb_sparseclass_{}.npy".format(dataset))
    p_bayes = np.load("results/p_bayes_sparseclass_{}.npy".format(dataset))
    # rep = 50
    sns.histplot(p_bayes[np.where(length_cb == 0)], label='CB length = 0',
                 color='darkred', stat='density')
    sns.histplot(p_bayes[np.where(length_cb == 1)], label='CB length = 1',
                 stat='density')
    plt.xlabel(r'$p(y_i = 1 \mid x_i,Z)$')
    plt.legend()
    plt.title('Breast cancer dataset')

    plt.subplot(1, 2, 2)
    dataset = 'parkinsons'
    length_cb = np.load("results/length_cb_sparseclass_{}.npy".format(dataset))
    p_bayes = np.load("results/p_bayes_sparseclass_{}.npy".format(dataset))
    # rep = 50
    sns.histplot(p_bayes[np.where(length_cb == 0)], label='CB length = 0',
                 color='darkred', stat='density')
    sns.histplot(p_bayes[np.where(length_cb == 1)], label='CB length = 1',
                 stat='density')
    plt.xlabel(r'$p(y_i = 1 \mid x_i,Z)$')
    plt.legend()
    plt.title('Parkinsons dataset')
    plt.savefig('plots/sparse_classification.png', dpi=250)


if __name__ == '__main__':
    os.makedirs('plots', exist_ok=True)
    plot_sparse_classification_results()
    report_mcmc_times()
    report_results()
    report_missclassification_rates()
