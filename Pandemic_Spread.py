"""
Computational Physics
Created on Fri Apr 30 15:55:41 2021
@author: lena, jan

https://docs.python.org/3/library/random.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def Task_1(T, S):  # death process

    # initial parameters
    No = 20  # time span
    r = 0.1  # decay rate
    x = np.arange(0, T+1)  # time axis

    # logistic model
    y_log = N_log(T, r, No)

    # algorithm realizations
    Y_alg = []
    for s in range(S):
        random.seed(s)
        Y_alg.append(N_alg(T, r, No))
    Y_alg = np.array(Y_alg)

    # plotting
    plot_models(T, S, No, x, y_log, Y_alg,
                title='death process with rate $r={}$'.format(r),
                ylabel='number of particles $N$')

    plt.yticks(np.arange(0, No+2, step=np.round(No/10)))  # only integers
    plt.show()


def N_log(T, r, No):  # logistic model
    N = [No]
    for t in range(T):
        N.append(N[-1] - r*N[-1])  # mean field eq: N(t+1) = N(t) + dN(t)/dt
    return np.array(N)


def N_alg(T, r, No):  # algorithm realizations
    N = [No]
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        num = random.choices([0, 1], weights=[r, 1-r], k=N[-1])
        # sum = new population size
        N.append(int(np.sum(num)))
    return np.array(N)


def Task_2(T, S):  # gene expression

    # initial parameters
    Mo = 20  # M(0)
    Po = 20  # P(0)
    x = np.arange(0, T+1)  # time axis

    # logistic model
    y_log = P_log(T, Po, Mo)

    # algorithm realizations
    Y_alg = np.zeros((S, T+1))

    # plotting
    plot_models(T, S, Po, x, y_log, Y_alg,
                title='gene expression for a single cell',
                ylabel='number of protein particles $P$')
    plt.show()


def M_log(T, Mo):  # logistic model
    M = [Mo]
    lambda_m = 1
    d_m = 0.2
    for t in range(T):
        M.append(M[-1] + lambda_m - d_m*M[-1])  # mean field eq
    return np.array(M)


def P_log(T, Po, Mo):  # logistic model
    P = [Po]
    lambda_p = 1
    d_p = 0.02
    for t in range(T):
        M = M_log(T, Mo)
        P.append(P[-1] + lambda_p*M[t] - d_p*P[-1])  # mean field eq
    return np.array(P)

def P_alg(T, d_m, d_p, Mo, Po):  # algorithm realizations
    M = [Mo]
    P = [Po]
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        M_down = random.choices([0, 1], weights=[d_m, 1-d_m], k=M[t])
        P_down = random.choices([0, 1], weights=[d_p, 1-d_p], k=P[t])
        # sum = new population siz
        M.append(int(np.sum(M_down)+1))
        P.append(int(np.sum(P_down)+M[t]))
    return np.array(P)


def Task_3(T, S):  # Verhulst extinction
    pass


def Task_4(T, S):  # SIR model
    pass


def plot_models(T, S, No, x, y_log, Y_alg, title, ylabel):  # general plotting
    """
    Parameters
    ----------
    T : int
        Time span.
    S : int
        Number of sample realizations.
    No : int
        Initial number of particles N_0.
    y_log : 1D array
        logistic model from mean-field equations.
    Y_alg : 2D array
        Gillespie algorithm realizations from Marcov process.
    title : str
        Plot title.
    """

    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    # logistic model
    l1, = plt.plot(x, y_log, lw=2, c='k',
                   label='mean-field equations (logistic model)')
    # algorithm realizations
    for s in range(S):
        l2, = plt.plot(x, Y_alg[s], 'o-', ms=2, zorder=-1,
                       label='Marcov process (Gillespie algorithm)')
    # algorithm mean+std
    Y_alg_mean = np.mean(Y_alg, axis=0)
    Y_alg_std = np.std(Y_alg, axis=0)
    l3 = plt.errorbar(x, Y_alg_mean, fmt='o', c='k', ms=5, yerr=Y_alg_std,
                      label=r'$\left< N(t) \right>$ mean+std of '
                            + str(S)+' Markov realizations')
    # layout
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(handles=[l1, l2, l3])
    ax2 = ax.twinx()  # secondary axis for additional labels
    ax2.set_ylim(ax.get_ylim())  # same ylim
    ax2.set_yticks([0, No])
    ax2.set_yticklabels(['0', '$N_0$'])


if __name__ == '__main__':
<<<<<<< Updated upstream
    # Task_1(T=60, S=10)
    Task_2(T=300, S=10)
=======
    Task_1(T=50, S=10)
    Task_2(T=10, S=10)
>>>>>>> Stashed changes
    Task_3(T=10, S=10)
    Task_4(T=10, S=10)
