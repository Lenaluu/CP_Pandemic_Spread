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
                ylabel='number of particles $N$',
                ax2_ticks=[0, No], ax2_ticklabels=['0', '$N_0$'])

    plt.yticks(np.arange(0, No+2, step=np.round(No/10)))  # only integers
    plt.show()


def N_log(T, r, No):  # logistic model
    N = [No]
    for t in range(T):
        N.append(N[t] - r*N[t])  # mean field eq: N(t+1) = N(t) + dN(t)/dt
    return np.array(N)


def N_alg(T, r, No):  # algorithm realizations
    N = [No]
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        num = random.choices([0, 1], weights=[r, 1-r], k=N[t])
        # sum = new population size
        N.append(int(np.sum(num)))
    return np.array(N)


def Task_2(T, S):  # gene expression

    # initial parameters
    Mo = 20  # M(0)
    Po = 20  # P(0)
    l_m = 1
    d_m = 0.2
    l_p = 1
    d_p = 0.02

    x = np.arange(0, T+1)  # time axis

    # logistic model
    y_log, t_break = P_log(T, Po, Mo, l_m, d_m, l_p, d_p)

    # algorithm realizations
    Y_alg = []
    for s in range(S):
        random.seed(s)
        Y_alg.append(P_alg(T, d_m, d_p, Mo, Po))
    Y_alg = np.array(Y_alg)

    # plotting
    plot_models(T, S, Po, x, y_log, Y_alg,
                title='gene expression for a single cell',
                ylabel='number of protein particles $P$',
                ax2_ticks=[0, Po, y_log[-1]],
                ax2_ticklabels=['0', '$P_0$', '$P_e$'])

    plt.show()

    w_P = np.array([Y_alg[s][t_break:] for s in range(S)]).flatten()
    P = np.array(range(w_P.min(), w_P.max()))
    Pm = l_m*l_p/(d_m*d_p)
    sigma2 = Pm*(1+l_p/(d_m+d_p))
    w_P_f = 1/(np.sqrt(2*np.pi*sigma2)) * np.exp(-(P-Pm)**2/(2*sigma2))

    plt.figure()
    plt.hist(w_P, bins=int(len(P)), density=True, histtype='step',
             label='P')
    plt.plot(P, w_P_f)
    plt.xlabel('$P$')
    plt.ylabel('$w(P)$')
    plt.title('equilibrium distribution density of $P$')
    plt.show()

    plt.figure()
    plt.hist(w_P, bins=int(len(P)), density=True, histtype='step',
             label='P')
    plt.plot(P, w_P_f)
    plt.yscale('log')
    plt.xlabel('$P$')
    plt.ylabel('$w(P)$')
    plt.title('equilibrium distribution density of $P$\n logarithmic scale')
    plt.show()

    skew = np.sum(((w_P-np.mean(w_P))/np.std(w_P))**3)/w_P.size
    kurt = np.sum(((w_P-np.mean(w_P))/np.std(w_P))**4)/w_P.size
    print('skewness = '+str(skew)+'; kurtosis = '+str(kurt))


def M_log(T, Mo, l_m, d_m):  # logistic model
    M = [Mo]
    for t in range(T):
        M.append(M[t] + l_m - d_m*M[t])  # mean field eq
    return np.array(M)


def P_log(T, Po, Mo, l_m, d_m, l_p, d_p):  # logistic model
    P = [Po]
    i = 0
    t_break = int(.9*T)
    for t in range(T):
        M = M_log(T, Mo, l_m, d_m)
        dP = l_p*M[t] - d_p*P[t]
        P.append(P[t] + dP)  # mean field eq
        if dP < 1e-3 and i == 0:
            print(t)
            t_break = t
            i = 1
    return np.array(P), t_break


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


def plot_models(T, S, No, x, y_log, Y_alg, title, ylabel,
                ax2_ticks, ax2_ticklabels):
    """
    General plotting function.

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
    l3 = plt.errorbar(x, Y_alg_mean, fmt='o', c='k', ms=5,  # yerr=[],
                      yerr=Y_alg_std, ecolor='gray', errorevery=3,  # zorder=-2,
                      label=r'$\left< N(t) \right>$ mean+std of '
                            + str(S)+' Markov realizations')
    # layout
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(handles=[l1, l2, l3])
    ax2 = ax.twinx()  # secondary axis for additional labels
    ax2.set_ylim(ax.get_ylim())  # same ylim
    ax2.set_yticks(ax2_ticks)
    ax2.set_yticklabels(ax2_ticklabels)


if __name__ == '__main__':
    # Task_1(T=60, S=10)
    Task_2(T=600, S=300)
    Task_3(T=10, S=10)
    Task_4(T=10, S=10)
