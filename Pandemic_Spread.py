"""
Computational Physics
Created on Fri Apr 30 15:55:41 2021
@author: lena, jan

https://docs.python.org/3/library/random.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit


def Task_1(T, S):  # death process

    # initial parameters
    N0 = 20  # time span
    r = 0.1  # decay rate
    x = np.arange(0, T+1)  # time axis

    # logistic model
    y_log = N_log(T, r, N0)

    # algorithm realizations
    Y_alg = []
    for s in range(S):
        random.seed(s)
        Y_alg.append(N_alg(T, r, N0))
    Y_alg = np.array(Y_alg)

    # plotting
    plot_models(T, S, N0, x, y_log, Y_alg,
                title='death process with rate $r={}$'.format(r),
                ylabel='number of particles $N$',
                ax2_ticks=[0, N0], ax2_ticklabels=['0', '$N_0$'])

    plt.yticks(np.arange(0, N0+2, step=np.round(N0/10)))  # only integers
    plt.show()


def N_log(T, r, N0):  # logistic model
    N = [N0]
    for t in range(T):
        N.append(N[t] - r*N[t])  # mean field eq: N(t+1) = N(t) + dN(t)/dt
    return np.array(N)


def N_alg(T, r, N0):  # algorithm realizations
    N = [N0]
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

    # initial parameters
    x = np.arange(0, T+1)  # time axis
    l = 1
    d = 0.1
    t_ex_mean = []
    t_ex_std = []
    t_extinction = []
    N0_range = range(10, 22+1)
    print('N_0 range:', N0_range[0], N0_range[-1])
    for N0 in N0_range:
        # logistic model
        y_log, t_ex_log = N3_log(T, N0, l, d)
        y_fill = np.zeros(T-t_ex_log)
        y_log = np.append(y_log, y_fill)

        # algorithm realizations
        Y_alg, t_ext = [], []
        for s in range(S):
            random.seed(s)
            N_alg, t_ex = N3_alg(T, N0, l, d)
            N_fill = np.zeros(T-t_ex)
            N_alg = np.append(N_alg, N_fill)
            Y_alg.append(N_alg)
            t_ext.append(t_ex)
        Y_alg = np.array(Y_alg)
        t_ext = np.array(t_ext)

        t_extinction.append(t_ext)
        t_ex_mean.append(t_ext.mean())
        t_ex_std.append(t_ext.std())

        # plotting
        # plot_models(T, S, N0, x, y_log, Y_alg,
        #             title='Verhulst extinction process with rates $l={}$, $d={}$'.format(
        #                 l, d),
        #             ylabel='number of particles $N$',
        #             ax2_ticks=[0, N0], ax2_ticklabels=['0', '$N_0$'])

        # plt.yticks(np.arange(0, N0+2, step=np.round(N0/10)))  # only integers

    plt.show()

    t_extinction = np.array(t_extinction)
    t_ex_mean = np.mean(t_extinction, axis=1)
    t_ex_std = np.std(t_extinction, axis=1)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(N0_range, t_extinction, alpha=.5,
            zorder=-1)
    l1, = plt.plot([], [], label=str(S)+' Markov samples',
                   c='tab:blue')  # for legend
    l2 = plt.errorbar(N0_range, t_ex_mean, yerr=t_ex_std,
                      fmt='o', c='k', ms=5, zorder=1, label='sample mean')

    # logistic fit
    t_ex_std[t_ex_std == 0] = 1  # replace zeros as we need 1/std for fitting
    popt, pcov = curve_fit(log, N0_range, t_ex_mean, p0=[.5, 19, 130],
                           sigma=t_ex_std, absolute_sigma=True)
    print('k, x0, N =', *popt)
    l3, = plt.plot(N0_range, log(N0_range, *popt), '--k', label='logistic fit')
    # l4, = plt.plot(N0_range, log(N0_range, 0.56, 19.5, 131.7),
    #                '--b', label='Lena\'s fit')

    plt.xlabel('$N_0$')
    plt.ylabel('mean extinction time $T_{ext}$')
    plt.title('Verhulst process mean extinction time')
    plt.legend(handles=[l1, l2, l3])

    ax2 = ax.twinx()  # secondary axis for additional labels
    ax2.set_ylim(ax.get_ylim())  # same ylim
    ax2.set_yticks([0, popt[2]])

    plt.show()


def log(x, k, x0, N):
    x = np.array(x)
    return N/(1+np.exp(k*(x-x0)))


def N3_log(T, N0, l, d):  # logistic model
    N = [N0]
    t_ex = T
    for t in range(T):
        # mean field eq: N(t+1) = N(t) + dN(t)/dt
        N_new = N[t] + l*N[t] - d*N[t]**2
        if N_new <= 0:
            t_ex = t
            break
        N.append(N_new)
    return np.array(N), t_ex


def N3_alg(T, N0, l, d):  # algorithm realizations
    N = [N0]
    t_ex = T
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        up = random.choices([2, 1], weights=[l, 1-l], k=N[t])
        down = random.choices([1, 0], weights=[d, 1-d], k=N[t]**2)
        # sum = new population size
        N_new = int(np.sum(up)-np.sum(down))
        if N_new <= 0:
            t_ex = t
            break
        N.append(N_new)
        # if t_ex == T:
        #     print('no extinction for N_0 =', N0)
    return np.array(N), t_ex


def Task_4(T, S):  # SIR model
    pass



def SIR_alg(T, S0, I0, R0, b, c):  # algorithm realizations
    S = [S0]
    I = [I0]
    R = [R0]
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        S_down = random.choices([0, 1], weights=[b, 1-b], k=S[t]*I[t])
        I_up = random.choices([2, 1], weights=[b, 1-b], k=S[t]*I[t])
        I_down = random.choices([1, 0], weights=[c, 1-c], k=I[t])
        R_up = random.choices([2, 1], weights=[c, 1-c], k=I[t])
        # sum = new population size
        S.append(int(np.sum(S_down)))
        I.append(int(np.sum(I_up)-int(np.sum(I_down))))
        R.append(int(np.sum(R_up)))
            
        
    return np.array(S), np.array(I), np.array(R)


def plot_models(T, S, N0, x, y_log, Y_alg, title, ylabel,
                ax2_ticks, ax2_ticklabels):
    """
    General plotting function.

    Parameters
    ----------
    T : int
        Time span.
    S : int
        Number of sample realizations.
    N0 : int
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
                      yerr=Y_alg_std, ecolor='gray', errorevery=3,
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
    # Task_2(T=600, S=300)

    Task_3(T=1000, S=1000)
    Task_4(T=10, S=10)
