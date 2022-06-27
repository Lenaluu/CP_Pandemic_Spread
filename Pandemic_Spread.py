"""
Computational Physics
Created on Fri Apr 30 15:55:41 2021
@author: lena, jan

https://docs.python.org/3/library/random.html
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import sys

import random
from scipy.optimize import curve_fit
import scipy.integrate as integrate


def Task_1(T, S):  # death process

    # initial parameters
    N0 = 20  # starting population number
    r = 0.1  # decay rate
    rates = [{"1" : r}]
    t_log = np.arange(0, T+1)  # time axis

    # logistic model
    y_log = N_log(T, r, N0)

    # algorithm realizations
    Y_alg = []
    t_alg = []
    for s in range(S):
        t, N = Gillespie_N_alg(T, rates, N0, s)
        t_alg.append(t)
        Y_alg.append(N)
    t_alg = np.array(t_alg)
    Y_alg = np.array(Y_alg)
    
    # plotting
    fig_T1, ax_T1 = plot_models(T, S, N0, t_log, y_log,t_alg, Y_alg,
                                title='death process with rate $r={}$'.format(r),
                                ylabel='number of particles $N$',
                                ax2_ticks=[0, N0], ax2_ticklabels=['0', '$N_0$'])

    ax_T1.set_yticks(np.arange(0, N0+2, step=np.round(N0/10)))  # only integers
    plt.show()
    fig_T1.savefig('Task1.pdf', bbox_inches='tight', dpi=300, transparent= False)


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


def Rates_to_list(rates, N):
    """
    taking "expr" to calculate rates of current state
    current_rate = expr*value
    Parameters
    ----------
    rates : list of dictionaries of form {"expr":value}
            each dictionary is representing a mean field equation
    N : tuple
        of numbers representing current state 
    """
    
    rate_vals = []
    for dic in rates:
        dic_vals = []
        for i in list(dic):
            expr = eval(i)
            dic_vals.append(expr*dic[i])
        rate_vals.append(dic_vals)
        
    return rate_vals
   
    
def Gillespie_N_alg(T,rates,N0,s=None):
    """
    Gillespie algorithm for Task 1
    Parameters
    ----------
    T : int
        max. time
    rates : list of dictionaries 
            of form {"expr":value}
    N0 : int or tuple
        numbers representing initial state 
    s : int
        seed for random number generator
    """

    t = [0]
    N = [N0]
    
    rng = np.random.default_rng(seed=s)
    
    while t[-1] < T:
        
        rate_list = Rates_to_list(rates,N[-1])[0]
        rate_sum = np.sum(rate_list)
        
        tau = rng.exponential(scale=1/rate_sum)
        t.append(t[-1]+tau)
        
        
        # rates
        # # delta_N = rate weighted choice if 1 individuum dies
        # delta_N = random.choices([-1, 0], weights=[rates[0], 1-rates[0]], k=1)
        # new population number
        N.append(N[-1]-1)
        if N[-1]== 0:
            break
       
    return t, N
        
        

    

def Task_2(T, S):  # gene expression

    # initial parameters
    Mo = 20  # M(0)
    Po = 20  # P(0)
    N0 = (Mo,Po)
    l_m = 1
    d_m = 0.2
    l_p = 1
    d_p = 0.02
    rates = [{"1" : l_m,
             "N[0]" : d_m},
             {"N[0]" : l_p,
             "N[1]" : d_p}]
    x = np.arange(0, T+1)  # time axis

    # logistic model
    y_log, t_break = P_log(T, Po, Mo, l_m, d_m, l_p, d_p)
    
    # algorithm realizations
    t_alg, Y_alg = [], []
    max_len = 0
    for s in range(S):
        #random.seed(s)
        Gillespie_P_alg
        t, P = Gillespie_P_alg(T, rates, N0, s)
        t_alg.append(t)
        Y_alg.append(P)
        if len(P) > max_len:
            max_len = len(P)
    
    t_equal, Y_equal = [], []
    for idx in range(len(Y_alg)):
        t_fill = list(np.full(max_len-len(t_alg[idx]),np.nan))
        t_row = t_alg[idx]+t_fill

        N_fill = list(np.full(max_len-len(Y_alg[idx]),np.nan))
        N_row = Y_alg[idx]+N_fill
        
        t_equal.append(t_row)
        Y_equal.append(N_row)
        
    t_alg = np.array(t_equal)   
    Y_alg = np.array(Y_equal)
    # plotting
    fig_T2, ax_T2 = plot_models(T, S, Po, x, y_log, t_alg, Y_alg,
                                title='gene expression for a single cell',
                                ylabel='number of protein particles $P$',
                                ax2_ticks=[0, Po, y_log[-1]],
                                ax2_ticklabels=['0', '$P_0$', '$P_e$'])
    
    plt.show()
    fig_T2.savefig('Task2.pdf', bbox_inches='tight', dpi=300, transparent= False)

    w_P = np.array([Y_alg[s][t_break:] for s in range(S)]).flatten()
    P = np.array(range(w_P.min(), w_P.max()))
    Pm = l_m*l_p/(d_m*d_p)
    sigma2 = Pm*(1+l_p/(d_m+d_p))
    w_P_f = 1/(np.sqrt(2*np.pi*sigma2)) * np.exp(-(P-Pm)**2/(2*sigma2))
    
    fig_T2_dens, ax_T2_dens = plt.subplots(1,dpi=150)
    ax_T2_dens.hist(w_P, bins=int(len(P)), density=True, histtype='step',
             label='P')
    ax_T2_dens.plot(P, w_P_f)
    ax_T2_dens.set(xlabel= '$P$',
                   ylabel= '$w(P)$',
                   title= 'equilibrium distribution density of $P$')
    fig_T2_dens.savefig('Task2_density.pdf', bbox_inches='tight', dpi=300, transparent= False)

    fig_T2_dens_log, ax_T2_dens_log = plt.subplots(1,dpi=150)

    ax_T2_dens_log.hist(w_P, bins=int(len(P)), density=True, histtype='step',
             label='P')
    ax_T2_dens_log.plot(P, w_P_f)
    ax_T2_dens_log.set(yscale= 'log',
                       xlabel= '$P$',
                       ylabel= '$w(P)$',
                       title= 'equilibrium distribution density of $P$\n logarithmic scale')
    
    fig_T2_dens_log.savefig('Task2_density_log.pdf', bbox_inches='tight', dpi=300, transparent= False)

    skew = np.sum(((w_P-np.mean(w_P))/np.std(w_P))**3)/w_P.size
    kurt = np.sum(((w_P-np.mean(w_P))/np.std(w_P))**4)/w_P.size
    txt_skew_kurt =f'skewness = {skew}\nkurtosis = {kurt}'
    print(txt_skew_kurt)
    txt_T2 = open("Task2 skewness+kurtosis.txt", "w")
    txt_T2.write(txt_skew_kurt)
    txt_T2.close()
    


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
            print(f'break after {t} steps')
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
        # sum = new population size
        M.append(int(np.sum(M_down)+1))
        P.append(int(np.sum(P_down)+M[t]))
    return np.array(P)

def Gillespie_P_alg(T,rates,N0,s=None):
    """
    Gillespie algorithm for Task 2
    Parameters
    ----------
    T : int
        max. time
    rates : list of dictionaries 
            of form {"expr":value}
    N0 : tuple
         numbers representing initial state 
    s : int
        seed for random number generator
    """
    
    t = [0]
    N = [N0]
    P = [N0[0]]
    rng = np.random.default_rng(seed=s)
    i=0
    while t[-1] < T:
        N_now = N[-1]
        rates_list = Rates_to_list(rates,N_now)
        
        rates_flat = np.array(rates_list).flatten()
        rate_sum = np.sum(rates_flat)
        
        tau = rng.exponential(scale=1/rate_sum)
        t.append(t[-1]+tau)
        
        # pick random number to determine event thats happening
        """
        https://arxiv.org/pdf/2112.05293.pdf
        direct method
        """
        event_prob = rates_flat/rate_sum
        event_number = rng.choice(len(rates_flat),p=event_prob)
        if event_number == 0:
            N_next = (N_now[0]+1,N_now[1])
        elif event_number == 1:
            N_next = (N_now[0]-1,N_now[1])
        elif event_number == 2:
            N_next = (N_now[0],N_now[1]+1)
        elif event_number == 3:
            N_next = (N_now[0],N_now[1]-1)
        
        N.append(N_next)
        P.append(N_next[0])
        i+=1
    return t, P

def Task_3(T, S):  # Verhulst extinction

    # initial parameters
    x = np.arange(0, T+1)  # time axis
    l = 1
    d = 0.1
    rates = [{"N[0]" : l,
             "N[0]**2" : d}]
    
    t_ex_mean = []
    t_ex_std = []
    t_extinction = []
    N0_range = range(10, 22+1)
    range_txt = f'N_0 range: {N0_range[0]}:{N0_range[-1]}'
    print(range_txt)
    
    txt_T3 = open("Task3.txt", "w")
    txt_T3.write(range_txt+'\n')
    for N0 in N0_range:
        print(f"start value: {N0}")
        # logistic model
        y_log, t_ex_log = N3_log(T, N0, l, d)
        y_fill = np.zeros(T-t_ex_log)
        y_log = np.append(y_log, y_fill)

        # algorithm realizations
        t_alg, Y_alg, t_ext = [], [], []
        max_len = 0
        i= 1
        for s in range(S):
            t, N, t_ex = Gillespie_N3_alg(T,rates,N0,s=s)
            # N_alg, t_ex = N3_alg(T, N0, l, d)
            # N_fill = np.zeros(T-t_ex)
            # N_alg = np.append(N_alg, N_fill)
            t_alg.append(t)
            Y_alg.append(N)
            t_ext.append(t_ex)
            if i % 5 == 0:
                print(f"Sample {i}")
            i+=1
            if len(N) > max_len:
                max_len = len(N)
        
                
        
        t_equal, Y_equal = [], []
        for idx in range(len(Y_alg)):
            t_fill = list(np.full(max_len-len(t_alg[idx]),np.nan))
            t_row = t_alg[idx]+t_fill
    
            N_fill = list(np.full(max_len-len(Y_alg[idx]),np.nan))
            N_row = Y_alg[idx]+N_fill
            
            t_equal.append(t_row)
            Y_equal.append(N_row)
          
        t_alg = np.array(t_equal)   
        Y_alg = np.array(Y_equal)
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

    

    t_extinction = np.array(t_extinction)
    t_ex_mean = np.mean(t_extinction, axis=1)
    t_ex_std = np.std(t_extinction, axis=1)

    fig_T3_extinct, ax_T3 = plt.subplots(figsize=(8.4, 4.8))
    ax_T3.plot(N0_range, t_extinction, alpha=.5,
            zorder=-1)
    l1, = ax_T3.plot([], [], label=str(S)+' Markov samples',
                   c='tab:blue')  # for legend
    l2 = ax_T3.errorbar(N0_range, t_ex_mean, yerr=t_ex_std,
                      fmt='o', c='k', ms=5, zorder=1, label='sample mean')

    # logistic fit
    t_ex_std[t_ex_std == 0] = 1  # replace zeros as we need 1/std for fitting
    popt, pcov = curve_fit(log, N0_range, t_ex_mean, p0=[.5, 19, 130],
                           sigma=t_ex_std, absolute_sigma=True)
    
    fit_txt = f'k, x0, N = {popt}'
    print(fit_txt)
    txt_T3.write(fit_txt)
    txt_T3.close()
    
    l3, = ax_T3.plot(N0_range, log(N0_range, *popt), '--k', label='logistic fit')
    # l4, = ax_T3.plot(N0_range, log(N0_range, 0.56, 19.5, 131.7),
    #                '--b', label='Lena\'s fit')

    ax_T3.set(xlabel= '$N_0$',
              ylabel= 'mean extinction time $T_{ext}$',
              title= 'Verhulst process mean extinction time')
    ax_T3.legend(handles=[l1, l2, l3])

    ax_T3_2 = ax_T3.twinx()  # secondary axis for additional labels
    ax_T3_2.set(ylim= ax_T3.get_ylim(),  # same ylim
                yticks=[0, popt[2]])
    fig_T3_extinct.savefig('Task3_extinct.pdf', bbox_inches='tight', dpi=300, transparent= False)



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

def Gillespie_N3_alg(T,rates,N0,s=None):
    """
    Gillespie algorithm for Task 3
    Parameters
    ----------
    T : int
        max. time
    rates : list of dictionaries 
            of form {"expr":value}
    N0 : tuple
         numbers representing initial state 
    s : int
        seed for random number generator
    """
    
    t = [0]
    N = [N0]
    rng = np.random.default_rng(seed=s)
    t_ex = -1
    i=0
    while t[-1] < T:
        N_now = N[-1]
        rates_list = Rates_to_list(rates,(N_now,))
        
        rates_flat = np.array(rates_list).flatten()
        rate_sum = np.sum(rates_flat)
        
        tau = rng.exponential(scale=1/rate_sum)
        t.append(t[-1]+tau)
        
        # pick random number to determine event thats happening
        event_prob = rates_flat/rate_sum
        event_number = rng.choice(len(rates_flat),p=event_prob)
        if event_number == 0:
            N_next = N_now+1
        elif event_number == 1:
            N_next = N_now-1
        
        N.append(N_next)
        if N_next <= 0:
            t_ex = t[-1]
            break
            
        i+=1
    return t, N, t_ex


def Task_4(T, S):  # SIR model

    # initial parameters
    b = 0.02  # infection
    c = 0.3   # recovery
    S0 = 30
    I0 = 1
    R0 = 0
    x = np.arange(0, T+1)  # time axis
    N0 = (S0,I0,R0)
    rates =  [{"N[0]*N[1]" : b,
               "N[1]" : c}]
    
    # logistic model
    Sus_log, Inf_log, Rec_log = SIR_log(T, S0, I0, R0, b, c)

    # algorithm realizations
    t_alg, Sus_alg, Inf_alg, Rec_alg = [], [], [], []
    max_len = 0
    for s in range(S):
        random.seed(s)
        # Sus, Inf, Rec = SIR_alg(T, S0, I0, R0, b, c)
        t, Sus, Inf, Rec = Gillespie_SIR_alg(T,rates,N0,s=None)
        t_alg.append(t)
        Sus_alg.append(Sus)
        Inf_alg.append(Inf)
        Rec_alg.append(Rec)
        if len(t) > max_len:
            max_len = len(t)
    
    t_equal, Sus_equal, Inf_equal, Rec_equal = [], [], [], []
    for idx in range(len(t_alg)):
        t_fill = list(np.full(max_len-len(t_alg[idx]),np.nan))
        t_row = t_alg[idx]+t_fill

        Sus_fill = list(np.full(max_len-len(Sus_alg[idx]),np.nan))
        Sus_row = Sus_alg[idx]+Sus_fill
        Inf_fill = list(np.full(max_len-len(Inf_alg[idx]),np.nan))
        Inf_row = Inf_alg[idx]+Inf_fill
        Rec_fill = list(np.full(max_len-len(Rec_alg[idx]),np.nan))
        Rec_row = Rec_alg[idx]+Rec_fill
        
        t_equal.append(t_row)
        Sus_equal.append(Sus_row)
        Inf_equal.append(Inf_row)
        Rec_equal.append(Rec_row)
        
    t_alg = np.array(t_equal)    
    Sus_alg = np.array(Sus_equal)
    Inf_alg = np.array(Inf_equal)
    Rec_alg = np.array(Rec_equal)

    # plotting
    fig_T4, ax_T4 = plot_models(T, S, I0, x, Inf_log, t_alg, Inf_alg,
                                title='SIR model: Infections',
                                ylabel='number of infected persons $I$',
                                ax2_ticks=[0, I0, Inf_log[-1]],
                                ax2_ticklabels=['0', '$I_0$', '$I_e$'])
    fig_T4.savefig('Task4.pdf', bbox_inches='tight', dpi=300, transparent= False)



def SIR_log(T, S0, I0, R0, b, c):  # logistic model
    Sus = [S0]
    Inf = [I0]
    Rec = [R0]
    P = S0+I0+R0
    for t in range(T):
        sus = Sus[t] - b*Sus[t]*Inf[t]
        inf = Inf[t] + b*Sus[t]*Inf[t] - c*Inf[t]
        rec = Rec[t] + c*Inf[t]
        sus_new = sus if sus > 0 else 0
        sus_new = sus_new if sus_new < P else P
        inf_new = inf if inf > 0 else 0
        inf_new = inf_new if inf_new < P else P
        rec_new = rec if rec > 0 else 0
        rec_new = rec_new if rec_new < P else P
        Sus.append(sus_new)
        Inf.append(inf_new)
        Rec.append(rec_new)
    return np.array(Sus), np.array(Inf), np.array(Rec)


def SIR_alg(T, S0, I0, R0, b, c):  # algorithm realizations
    S = [S0]
    I = [I0]
    R = [R0]
    P = S0+I0+R0
    t_ex = T
    for t in range(T):
        # num = k-sized list of population elements chosen with replacement
        prob = b*I[t] if b*I[t] < 1 else 1  # TODO
        S_down = random.choices([1, 0], weights=[prob, 1-prob], k=S[t])
        I_down = random.choices([1, 0], weights=[c, 1-c], k=I[t])
        # sum = new population size
        S.append(int(S[t]-np.sum(S_down)))
        I.append(int(I[t]+int(np.sum(S_down))-int(np.sum(I_down))))
        R.append(R[t]+int(np.sum(I_down)))
        if I[t+1] <= 0:
            t_ex = t+1
            break

    N_fill = np.zeros(T-t_ex)  # TODO
    S = np.append(S, N_fill)
    I = np.append(I, N_fill)
    R = np.append(R, N_fill)
    return np.array(S), np.array(I), np.array(R)

def Gillespie_SIR_alg(T,rates,N0,s=None):
    """
    Gillespie algorithm for Task 4
    Parameters
    ----------
    T : int
        max. time
    rates : list of dictionaries 
            of form {"expr":value}
    N0 : tuple
         numbers representing initial state 
    s : int
        seed for random number generator
    """
    
    t = [0]
    S = [N0[0]]
    I = [N0[1]]
    R = [N0[2]]
    rng = np.random.default_rng(seed=s)
    
    while t[-1] < T:
        N_now = (S[-1], I[-1], R[-1])
        rates_list = Rates_to_list(rates,N_now)
        
        rates_flat = np.array(rates_list).flatten()
        rate_sum = np.sum(rates_flat)
        
        tau = rng.exponential(scale=1/rate_sum)
        t.append(t[-1]+tau)
        
        # pick random number to determine event thats happening
        """
        https://arxiv.org/pdf/2112.05293.pdf
        direct method
        """
        
        event_prob = rates_flat/rate_sum
        event_number = rng.choice(len(rates_flat),p=event_prob)
        if event_number == 0:
            S.append(S[-1]-1)
            I.append(I[-1]+1)
            R.append(R[-1])
        elif event_number == 1:
            S.append(S[-1])
            I.append(I[-1]-1)
            R.append(R[-1]+1)
        else:
            sys.exit("too much rates")
        if I[-1] <= 0:
            break

    return t, S, I, R

def plot_models(T, S, N0, t_log, y_log, t_alg, Y_alg, title, ylabel,
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
    t_alg : 2D array
        times of Gillespie algorithm realizations from Marcov process.
    Y_alg : 2D array
        Population values of Gillespie algorithm realizations from Marcov process.
    title : str
        Plot title.
    """

    fig, ax = plt.subplots(figsize=(8.4, 4.8),dpi=150)

    # logistic model
    l1, = plt.plot(t_log, y_log, lw=2, c='k',
                   label='mean-field equations (logistic model)')

    # algorithm realizations
    Oranges = pl.cm.Oranges(np.linspace(0.1,1,S))
    j=0
    for s in range(S):
        l2, = plt.plot(t_alg[s], Y_alg[s], 'o-', ms=2, zorder=-1,
                       color= Oranges[j],
                       label='Marcov process (Gillespie algorithm)')
        j+=1


    # algorithm mean+std
    t_alg_mean = np.nanmean(t_alg, axis=0)
    t_alg_std = np.nanstd(t_alg, axis=0)    
    Y_alg_mean = np.nanmean(Y_alg, axis=0)
    Y_alg_std = np.nanstd(Y_alg, axis=0)
    l3 = plt.errorbar(t_alg_mean, Y_alg_mean, fmt= 'o', c= 'k', ms= 5,  # yerr=[],
                      xerr= t_alg_std, yerr= Y_alg_std, ecolor= 'gray', errorevery= 3,
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
    
    return fig, ax

if __name__ == '__main__':
    # Task_1(T=800, S=30)
    # Task_2(T=60, S=30)#S=300
    Task_3(T=2000, S=10)#T=1000, S=1000
    # Task_4(T=100, S=500)#T=60, S=500
    pass
