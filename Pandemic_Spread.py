"""
Computational Physics
Created on Fri Apr 30 15:55:41 2021
@author: lena, jan

https://docs.python.org/3/library/random.html
"""

import numpy as np
import matplotlib.pyplot as plt
import random

######### Task 1 ##########

def Task_1(T, No=20, r=0.1):
    for s in range(3):
        random.seed(s)
        plot_models(T, No, r, 'death process')

def plot_models(T, No, r, title):
    x = np.arange(0, T+1)
    y_log = np.array(N_log(T, r, No))
    y_alg = np.array(N_alg(T, r, No))
    
    fig, ax = plt.subplots()
    plt.plot(x, y_log, c='tab:blue', lw=2,
             label='mean-field equations (logarithmic model)')
    plt.plot(x, y_alg, 'o', c='tab:orange', zorder=-1,
             label='Marcov process (Gillespie algorithm)')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title(title)
    plt.legend()
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([0, No])
    ax2.set_yticklabels(['0', r'$N_0$'])
    plt.show()
    
def N_log(T, r, No):
    N = [No]
    for t in range(T):
        N.append(N[-1] + dNdt_log(r, N[-1]))
    return N

def dNdt_log(r, N):
    dNdt = -r*N
    return dNdt

def N_alg(T, r, No):
    N = [No]
    for t in range(T):
        N.append(dNdt_alg(r, N[-1]))
    return N

def dNdt_alg(r, N):
    num = random.choices([0,1], weights=[r, 1-r], k=N)
    dNdt = int(np.sum(num))
    return dNdt


if __name__ == '__main__':
    Task_1(60)