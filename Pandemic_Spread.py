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

def plot_Task_1(T, No=20, r=0.1):
    x = np.arange(T)
    y_log = np.array(N_log(T, r, No))
    y_alg = np.array(N_alg(T, r, No))
    plt.plot(x, y_log)
    plt.plot(x, y_alg)
    
def N_log(T, r, No):
    N = [No]
    for t in range(T):
        N.append(N[-1] + dNdt_log(r, N[-1]))
    return N[1:]

def dNdt_log(r, N):
    dNdt = -r*N
    return dNdt

def N_alg(T, r, No):
    N = [No]
    for t in range(T):
        N.append(dNdt_alg(r, N[-1]))
    return N[1:]

def dNdt_alg(r, N):
    num = random.choices([0,1], weights=[r, 1-r], k=N)
    dNdt = int(np.sum(num))
    return dNdt


if __name__ == '__main__':
    random.seed(5)
    plot_Task_1(60)