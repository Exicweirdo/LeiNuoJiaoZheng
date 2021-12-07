from functools import partial
import numpy as np
import os, re, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
sys.setrecursionlimit(int(1e6))

# linear fitting function
# input: x::arraylike, y::arraylike show_param=True::bool
# output: m::any|slope, c::any|intercept, R2::any|pearson's r square, sigma_a, sigma_b::any| standard deviation 
def linearfitting(x,y, show_param = True):
    x1 = np.array(x)
    y1 = np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    sigmay = (sum((m*x1+c-y1)**2)/(len(x1)-2))**0.5
    dy = sum(y1**2)*len(y1)-sum(y1)**2
    dx = sum(x1**2)*len(x1)-sum(x1)**2
    R2 = (len(x1)*sum(x1*y1)-sum(x1)*sum(y1))**2/(dx*dy)
    sigma_a = (len(x1)*sigmay**2/dx)**0.5
    sigma_b = (sum(x1**2)*sigmay**2/dx)**0.5
    if show_param = True:
        print('a={}    b={}    R2={}    sigma_a={}    sigma_b={}'.format(m, c, R2,sigma_a,sigma_b))
    return m,c,R2,sigma_a,sigma_b

# binary searching function(abandoned)
def binarySearch(arr, l, r, x, err):
    if r >= l: 
        mid = int(l + (r - l)/2)
  
        if (arr[mid]-x)**2<=err**2: 
            return mid 

        elif (arr[mid]-x) > err: 
            return binarySearch(arr, l, mid-1, x, err) 
  
        else: 
            return binarySearch(arr, mid+1, r, x, err) 
    else: 
        return -1
# binary searching function(used)
# input: func::function you need to search, should increasing
# low, high::float bound of searching
# x::float value to be located
# err::float error of your searching 
def binarysearchfunc(func, low, high, x, err, print_mid = False):
    mid = low +(high-low)/2
    if print_mid:
        print(mid)
    if (func(mid)-x)**2<err**2:
        return mid
    elif (func(mid)-x)>err:

        return  binarysearchfunc(func, low, mid, x, err)
    else:

        return binarysearchfunc(func, mid, high, x, err)
#integration
def integ(func, low, high, split = 1000):
    xi = np.linspace(low, high, split)
    mean_y = (func(xi)[:-1]+func(xi)[1:])/2
    return sum(mean_y)*(high-low)/(split-1)
#main func of renold
def Renold(tick, curve, line1, line2, left, right):
    S1 = integ(lambda x: curve(x)-line1(x), left, tick)
    S2 = integ(lambda x: line2(x)-curve(x), tick, right)
    return S1 - S2
if __name__ == '__main__':
    #print(integ(lambda x: x**2, 0, 1))
    #load data
    data_correct = pd.read_csv('./benzoic_acid.csv')
    Temp_time = interp1d(data_correct['time'], data_correct['dT'], kind='cubic') #interpolation
    # correct
    k1, b1 = linearfitting(data_correct['time'].iloc[4:13].to_list(), data_correct['dT'].iloc[4:13].to_list())[0:2]
    k2, b2 = linearfitting(data_correct['time'].iloc[-10:].to_list(), data_correct['dT'].iloc[-10:].to_list())[0:2]
    func_correct = partial(Renold, curve = Temp_time, line1 = lambda x: k1*x + b1, line2 = lambda x: k2*x +b2, left = data_correct['time'].iloc[4], right = data_correct['time'].iloc[-10])
    tick = binarysearchfunc(func_correct, data_correct['time'].iloc[4], data_correct['time'].iloc[-1], 0, 1e-7)
    print('tick ={}'.format(tick))
    print('Delta T = {}'.format((k2-k1)*tick+(b2-b1)))
    S1 = integ(lambda x: Temp_time(x)-k1*x-b1, data_correct['time'].iloc[4], tick)
    S2 = integ(lambda x: k2*x+b2-Temp_time(x), tick, data_correct['time'].iloc[-10])
    # plot
    fig = plt.figure()
    time_new = np.linspace(data_correct['time'][0],data_correct['time'].iloc[-1],1000) 
    plt.vlines(x=tick, ymin=0, ymax= 2, linestyles='--', color = 'black')
    plt.plot(time_new, time_new*k1+b1, '--')
    plt.plot(time_new, time_new*k2+b2, '--')
    range1 = np.linspace(data_correct['time'].iloc[4], tick, 100)
    range2 = np.linspace(tick, data_correct['time'].iloc[-10], 100)
    plt.plot(time_new, Temp_time(time_new), linewidth = 1)
    plt.plot(data_correct['time'], data_correct['dT'],'bo', markersize = 4)
    plt.fill_between(range1, range1*k1 + b1, Temp_time(range1), alpha = 0.5)
    plt.fill_between(range2, Temp_time(range2), range2*k2 + b2, alpha = 0.5)
    plt.text(80, 0.8, 'S1 = {:.4f}'.format(S1))
    plt.text(350, 1.5, 'S2 = {:.4f}'.format(S2))
    plt.xlabel('t/s')
    plt.ylabel(r'$\Delta T$/$ \mathrm{C^{\circ}}$')
    plt.savefig('benzoic acid.pdf')
