"""
Created on Fri Jan 17 14:18:00 2025
""" 

import numpy as np

def moving_mean(t,X,w=4,kind=0):
    """
    @params
        t : array-like. It is the data of the x-axis, which will not be averaged.
It can be for example the time data
        X : array-like. It is the data that will be averaged.
        w : int. It is the size of the window where we will make the average.
        kind : int. It is the kind of moving mean
                0 : centered mean
                1 : left-centered mean
                2 : right-centered mean
    """
    
    if kind == 0:
        w1 = int((w-1)/2); w2 = w-w1-1; 
        Xmean = np.zeros(len(X))
        #Xmean = np.zeros(len(X)-w1-w2)
        #for i in range(w1,len(X)-w2): Xmean[i-w1] = np.mean(X[i-w1:i+w2+1])
        for i in range(len(X)):
            if i-w <0: w1 = 0; w2 = w
            elif i+w+1 > len(X): w2=-i-2; w1 = w-w2-1
            else: w1 = int((w-1)/2); w2 = w-w1-1
            Xmean[i] = np.mean(X[i-w1:i+w2+1])
        #t1 = t[w1:-w2]
        t1 = t
    return t1, Xmean