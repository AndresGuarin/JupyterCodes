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
                0 : centered
                1 : left-centered
                2 : right-centered
                3 : adapted
    """
    
    if kind == 0:
        w1 = (w-1)//2; w2 = w-w1-1; 
        Xmean = np.zeros(len(X)-w1-w2)
        for i in range(w1,len(X)-w2): 
            Xmean[i-w1] = np.mean(X[i-w1:i+w2+1])
        t1 = t[w1:-w2]
    
    elif kind == 1: 
        Xmean = np.zeros(len(X)-w)
        for i in range(len(X)-w): 
            Xmean[i] = np.mean(X[i:i+w])
        t1 = t[w1:-w2]

    elif kind == 2: 
        Xmean = np.zeros(len(X)-w+1)
        for i in range(w,len(X)): 
            Xmean[i-w] = np.mean(X[i-w:i])
        t1 = t[w1:-w2]

    elif kind == 3:
        Xmean = np.zeros(len(X))
        for i in range(1,len(X)+1):
            if i-w//2 <= 0: Xmean[i-1] = np.mean(X[0:i*2])
            elif i-w//2+w <= len(X): Xmean[i-1] = np.mean(X[i-w//2:i-w//2+w])
            else: Xmean[i-1] = np.mean(X[i-w//2:]); print(X[i-w//2:])
        t1 = t

    return t1, Xmean