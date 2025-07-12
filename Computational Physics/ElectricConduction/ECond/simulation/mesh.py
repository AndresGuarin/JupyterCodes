"""
Created on Mon Jan 30 15:01:00 2025

This code give us the positions of the positive cores that form the 
conductor.
"""

import numpy as np

# Auxiliary functions
def super_concatenate(set):
    XC = np.array([])
    YC = np.array([])
    for element in set:
        XC = np.concatenate([XC,element[0]])
        YC = np.concatenate([YC,element[1]])
    return XC, YC

# Mesh functions
def hexagonal(P,nrows,ncols,a,coef=1):
    """
        This functions return the flatten positions XC and YC of the 
        positive cores of the conductor.

        @params
            P: list. It contains the initial position of lower-left conner the grid.
            nrows: int. Number of rows of the grid.
            ncols: int. Number of columns of the grid.
            a: float. Lattice constant. Is the vertical distance between near cores.
            coef: float. Factor of the horizontal distance between near cores. Its deviation from 1 makes the grid change its hexagonal form.
    """
    nrows -= 1
    b = a*np.sqrt(3)/2 * coef
    tx = np.arange(0,ncols,1.); ty = np.arange(0,nrows,1.)
    XC, YC  = np.meshgrid(tx*b+P[0],ty*a+P[1])
    vx = np.arange(1,1+YC.shape[1],1); vy = np.arange(1,1+YC.shape[0],1)
    YC += (np.meshgrid(vx,vy)[0]%2==0)*a/2
    XC, YC = XC.flatten(), YC.flatten()
    nleyer = ncols//2 + ncols%2
    YC = np.concatenate([YC,np.zeros(nleyer)+(ty[-1]+1)*a+P[1]])
    XC = np.concatenate([XC,np.arange(0,nleyer,1.)*2*b + P[0]])
    return XC, YC

def x_squared(P,nrows,ncols,a):
    """
        This functions return the flatten positions XC and YC of the 
        positive cores of the conductor.

        @params
            P: list. It contains the initial position of lower-left conner the grid.
            nrows: int. Number of principal rows of the grid.
            ncols: int. Number of principal columns of the grid.
            a: float. Lattice constant. Is the vertical and horizontal distance between near cores.
    """
    return hexagonal(P=P,nrows=nrows,ncols=2*ncols-1,a=a,coef=a/6)

def rectangular(P,nrows,ncols,a,b):
    """
        This functions return the flatten positions XC and YC of the 
        positive cores of the conductor.

        @params
            P: list. It contains the initial position of lower-left conner the grid.
            nrows: int. Number of rows of the grid.
            ncols: int. Number of columns of the grid.
            a: float. Vertical distance between near cores.
            b: float. Horizontal distance between near cores.
    """
    tx = np.arange(0,ncols,1.); ty = np.arange(0,nrows,1.)
    XC, YC  = np.meshgrid(tx*b+P[0],ty*a+P[1])
    return XC.flatten(), YC.flatten()

def squared(P,nrows,ncols,a):
    """
        This functions return the flatten positions XC and YC of the 
        positive cores of the conductor.

        @params
            P: list. It contains the initial position of lower-left conner the grid.
            nrows: int. Number of rows of the grid.
            ncols: int. Number of columns of the grid.
            a: float. Lattice constant. It is the vertical and horizontal distance between near cores.        
    """
    tx = np.arange(0,ncols,1.); ty = np.arange(0,nrows,1.)
    XC, YC  = np.meshgrid(tx*a+P[0],ty*a+P[1])
    return XC.flatten(), YC.flatten()