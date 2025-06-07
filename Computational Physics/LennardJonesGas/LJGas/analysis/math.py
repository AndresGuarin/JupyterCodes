"""
Created on Fri Jan 6 22:19:00 2025
""" 

import numpy as np

def discrete_diff(t,x):
    dx = x[1:]-x[:-1]
    dt = t[1:]-t[:-1]
    return dx/dt


