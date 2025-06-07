"""
Created on Fri Jan 17 14:18:00 2025
""" 

import numpy as np
import matplotlib.pyplot as plt
from LJGas.analysis.visualization import moving_mean
from LJGas.analysis.math import discrete_diff

def get_Pressures(s,time,box,dt):
    """ 
    @params:
        s : list of arrays. It is the data of position and velocity gotten from LJGas.simulate()
        time : array-like. It is the dimensionless time assigned for each iteration.
        box : list. It contains the dimensions of the box that confines the gas.
        dt : float. It is the amplitud of the interval used for calculating the mean force 
            assigned to the colision between the particles and the walls. Choose it carefully.

    @exits:
        time_P : array-like. It is the time associated to each component of the pressures arrays.
        P_top : array-like. Pressure of the top wall.
        P_bottom : array-like. Pressure of the bottom wall.
        P_right : array-like. Pressure of the right wall.
        P_left : array-like. Pressure of the left wall.

    @example:
        >> ...
        >> dt = 5*h
        >> time_P, P_top, P_bottom, P_right, P_left = get_Pressures(s,time,box,dt)
    """

    # Condition for bouncing in any of the walls
    LVX = s[2]; LVY = s[3]
    ii = LVX[:-1]+LVX[1:]==0; jj = LVY[:-1]+LVY[1:]==0

    # Get change in momentum
    aux = LVX.shape[0]-1
    p_right = np.zeros(aux); p_left = np.zeros(aux); p_top = np.zeros(aux); p_bottom = np.zeros(aux)

    for i in range(len(ii)):
        A = LVX[i][ii[i]]; B = LVY[i][jj[i]]
        p_right[i] = 2*np.sum(A[A>=0])
        p_left[i] = 2*np.sum(A[A<0])
        p_top[i] = 2*np.sum(B[B>=0])
        p_bottom[i] = 2*np.sum(B[B<0])

    # Get Pressure per unit deep in each wall
    Nk = int(time[-1]/dt)
    P_right = np.zeros(Nk); P_left = np.zeros(Nk); P_top = np.zeros(Nk); P_bottom = np.zeros(Nk); time_P = np.arange(Nk)*dt

    time1 = time[:-1]
    for k in range(Nk):
        A = (time1>=k*dt)*(time1<(k+1)*dt)
        P_right[k] = np.sum(p_right[A])/(dt*box[1])
        P_left[k] = np.abs(np.sum(p_left[A])/(dt*box[1]))
        P_top[k] = np.sum(p_top[A])/(dt*box[0])
        P_bottom[k] = np.abs(np.sum(p_bottom[A])/(dt*box[0]))

    return time_P, P_top, P_bottom, P_right, P_left

def plot_pressure_charts(s,time,box,Ldt,dir='images/NparticlesGas/',date=' ',save=False,dpi=300,backup=True):
    """
    Plots a table of pressures based on differents time intervals. 
    By consistency the value of the time mean of P must be invariant
    under changes of this time interval.
    """
    LPmean = []
    fig = plt.figure(figsize=(18,12))
    fig.suptitle('\nPressures vs time', fontsize=14)
    for i in range(len(Ldt)):
        plt.subplot(3,3,i+1)
        time_P, P_top, P_bottom, P_right, P_left = get_Pressures(s,time,box,Ldt[i])
        w = 5; kind = 0
        time_P1, P_top1 = moving_mean(time_P,P_top,w,kind)
        time_P1, P_bottom1 = moving_mean(time_P,P_bottom,w,kind)
        time_P1, P_right1 = moving_mean(time_P,P_right,w,kind)
        time_P1, P_left1 = moving_mean(time_P,P_left,w,kind)
        P_mean1 = (P_top1+P_bottom1+P_right1+P_left1)/4

        if i in [0,3,6] : plt.ylabel(r'$\overline{P}$',fontsize=14)
        if i in [3,4,5] : plt.xlabel(r'$\overline{t}$',fontsize=14)
        plt.plot(time_P1,P_top1,'-r',label=r'$\overline{P}$ top',lw=1)        #;plt.plot(time_P, P_top,'o',color='red',alpha=0.5, ms=1.5)
        plt.plot(time_P1,P_bottom1,'-b',label=r'$\overline{P}$ bottom',lw=1)  #;plt.plot(time_P, P_bottom,'o',color='blue',alpha=0.5, ms=1.5)
        plt.plot(time_P1,P_right1,'-g',label=r'$\overline{P}$ right',lw=1)    #;plt.plot(time_P, P_right,'o',color='green',alpha=0.5, ms=1.5)
        plt.plot(time_P1,P_left1,'-m',label=r'$\overline{P}$ left',lw=1)      #;plt.plot(time_P, P_left,'o',color='magenta',alpha=0.5, ms=1.5)
        plt.plot(time_P1,P_mean1,'-',color='cyan',label=r'$\left< \overline{P} \right>$ Time mean '+f'{np.round(np.mean(P_mean1)*10**3,2)}E-3')
        LPmean.append(np.mean(P_mean1))
        plt.legend(loc='upper right')
    if save: plt.savefig(dir+date+'pressures.png',dpi=dpi)
    if backup: 
        return LPmean

def plot_pressure_line(s,time,box,Ldt,h,w1=5,w2=200,dir='images/NparticlesGas/',date=' ',save=False, dpi=300):
    fig = plt.figure(figsize=(20,5))
    fig.suptitle('\nMean pressure vs time\n', fontsize=14)
    for i in range(len(Ldt)):
        plt.subplot(1,3,i+1)
        time_P, P_top, P_bottom, P_right, P_left = get_Pressures(s,time,box,Ldt[i])
        w = w1; kind = 0
        time_P1, P_top1 = moving_mean(time_P,P_top,w,kind)
        time_P1, P_bottom1 = moving_mean(time_P,P_bottom,w,kind)
        time_P1, P_right1 = moving_mean(time_P,P_right,w,kind)
        time_P1, P_left1 = moving_mean(time_P,P_left,w,kind)
        P_mean1 = (P_top1+P_bottom1+P_right1+P_left1)/4
        P_mean2 = np.mean(P_mean1)*np.ones(len(time_P1))

        plt.ylabel(r'$\left< \overline{P} \right>$',fontsize=14)
        plt.xlabel(r'$\overline{t}$',fontsize=14)
        plt.plot(time_P1,P_mean1,'-',color='blue')
        plt.plot(0,0,'o',ms=0,label=r'$d\overline{t} = $'+f'{np.round(Ldt[i],2)}')
        plt.plot(time_P1,P_mean2,'-',color='red',label=r'Time average '+f'{np.round(np.mean(P_mean1)*10**3,1)}E-3')

        time_P, P_top, P_bottom, P_right, P_left = get_Pressures(s,time,box,5*h)
        w = w2; kind = 0; lw=1; alpha=0.4
        time_P1, P_top1 = moving_mean(time_P,P_top,w,kind)
        time_P1, P_bottom1 = moving_mean(time_P,P_bottom,w,kind)
        time_P1, P_right1 = moving_mean(time_P,P_right,w,kind)
        time_P1, P_left1 = moving_mean(time_P,P_left,w,kind)
        P_mean1 = (P_top1+P_bottom1+P_right1+P_left1)/4
        plt.plot(time_P1,P_top1,'-',color='gray',lw=lw,alpha=alpha)
        plt.plot(time_P1,P_bottom1,'-',color='gray',lw=lw,alpha=alpha)
        plt.plot(time_P1,P_right1,'-',color='gray',lw=lw,alpha=alpha)
        plt.plot(time_P1,P_left1,'-',color='gray',lw=lw,alpha=alpha)
        plt.legend(loc='upper right')
    if save: plt.savefig(dir+date+'pressures1.png',dpi=dpi)

def plot_cumulative_momentum(s,time,box,h,w=500,kind=0,dir='images/NparticlesGas/',date=' ',save=False, dpi=300):
    plt.figure(figsize=(8,6))
    dt = 2*h
    time_P, P_top, P_bottom, P_right, P_left = get_Pressures(s,time,box,dt)
    Cp_top = np.cumsum(P_top*dt*box[0])
    Cp_bottom = np.cumsum(P_bottom*dt*box[0])
    Cp_right = np.cumsum(P_right*dt*box[1])
    Cp_left = np.cumsum(P_left*dt*box[1])
    
    time_P1, Cp_top1 = moving_mean(time_P, Cp_top,w,kind)
    time_P1, Cp_bottom1 = moving_mean(time_P,Cp_bottom,w,kind)
    time_P1, Cp_right1 = moving_mean(time_P,Cp_right,w,kind)
    time_P1, Cp_left1 = moving_mean(time_P,Cp_left,w,kind)
    Cp_mean1 = (Cp_top1+Cp_bottom1+Cp_right1+Cp_left1)/4

    plt.title('Cumulative Momentum', fontsize=14)
    plt.ylabel(r'$\overline{p}$',fontsize=14)
    plt.xlabel(r'$\overline{t}$',fontsize=14)
    plt.plot(time_P1,Cp_top1,'-r',label='Top',lw=1)        #;plt.plot(time_P, Cp_top,'o',color='red',alpha=0.5, ms=1.5)
    plt.plot(time_P1,Cp_bottom1,'-b',label='Bottom',lw=1)  #;plt.plot(time_P, Cp_bottom,'o',color='blue',alpha=0.5, ms=1.5)
    plt.plot(time_P1,Cp_right1,'-g',label='Right',lw=1)    #;plt.plot(time_P, Cp_right,'o',color='green',alpha=0.5, ms=1.5)
    plt.plot(time_P1,Cp_left1,'-m',label='Left',lw=1)      #;plt.plot(time_P, Cp_left,'o',color='magenta',alpha=0.5, ms=1.5)
    plt.plot(time_P1,Cp_mean1,'-',color='cyan',label='Mean')
    plt.legend(loc='upper left')
    if save: plt.savefig(dir+date+'pressures2.png',dpi=dpi)
    return np.mean(discrete_diff(time_P1,Cp_mean1))/box[1]