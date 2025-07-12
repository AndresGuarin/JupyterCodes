"""
Created on Tuesday Jan 17 13:23:00 2025
""" 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_path(self,s,kind=0,save=False,name=None,verbose=True,
                 plot_params=['-r','or',0.2,5.0,0.5],length=8,interval=100,dj=1,
                 j0=1,L=10,ltraj=100):
    """
    Animates the positions of the particles of the system.

    @params:
        self : self object. 
                It contains the data of LJGas.solver.particles object.
        s : array-like. 
                It contains the positions and velocities of the particles.
        duration : float. 
                Time in seconds of the simulation.
        save : bool. 
                If it is true the animation is saved in pc.
        name : str. 
                If save is true, then this is the name of the saved video.
                It includes the extension of the video (e.g.: .mp4)
        verbose : bool. 
                If it is true, we show the labels of time and iteration.
        plot_params : list. 
                It contains [fmt1, fmt2, lw1, ms1, alpha1].
        amp : 1. float. 
                It is associated to a parameter of get_uniform function.
        length : float. 
                It is the length of the figure.
        interval : float. 
                Interval in miliseconds between each frame in the 
            animation.
    """
    
    # Time list
    time = np.arange(0,self.N+1,1)*self.h
    
    # Make the steps uniform
    LX = s[0]
    LY = s[1]
    
    # Plot parameters
    fmt1 = plot_params[0]
    fmt2 = plot_params[1]
    lw1 = plot_params[2]
    ms1 = plot_params[3]
    alpha1 = plot_params[4]

    #local function for update each frame
    def update(j):
        if kind==0:
                ax.clear() # Clear the before plot
                jmin = np.max([0,j-ltraj])
                for i in range(self.Np):
                      plt.plot(LX[jmin:j,i],LY[jmin:j,i],fmt1,lw=lw1,alpha=alpha1); # Trajectories
                      plt.plot(LX[j,i], LY[j,i], fmt2,ms=ms1)               # Positions
                plt.plot(self.XC,self.YC,'ob')                              # Positive Cores
                plt.plot(np.mean(LX[j,:]),np.mean(LY[j,:]),'xg',ms=5)
                plt.plot(0,0,'o',ms=0, label=r'$n=$'+f'{j}\n'+r'$\overline{t}=$ '+f'{np.round(time[j],1)}') # Legends
        elif kind==1:
              for i in range(self.Np):
                    plt.plot(LX[j,i],LY[j,i],fmt1,ms=ms1,alpha=alpha1); # Trajectories
        else:
              for i in range(self.Np):
                    plt.plot(LX[j-1:j+1,i],LY[j-1:j+1,i],fmt1,lw=lw1,alpha=alpha1); # Trajectories
        plt.xlim(-L,L)
        plt.ylim(-L,L)
        if verbose: plt.legend(loc='upper right')
    
    # Create figure and axis
    fig = plt.figure(figsize=(length*1.1,length))
    ax = fig.gca()
    plt.plot(self.XC,self.YC,'ob')                              # Positive Cores

    # Animate the movement
    anim = animation.FuncAnimation(fig,update,range(j0,LX.shape[0],dj), repeat=False, interval=interval)
    if save: anim.save(name,writer='ffmpeg')
    return anim
    #plt.show(ani)