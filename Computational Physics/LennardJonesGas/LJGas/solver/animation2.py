"""
Created on Fri Jan 17 14:18:00 2025
""" 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_path(self,s,duration=5,kind=0,save=False,name=None,verbose=True,
                 plot_params=['-b','or',0.5,5,0.5],amp=1,length=8,interval=40):
    """
    Animates the positions of the particles of the system.

    @params:
        self : self object. It contains the data of LJGas.solver.particles object.
        s : array-like. It contains the positions and velocities of the particles.
        duration : float. Time in seconds of the simulation.
        kind : int. If it is 0, then we plot lines and point. If it is 1, we plot
            only lines
        save : bool. If it is true the animation is saved in pc.
        name : str. If save is true, then this is the name of the saved video.
            It includes the extension of the video (e.g.: .mp4)
        verbose : bool. If it is true, we show the labels of time and iteration.
        plot_params : list. It contains [fmt1, fmt2, lw1, ms1, alpha1].
        amp : 1. float. It is associated to a parameter of get_uniform function.
        length : float. It is the length of the figure.
        interval : float. Interval in miliseconds between each frame in the 
            animation.
    """

    # Make the steps uniform
    LX1 = s[0]
    LY1 = s[1]
    time1 = np.arange(len(LX1))*self.alpha*self.h
     
    # Plot parameters
    fmt1 = plot_params[0]
    fmt2 = plot_params[1]
    lw1 = plot_params[2]
    ms1 = plot_params[3]
    alpha1 = plot_params[4]
    b0 = self.box[0]/2; b1 = self.box[1]/2
    
    #local function for update each frame
    def update(j):    
        ax.clear() # Clear the before plot
        if kind==0:
            for i in range(self.Np): plt.plot(LX1[:j,i],LY1[:j,i],fmt1,lw=lw1,alpha=alpha1); ax.plot(LX1[j,i], LY1[j,i], fmt2,ms=ms1)
        if kind==1: 
            for i in range(self.Np): ax.plot(LX1[j,i], LY1[j,i], fmt2,ms=ms1)
        plt.plot(0,0,'o',ms=0, label=r'$n=$'+f'{j}\n'+r'$\overline{t}=$ '+f'{np.round(time1[j],1)}') #legends
        plt.plot([-b0,b0,b0,-b0,-b0],[-b1,-b1,b1,b1,-b1],'-b',lw=2) #box
        if verbose: plt.legend(loc='upper right')
    
    # Create figure and axis
    heigth = b1/b0*length
    fig = plt.figure(figsize=(length*1.1,heigth))
    ax = fig.gca()
    
    # Set the steps of the animation
    local_time = interval/100   # Empirical time in seconds that matplotlib.pyplot lasts for each plot of the animation
    Nf = LX1.shape[0]
    dj = int(Nf*local_time/duration)
    
    # Animate the movement
    anim = animation.FuncAnimation(fig,update,range(1,Nf,dj), repeat=False, interval=interval)
    if save: anim.save(name,writer='ffmpeg')
    return anim
    #plt.show(ani)