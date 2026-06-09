"""
Created on Tuesday Jan 17 13:23:00 2025
""" 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_conners(box):
      a,b,c,d = box
      X = [a,b,b,a,a]
      Y = [c,c,d,d,c]
      return X,Y

def plot_boxes(box1,box2):
    X1, Y1 = get_conners(box1) 
    X2, Y2 = get_conners(box2); 
    plt.plot(X1,Y1,'-b',lw=1.7)
    plt.plot(X2,Y2,'-b',lw=1.7)

def animate_path(self,s,kind=0,save=False,name=None,verbose=True,
                 plot_params=['-r','or',0.2,5.0,0.5],length=8,interval=100,dj=1,
                 j0=1,L=None,comet=False,ltraj=100,box1=None,box2=None,b_param=1):
    """
    Animates the positions of the particles of the system.

    @params:
        self : self object. 
                It contains the data of LJGas.solver.particles object.
        s : array-like. 
                It contains the positions and velocities of the particles.
        save: boolean
                If it is true, the animation is saved as a video file
        name : str or None. 
                If save is true, then this is the name of the saved video.
                It includes the extension of the video (e.g.: .mp4)
        verbose : bool. 
                If it is true, we show the labels of time and iteration.
        plot_params : list. 
                It contains [fmt1, fmt2, lw1, ms1, alpha1].
        length : float. 
                It is the length of the figure.
        interval : float. 
                Interval in miliseconds between each frame in the 
                animation.
        dj: int
                Number of the steps of the iteration function that plots each frame
                of the animation on screen.
        j0: int
                Starting point of the animation.
        L: float or None
                It is the limit of the x- and y-axis
        comet: boolean
                If it is true, only a portion of the whole path of each particle is plotted.
                animation
        ltraj: int
                It controls how much of the path is plotted in each frame when comet is True.
        box1 and box2: list or None
                It contains the x and y coordinates associated to 2 rectangles that are
                plotted in the animation.
        b_param: float
                Parameter of the x- and y-cut of the line of code that computes the "current density" J 
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
    b=b_param


    #local function for update each frame
    def update(j):
        jmin = 0
        if kind==0:
                ax.clear() # Clear the before plot
                if comet: jmin = np.max([0,j-ltraj])
                for i in range(self.Np):
                      plt.plot(LX[jmin:j,i],LY[jmin:j,i],fmt1,lw=lw1,alpha=alpha1); # Trajectories
                      plt.plot(LX[j,i], LY[j,i], fmt2,ms=ms1)               # Positions
                plt.plot(np.mean(LX[j,:]),np.mean(LY[j,:]),'xg',ms=5)
                plt.plot(0,0,'o',ms=0, label=r'$n=$'+f'{j}\n'+r'$\overline{t}=$ '+f'{np.round(time[j],1)}') # Legends
        elif kind==1:
              for i in range(self.Np):
                    plt.plot(LX[j,i],LY[j,i],fmt1,ms=ms1,alpha=alpha1,label='partícula'); # Trajectories
        if L!= None: plt.xlim(-L,L); plt.ylim(-L,L)
        if box1 != None: plot_boxes(box1,box2)
        if verbose: plt.legend(loc='upper right')
        J = -np.sum(s[3][j,:][(LY[j,:]>=-b)*(LY[j,:]<=b)*(LY[j,:]<0)]) + np.sum(s[2][j,:][LY[j,:]<b])+ np.sum(s[3][j,:][(LY[j,:]>=-b)*(LY[j,:]<=b)*(LX[j,:]>0)]) - np.sum(s[2][j,:][LY[j,:]>-b])
        ax.quiver(0, 0, J, 0, scale_units='xy',angles='xy',scale=4,color='orange',width=0.01)
    # Create figure and axis
    fig = plt.figure(figsize=(length,length))
    ax = fig.gca()
    
    # Animate the movement
    anim = animation.FuncAnimation(fig,update,range(j0,LX.shape[0],dj), repeat=False, interval=interval)
    if save: 
        if interval==0: print("WARNING: Interval must not be zero.")
        anim.save(name,writer='ffmpeg')
    return anim
    #plt.show(ani)


