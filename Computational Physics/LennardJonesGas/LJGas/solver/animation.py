"""
Created on Fri Jan 17 14:18:00 2025
""" 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_path(self,s,duration=5,kind=0,save=False,name=None,verbose=True,plot_params=['-b','or',0.5,5,0.5],amp=1):
    """
    @params:
        self: self object that contains the data of 
              LJGas.solver.particles object.
    """
    def get_uniform(self,M,amp=1):
        """
        @params 
            M :  np.array of data of the motion as LX, LY, LVx, and LVy. 
                It can be for example s[0], s[1], time
        """
        if self.LContacts != []:
            slice_step = int(self.h/self.h0)*amp
            i0 = self.LContacts[0][0]
            Ls = [M[:i0]]
            for k in range(len(self.LContacts)-1):
                i,j = self.LContacts[k]
                i1,j1 = self.LContacts[k+1]
                Ls.append(M[i:j:slice_step])
                Ls.append(M[j:i1])
            i1,j1 = self.LContacts[-1]
            Ls.append(M[i1:j1:slice_step])
            Ls.append(M[j1:])
            s1 = np.concatenate(Ls)
        else:
            s1 = M
        return s1

    # Time list
    Lh = self.Lh
    time = np.zeros(self.N+1)
    for i in range(self.N):
        time[i+1] = time[i]+Lh[i]

    # Make the steps uniform
    LX1 = get_uniform(self,s[0],amp)
    LY1 = get_uniform(self,s[1],amp)
    time1 = get_uniform(self,time,amp)
    
    # Plot parameters
    fmt1 = plot_params[0]
    fmt2 = plot_params[1]
    lw1 = plot_params[2]
    ms1 = plot_params[3]
    alpha1 = plot_params[4]

    #local function for update each frame
    def update(j):    
        ax.clear() # Clear the before plot
        
        # Plot positions
        if kind==0:
            for i in range(self.Np):
                ax.plot(LX1[:j,i],LY1[:j,i],fmt1,lw=lw1,alpha=alpha1); ax.plot(LX1[j,i], LY1[j,i], fmt2,ms=ms1)

        if kind==1: 
            for i in range(self.Np): ax.plot(LX1[j,i], LY1[j,i], fmt2,ms=ms1)

        # Put on legend the Iteration and time of the frame
        ax.plot(0,0,'o',ms=0, label=r'$n=$'+f'{j}')
        ax.plot(0,0,'o',ms=0, label=r'$\overline{t}=$ '+f'{np.round(time1[j],1)}')
        
        # Put grids, legends, and limits
        b0 = self.box[0]/2+self.R0/2; b1 = self.box[1]/2+self.R0/2
        ax.plot([-b0,b0,b0,-b0,-b0],[-b1,-b1,b1,b1,-b1],'-b',lw=2)
        if verbose: plt.legend()
        l0 = self.lim[0]+1.1*self.R0/2
        #l0 = 8
        plt.xlim(-l0, l0)
        plt.ylim(-l0, l0) 
    
    # Create figure and axis
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    
    # Put labels, title, grids, and legend
    plt.xlabel(r'$\overline{x}$')
    plt.ylabel(r'$\overline{y}$')
    plt.title('Lennard-Jones Gas of N partilces')
    #plt.grid()
    plt.legend()
    
    # Set the steps of the animation
    local_time = 0.25   # Empirical time in seconds that matplotlib.pyplot lasts for each plot of the animation
    N_frames = duration/local_time
    step = int(self.N/N_frames)
    Nf = LX1.shape[0]

    # Animate the movement
    ani = animation.FuncAnimation(fig,update,range(1,Nf,step), repeat=False) 
    
    if save: ani.save(name,writer='ffmpeg')
    plt.show(ani)