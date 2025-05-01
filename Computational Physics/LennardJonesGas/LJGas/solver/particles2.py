"""
Created on Fri Jan 17 14:18:00 2025

This codes calculates the positions and velocities of a set of partilces confined in a 
2D box using the RK4 method. It was assumed a Lennard-Jones 12-6 potential between all the
particles and perfect collisions with the walls of the box. 

The default time step is h1, but when two particles are near to a collision the time step
changes to h0 (this occurs when any force between particles became >= F(R0), which is the 
force between two particles at a distance R0 defined by the user)
"""
import numpy as np

class LJGas:
    def __init__(self,h=0.05,N=100,Np=2,R1=3,box=[20,10],verbose=False):
        # Dynamic parameters
        self.h = h  # Time lapse between steps
        self.N = N   # Number of steps
        self.Np = Np    # Number of particles
        self.R1 = R1

        # Other parameters
        self.verbose = verbose

        # Rectangular box parameters. Note: the box is centered at origin
        self.box = box
        self.lim = [box[0]/2*1.1, box[1]/2*1.1]

    def get_self(self):
        return self

    # Functions
    def Fk(self,X,Y,Vx,Vy):
        Ax = np.zeros(self.Np)
        Ay = np.zeros(self.Np)
        for i in range(self.Np):
            ii = np.abs(X-X[i])<=self.R1; jj = np.abs(Y-Y[i])<=self.R1
            ii[i] = False; jj[i] = False
            X1 = X[ii+jj]; Y1 = Y[ii+jj]; dX = X1-X[i]; dY = Y1-Y[i]
            R = np.sqrt(dX**2 + dY**2); FR = 2/R**14-1/R**8 #FR = F/R
            Ax[i] = np.sum(FR*dX); Ay[i] = np.sum(FR*dY)
        return np.array([Vx, Vy, Ax, Ay])

    def simulate(self,CI):
        LX = np.zeros((self.N+1,self.Np))
        LY = np.zeros((self.N+1,self.Np))
        LVx = np.zeros((self.N+1,self.Np))
        LVy = np.zeros((self.N+1,self.Np))
        self.Lh = np.zeros(self.N+1)

        LX[0] = CI[0]
        LY[0] = CI[1]
        LVx[0] = CI[2]
        LVy[0] = CI[3]

        for i in range(self.N):
            next_val = self.next_value(LX[i],LY[i],LVx[i],LVy[i])
            LX[i+1] = next_val[0]
            LY[i+1] = next_val[1]
            LVx[i+1] = next_val[2]
            LVy[i+1] = next_val[3]
        
        return [LX, LY, LVx, LVy]

    def next_value(self,X,Y,Vx,Vy):
        """
            Calculates the next positions and velocities after a time lapse of self.h
            Find the 4 k-values of the Runge-Kutta method (RK4)
            @params
            X,Y,Vx,Vy:
                np.array of the position and velocity of the particles

            @returns
            xf,yf,vxf,vyf
                np.array that contains the final positions and velocities of the particles
        """
        K1 = self.h*self.Fk(X,Y,Vx,Vy); X1 = X + K1[0]/2; Y1 = Y + K1[1]/2; Vx1 = Vx + K1[2]/2; Vy1 = Vy + K1[3]/2
        K2 = self.h*self.Fk(X1,Y1,Vx1,Vy1); X2 = X + K2[0]/2; Y2 = Y + K2[1]/2; Vx2 = Vx + K2[2]/2; Vy2 = Vy + K2[3]/2
        K3 = self.h*self.Fk(X2,Y2,Vx2,Vy2); X3 = X + K3[0]; Y3 = Y + K3[1]; Vx3 = Vx + K3[2]; Vy3 = Vy + K3[3]
        K4 = self.h*self.Fk(X3,Y3,Vx3,Vy3) 
        Xf = X + 1/6*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0]); Yf = Y + 1/6*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
        Vxf = Vx + 1/6*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2]); Vyf = Vy + 1/6*(K1[3] + 2*K2[3] + 2*K3[3] + K4[3])

        # Reflection at the walls of the box
        ii = Xf>=self.box[0]/2    # Right wall
        jj = Xf<=-self.box[0]/2   # Left wall
        kk = Yf>=self.box[1]/2    # Top wall
        ll = Yf<=-self.box[1]/2   # Bottom wall
        
        if np.sum(ii) != 0:Xf += ii*(self.box[0]-2*Xf); Vxf += ii*(-Vxf-Vx)     # Bouncing off the right wall
        if np.sum(jj) != 0:Xf += jj*(-self.box[0]-2*Xf); Vxf += jj*(-Vxf-Vx)    # Bouncing off the left wall
        if np.sum(kk) != 0:Yf += kk*(self.box[1]-2*Yf); Vyf += kk*(-Vyf-Vy)     # Bouncing off the top wall
        if np.sum(ll) != 0:Yf += ll*(-self.box[1]-2*Yf); Vyf += ll*(-Vyf-Vy)    # Bouncing off the bottom wall
        
        return [Xf, Yf, Vxf, Vyf]

    def get_net_force(self,s):
        LF = np.zeros((self.N+1,self.Np))
        for i in range(self.N+1):
            X = s[0][i,:]
            Y = s[1][i,:]
            I = np.eye(self.Np)
            A, B = np.meshgrid(X,X); dX = A-B
            A, B = np.meshgrid(Y,Y); dY = A-B 
            R = np.sqrt(dX**2+dY**2) + I
            F = 2/R**13-1/R**7
            LF[i] = np.sum(F,axis=0)
        return LF

    def get_Energy(self,s):
        Ek = np.zeros(self.N+1)
        V = np.zeros(self.N+1)
        for i in range(self.N+1):
            X = s[0][i,:]
            Y = s[1][i,:]
            Vx = s[2][i,:]
            Vy = s[3][i,:]
            I = np.eye(self.Np)
            A, B = np.meshgrid(X,X); dX = A-B
            A, B = np.meshgrid(Y,Y); dY = A-B 
            R = np.sqrt(dX**2+dY**2) + I
            Vi = 1/(6*R**12)-1/(6*R**6)
            Eki = 1/2*(Vx**2+Vy**2)
            V[i] = np.sum(Vi)/2
            Ek[i] = np.sum(Eki)
        return [Ek,V,Ek+V]