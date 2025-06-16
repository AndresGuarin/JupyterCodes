"""
Created on Fri Jan 13 15:35:00 2025

This code simulates the electric conduction in a wire ...
"""

import numpy as np

def EF(X,Y,t):
     return [X*0, X*0] #In order: [Ex, Ey]
                
def BF(X,Y,t):
    return X*0 #In order: [Bz]

class ECond:
    def __init__(self,h=0.05,N=100,NL=1,NF=1,EField=EF,BField=BF,verbose=False):
        """
            @params
                EField: func. 
                        It is a function of X,Y,t and it returns a list of the form [Ex, Ey]
                BField: func.
                        It is a function of X,Y,t and it retunrs Bz 

        """
        # Dynamic parameters
        self.h = h    # Time lapse between steps
        self.N = N    # Number of steps
        self.NL = NL  # Number of free particles
        self.NF = NF  # Numer of fixed particles
        self.Np = NL+NF # Total number of particles

        # Other parameters
        self.h1 = self.h
        self.verbose = verbose

        # Auxiliar parameters
        self.IsFree = np.arange(self.Np) < self.Np-NF

        # Fields
        self.EField = EField
        self.BField = BField

    def get_self(self):
        return self

    # Functions
    def Fk(self,X,Y,Vx,Vy): #Force function
        # Associated matrixes of distance and force betweeen particles
        I = np.eye(self.Np)
        A, B = np.meshgrid(X,X); dX = A-B
        A, B = np.meshgrid(Y,Y); dY = A-B
        A, B = np.meshgrid(self.Q,self.Q); Q2 = A*B
        R = np.sqrt(dX**2+dY**2) + I
        F = Q2/R**2
        Ax1 = F*dX/R
        Ay1 = F*dY/R

        # Flatten vectors
        Ax = np.sum(Ax1,axis=0)
        Ay = np.sum(Ay1,axis=0)
        
        # Force due to an external field E and B with Ez=0, Bx=0, By=0
        E = self.EField(X,Y,self.h*self.i)
        B = self.BField(X,Y,self.h*self.i)

        Ax += self.Q*(E[0] + Vy*B)
        Ay += self.Q*(E[1] - Vx*B)

        return np.array([Vx, Vy, Ax*self.IsFree, Ay*self.IsFree])

    def simulate(self,CI):
        LX = np.zeros((self.N+1,self.Np))
        LY = np.zeros((self.N+1,self.Np))
        LVx = np.zeros((self.N+1,self.Np))
        LVy = np.zeros((self.N+1,self.Np))

        LX[0] = CI[0]
        LY[0] = CI[1]
        LVx[0] = CI[2]
        LVy[0] = CI[3]
        self.Q = CI[4]

        for i in range(self.N):
            self.i = i
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
            Q:
                np.array of the charges of the particles

            @returns
            xf,yf,vxf,vyf
                np.array that contains the final positions and velocities of the particles
        """
        #K1 and calculations
        K1 = self.h*self.Fk(X,Y,Vx,Vy)
        X1 = X + K1[0]/2
        Y1 = Y + K1[1]/2
        Vx1 = Vx + K1[2]/2
        Vy1 = Vy + K1[3]/2

        #K2 and calculations
        K2 = self.h*self.Fk(X1,Y1,Vx1,Vy1)
        X2 = X + K2[0]/2
        Y2 = Y + K2[1]/2
        Vx2 = Vx + K2[2]/2
        Vy2 = Vy + K2[3]/2

        #K3 and calculations
        K3 = self.h*self.Fk(X2,Y2,Vx2,Vy2)
        X3 = X + K3[0]
        Y3 = Y + K3[1]
        Vx3 = Vx + K3[2]
        Vy3 = Vy + K3[3]

        #K4 and calculating final positions
        K4 = self.h*self.Fk(X3,Y3,Vx3,Vy3)
        Xf = X + 1/6*(K1[0] + 2*K2[0] + 2*K3[0] + K4[0])
        Yf = Y + 1/6*(K1[1] + 2*K2[1] + 2*K3[1] + K4[1])
        Vxf = Vx + 1/6*(K1[2] + 2*K2[2] + 2*K3[2] + K4[2])
        Vyf = Vy + 1/6*(K1[3] + 2*K2[3] + 2*K3[3] + K4[3])
        
        return np.array([Xf, Yf, Vxf, Vyf])

    def get_net_force(self,s):
        LF = np.zeros((self.N+1,self.Np))
        for i in range(self.N+1):
            X = s[0][i,:]
            Y = s[1][i,:]
            I = np.eye(self.Np)
            A, B = np.meshgrid(X,X); dX = A-B
            A, B = np.meshgrid(Y,Y); dY = A-B 
            A, B = np.meshgrid(self.Q,self.Q); Q2 = A*B
            R = np.sqrt(dX**2+dY**2) + I
            F = Q2/R**2
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
            A, B = np.meshgrid(self.Q,self.Q); Q2 = A*B
            R = np.sqrt(dX**2+dY**2) + I
            Vi = (Q2/R)*(np.eye(self.Np)==0)
            Eki = 1/2*(Vx**2+Vy**2)
            V[i] = np.sum(Vi)/2
            Ek[i] = np.sum(Eki)

        return [Ek,V,Ek+V]