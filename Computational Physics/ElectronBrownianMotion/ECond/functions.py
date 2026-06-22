import numpy as np
import matplotlib.pyplot as plt
# import ECond.simulation.solver as solver
# import ECond.simulation.animation as ani

from datetime import datetime
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_intial_state(X0,Y0,Vx0,Vy0,scale):
    plt.figure(figsize=(4,4))
    plt.plot(X0,Y0,'or')
    plt.quiver(X0, Y0, Vx0, Vy0, scale_units='xy',angles='xy',scale=scale,color='orange',width=0.01)

    bb = np.max([np.abs(X0),np.abs(Y0)])*1.4
    plt.xlim(-bb,bb)
    plt.ylim(-bb,bb)
    plt.show()

def get_conners(box):
      a,b,c,d = box
      X = np.array([a,b,b,a,a])
      Y = np.array([c,c,d,d,c])
      return X,Y

def plot_boxes(box1,box2):
    X1, Y1 = get_conners(box1) 
    X2, Y2 = get_conners(box2); 
    plt.plot(X1,Y1,'-b',lw=1.7)
    plt.plot(X2,Y2,'-b',lw=1.7)

def get_uniques(X,Y):
    Z=X+Y*1j
    W=np.unique(Z)
    return np.real(W), np.imag(W)

def get_positions_square_wire(box1,box2,Np=50,mx=50,my=5,seed=None):
    a,b,c,d = box1
    l,m,n,p = box2
    
    if b-a < p-n: print("WARNING: box 1 is smaller than box 2")

    if seed!=None: np.random.seed(seed)
    Lx=(b-a)+(d-l)+(p-a)+(m-l); Ly=(l-c)
    X0 = np.random.randint(low=0, high=mx,size=Np)*(Lx/mx)+a
    Y0 = np.random.randint(low=0, high=my,size=Np)*(Ly/my)+c
    X0, Y0 = get_uniques(X0,Y0)

    b0=b
    b1=b+(d-l)
    b2=b+(d-l)+(p-a)

    X1, Y1 = X0[X0<=b0],Y0[X0<=b0]
    X2, Y2 = X0[(X0>b0)*(X0<=b1)],Y0[(X0>b0)*(X0<=b1)]
    X3, Y3 = X0[(X0>b1)*(X0<=b2)],Y0[(X0>b1)*(X0<=b2)]
    X4, Y4 = X0[X0>b2],Y0[X0>b2]

    delta1=b-l
    delta2=b+(d-l)+(p-a)-p
    delta3=d-l
    delta4=c-a
    delta5=l-b-(d-l)-(p-a)

    X0f = np.concatenate([X1,Y2+delta1,X3-delta2,Y4+delta4])
    Y0f = np.concatenate([Y1,X2-delta1,Y3+delta3,X4+delta5])
    return X0f, Y0f

def get_positions_rounded_wire(slope, width, m=20,Np=80,seed=None,n=6):
    if seed!=None: np.random.seed(seed)
    x0 = np.arctanh(-0.9)/slope + 5*width
    tt = np.linspace(-x0,x0,m)
    X0, Y0 = np.meshgrid(tt,tt)
    Rho = (np.abs(X0)**n+np.abs(Y0)**n)**(1/n)
    ii = Rho<=np.arctanh(-0.9)/slope + 5*width
    jj = Rho>=np.arctanh(0.9)/slope + 3*width
    X1, Y1 = X0[ii*jj], Y0[ii*jj]

    ix = np.random.randint(low=0,high=len(X1),size=Np)
    X2, Y2 = X1[ix], Y1[ix]

    X2f, Y2f = get_uniques(X2,Y2)
    return X2f, Y2f

def get_positions_rectcircle_wire(slope1, slope2, width,c1=3, c2=5,f1=4,f2=2, m=20,Np=80,
                                  seed=None,n=6,z1=0.9,z2=0.9):
    """
    @params
        slope1: float
                Slope of the exterior border of the wire
        slope2: float
                Slope of the interior border of the wire
        width: float
                Width of the wire
        c1: float
                Factor of amplification of the exterior border length (in x-axis)
        c2: float
                Factor of amplification of the interior border length (in x-axis)
        f1: float
                Coefficient that controls the size of the unstranched exterior border 
        f2: float
                Coefficient that controls the size of the unstranched interior border
        m: int
            Number of subdivisions of the grid that is put over the wire
        Np: int
            Numer of particles
        seed: int
            Seed of the random values
        n: int
            Degree of the squircle
    """
    if seed!=None: np.random.seed(seed)
    if slope2<slope1: print('WARNING: slope2 is less than slope1')
    x0 = (np.arctanh(-z1)/slope1 + f1*width)*c1
    tt = np.linspace(-x0,x0,m)    
    X0, Y0 = np.meshgrid(tt,tt)
    Rho1 = (np.abs(X0/c1)**n+np.abs(Y0)**n)**(1/n)
    Rho2 = (np.abs(X0/c2)**n+np.abs(Y0)**n)**(1/n)
    ii = Rho1<=np.arctanh(-z1)/slope1 + f1*width
    jj = Rho2>=np.arctanh(z2)/slope2 + f2*width
    X1, Y1 = X0[ii*jj], Y0[ii*jj]

    ix = np.random.randint(low=0,high=len(X1),size=Np)
    X2, Y2 = X1[ix], Y1[ix]

    X2f, Y2f = get_uniques(X2,Y2)
    return X2f, Y2f


def get_time():
    date = [datetime.today().hour, datetime.today().minute, datetime.today().second]
    num = int(str(date[0]))
    if num>=12: ztime='PM'
    else: ztime='AM'
    time = str(date[0])+'.'+str(date[1])+' '+str(date[2])+' '+ztime
    return time