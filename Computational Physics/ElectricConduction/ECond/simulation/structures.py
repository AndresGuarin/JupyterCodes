"""
Created on Mon Jan 30 15:01:00 2025

This code give us the positions of the positive cores that form the 
conductor.
"""

import ECond.simulation as sm
from ECond.simulation.mesh import super_concatenate

# structures
def squared_wire(P,a,d,w,h):
    """ 
    Structure of a closed wire in the form of a square with an squared lattice.

    @params
        P: list. It contains the initial position of lower-left conner the grid.
        a: float. Lattice constant. 
        d: int. Depth of the wire. It is measured in number of atoms.
        w: int > d. Width of the square. It is measured in number of atoms.
        h: int > d. Height of the squre. It is measured in number of atoms.

    """
    P0, P1 = P[0], P[1]
    XC, YC = super_concatenate([sm.mesh.squared(P=[P0,P1],nrows=h,ncols=d,a=a),
                                sm.mesh.squared(P=[P0 + a*d,P1],nrows=d,ncols=w-2*d,a=a),
                                sm.mesh.squared(P=[P0 + a*(w-d),P1],nrows=h,ncols=d,a=a),
                                sm.mesh.squared(P=[P0 + a*d,P1+a*(h-d)],nrows=d,ncols=w-2*d,a=a)
                                ])
    return XC, YC