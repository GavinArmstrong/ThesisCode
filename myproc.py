import numpy as np
import math

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def Gradline(MAG3N, MAG3E, index):
    x1 = MAG3E[index]
    x2 = MAG3E[index+1]
    y1 = MAG3N[index]
    y2 = MAG3N[index+1]
    deltaN = y2-y1
    deltaE = x2-x1
    dist = math.sqrt(deltaN**2+deltaE**2)
    if deltaE == 0 or deltaN == 0:
            theta = 0.0
    else:
        m = deltaN/deltaE
        theta = math.atan(-1/m)
        
    return deltaN, deltaE, theta, dist


def Gradients(MAGSET1,MAGNPos, MAGEPos, ALT, i):
    MAG2NPos = MAGNPos[i][1]
    MAG2EPos = MAGEPos[i][1]
    Altitude = ALT[i]
    dist = math.sqrt((MAGNPos[i+1][1]-MAGNPos[i][1])**2+(MAGEPos[i+1][1]-MAGEPos[i][1])**2)
    dtdx = (MAGSET1[i][2]-MAGSET1[i][0])/2.5
    Mid = (MAGSET1[i][2]+MAGSET1[i][0])/2
    NextMid = (MAGSET1[i+1][2]+MAGSET1[i+1][0])/2
    dtdy = (NextMid - Mid)/dist
    dtdz = (Mid - MAGSET1[i][1])/0.5
    N = 2.5
    l = (MAG2EPos * dtdx)+(MAG2NPos * dtdy) + (Altitude * dtdz) + (N * Mid)
        
    return Mid, dtdx, dtdy, dtdz, N, l
    
def CheckQuadrant(deltaN, deltaE, theta):
    theta12 = 0.0
    theta45 = 0.0
    if deltaN<0 and deltaE>0 or deltaN>0 and deltaE>0:
        theta12 = theta
        theta45 = np.pi + theta
    elif deltaN>0 and deltaE<0 or deltaN<0 and deltaE<0:
        theta12 = np.pi + theta
        theta45 = theta
    elif deltaN==0 and deltaE>0:
        theta12 = np.pi/2
        theta45 = (3*np.pi/2)
    elif deltaN == 0 and deltaE<0:
        theta12 = (3*np.pi/2)
        theta45 = np.pi/2
    elif deltaN>0 and deltaE==0:
        theta12 = np.pi
        theta45 = 0.0
    elif deltaN<0 and deltaE==0:
        theta12 = 0.0
        theta45 = np.pi
        
    return theta12, theta45
    
def CalcPos(MAG3N, MAG3E, theta12,theta45):
    b = 1.25
    MAG1N = MAG3N + 2*b*np.sin(theta12) 
    MAG1E = MAG3E + 2*b*np.cos(theta12)
    MAG2N = MAG3N + b*np.sin(theta12)
    MAG2E = MAG3E + b*np.cos(theta12)
    MAG4N = MAG3N + b*np.sin(theta45)
    MAG4E = MAG3E + b*np.cos(theta45)
    MAG5N = MAG3N + 2*b*np.sin(theta45)
    MAG5E = MAG3E + 2*b*np.cos(theta45)
    
    return MAG1N, MAG1E, MAG2N, MAG2E, MAG4N, MAG4E, MAG5N, MAG5E
