import numpy as np
import math
from myio import readObsFile
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import myproc as pr
import myplot as pt
import pandas as pd
from matplotlib.mlab import griddata
observations= readObsFile('160465_20160227_SIT_3m_PORT.csv')
#observations = readObsFile('160465_20160227_SIT_3m_STBD.csv')
#observations = readObsFile('Line2016.csv')
#observations = readObsFile('160465_20160227_SIT_5m_PORT.csv')
#observations = readObsFile('160465_20160227_SIT_5m_STBD.csv')
s=set()
line=[]
#List of line names used as a check
for i in range(len(observations)):
    
    line = observations[i]['LINE']
    s.add(line)
    line=list(s)

MAGSET1 = []
MAGSET2 = []
MAGSETFULL = []
M1 = []
M2 = []
M3 = []
hour=[]
MAG3N = []
MAG3E = []
ALT=[]
countlist = []
count = 0

for O in observations:
    if O['LINE'] == '2019':
        MAG1 = O['MAG1']
        MAG2 = O['MAG2']
        MAG3 = O['MAG3']
        MAG4 = O['MAG4']
        MAG5 = O['MAG5']
        TIME = O['TIME']
        ALTITUDE = O['ALTITUDE']
        MAGNorth = O['MAG3_NORTH']
        MAGEast = O['MAG3_EAST']
        countlist+=[count]
        LINE = O['LINE']
        M1 += [MAG1]
        M2 += [MAG2]
        M3 += [MAG3]
        MAGSET1 += [[MAG1, MAG2, MAG3]]
        MAGSET2 += [[MAG3, MAG4, MAG5]]
        MAGSETFULL += [[MAG1, MAG2, MAG3, MAG4, MAG5]]
        MAG3N +=[MAGNorth]
        MAG3E += [MAGEast]
        ALT += [ALTITUDE]
        count += 1



## ATTEMPT TO REDUCE TO THE POLE  ## Dont worry about this at the moment (this will be done after gridding the data)
#count = 0
#newl1 = []
#newl2 = []
#newl3 = []
#total = 0
#mean1 =[]
#mean2 = []
#mean3 = []
#for c1,c2, c3 in zip(pr.chunks(M1, 5), pr.chunks(M2, 5), pr.chunks(M3, 5)):
#    sd1 = np.std(c1)
#    sd2 = np.std(c2)
#    sd3 = np.std(c3)
#    carr1 = np.array(c1)
#    carr2 = np.array(c2)
#    carr3 = np.array(c3)
#    if sd1 > 0.3 or sd2 > 0.3 or sd3> 0.3:
#        m1 = np.mean(mean1)
#        m2 = np.mean(mean2)
#        m3 = np.mean(mean3)
#        newl1.append((carr1-m1).tolist())
#        newl2.append((carr2-m2).tolist())
#        newl3.append((carr3-m3).tolist())
#    else:
#        mean1  += [np.mean(c1)]
#        mean2 += [np.mean(c2)]
#        mean3 += [np.mean(c3)]
#        m1 = np.mean(c1)
#        m2 = np.mean(c2)
#        m3 = np.mean(c3)
#        newl1.append((carr1-m1).tolist())
#        newl2.append((carr2-m2).tolist())
#        newl3.append((carr3-m3).tolist())
#    count += 1
#    
#
#newl1 = [item for sublist in newl1 for item in sublist]
#newl2 = [item for sublist in newl2 for item in sublist]
#newl3 = [item for sublist in newl3 for item in sublist]
#
#MAGSET1n =[]
#for i in range(len(newl1)):
#    MAGSET1n += [[newl1[i],newl2[i], newl3[i]]]
#def movingaverage(interval, window_size):
#    window = np.ones(int(window_size))/float(window_size)
#    return np.convolve(interval, window, 'same')
#
#MA = movingaverage(M1, 3)

#Average
#M1=[]
#avg = [float(sum(col))/len(col) for col in zip(*MAGSET1)]
#for s in MAGSET1:
#    M1 += [[a-b for a,b in zip(s, avg)]]

MAGNPos =[]
MAGEPos =[]

i=0
for i in range(len(MAG3N)):
    if i<(len(MAG3N)-1):
        deltaN, deltaE, theta, dist= pr.Gradline(MAG3N, MAG3E, i)
        if dist<1:
            theta12, theta45 = pr.CheckQuadrant(deltaN, deltaE, theta)
            MAG1N, MAG1E, MAG2N, MAG2E, MAG4N, MAG4E, MAG5N, MAG5E = pr.CalcPos(MAG3N[i], MAG3E[i], theta12, theta45)
#        if i>100 and i<300:
            MAGNPos += [[MAG1N, MAG2N, MAG3N[i], MAG4N, MAG5N]]
            MAGEPos +=[[MAG1E, MAG2E, MAG3E[i], MAG4E, MAG5E]]
        else:
            MAGSETFULL = MAGSETFULL[:i] + MAGSETFULL[i+1 :]
        i += 1
del MAGSETFULL[-1]
##
flatMSET = np.array([val for sublist in MAGSETFULL for val in sublist])
flatMN = np.array([val for sublist in MAGNPos for val in sublist])
flatME= np.array([val for sublist in MAGEPos for val in sublist])
##
xi = np.linspace(int(min(flatME)), int(max(flatME)),(int(max(flatME)) - int(min(flatME)))*2)
yi = np.linspace(int(min(flatMN)), int(max(flatMN)), (int(max(flatMN)) - int(min(flatMN)))*2)
zi = griddata(flatME,flatMN,flatMSET,xi,yi)
#
#contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
#:
#pt.SurveyLine(MAGNPos, MAGEPos, countlist)
#
## MAGPositions put into csv file, was used to visualise in excel
#df = pd.DataFrame(MAGNPos)
#df2 = pd.DataFrame(MAGEPos)
#df.to_csv('MAGNPos.csv', index=False,header=False)
#df2.to_csv('MAGEPos.csv', index=False,header=False)
##

# Calculating the tranverse, longitudinal and vertical gradient for the analytical signal
diff = []
MidMag = []
L = []
for i in range(len(MAGSET1n)):
    if i<(len(MAGSET1n)-2):
        Mid, dtdx, dtdy, dtdz, N, l = pr.Gradients(MAGSET1n,MAGNPos, MAGEPos, ALT, i)
        diff += [[dtdx, dtdy, dtdz, N]]
        L += [l]
        MidMag += [Mid]

## Least Squares process, currently not looking at
#Xlist = []
#i = 0
#for A,l, NP, EP in zip(pr.chunks(diff, 10),pr.chunks(L, 10), MAGNPos, MAGEPos):
#    if len(A)>=10 and len(l)>=10:
#        Amat = np.matrix(A)
#        Lmat = np.matrix(l).T
#        ATAinv = (Amat.T * Amat).I
#        print(Amat)
#        ATl = Amat.T * Lmat
#        X = ATAinv * ATl
#        Xlist.append(X.tolist())
#        i = i+1
#Easting = []
#Northing = []
#Depth = []
#i=0
#MidNorth = MAG3N[::25]
#MidNorth.pop(0)
#MidNorth = MidNorth[::2]
#MidEast = MAG3E[::25]
#MidEast.pop(0)
#MidEast = MidEast[::2]
#i=0
#for x in Xlist:
#    Easting += x[0]
#    Northing += x[1]
#    Depth += x[2]
#fig = plt.figure()
#plt.scatter(Easting, Northing)
##plt.scatter(MidEast, MidNorth, s=50)
#i = 0
#for p1, p2 in zip(MidNorth, Northing):
#    d = abs(p2-p1)
#    if d < 0.3:
#        print(i)
#    i += 1

##PLOTTING CODE         
#MAGGY1 = np.array(newl1)
#Count2 = np.array(countlist)
##MAGGY2 = np.array(M2)
##MAGGY3 = np.array(M3)
###MAGGY4 = np.array(M4)
###MAGGY5 = np.array(M5)
#fig1 = plt.figure()
#a = fig1.add_subplot(111)
##plt.xticks(np.arange(min(Count2), max(Count2)+1, 10))
#plt.plot(Count2, MAGGY1, '.r-', label = 'MAG1')
#plt.plot(Count2, MAGGY2, '.b-', label = 'MAG2')
#plt.plot(Count2, MAGGY3, '.g-', label = 'MAG3')
#plt.plot(Count2, MAGGY4, '.y-', label = 'MAG4')
#plt.plot(Count2, MAGGY5, '.c-', label = 'MAG5')
#
#box = a.get_position()
#a.set_position([box.x0, box.y0 + box.height * 0.1,
#                 box.width, box.height * 0.9])
#
## Put a legend below current axis
#a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#          fancybox=True, shadow=True, ncol=5)
#
#fig1.set_size_inches(25, 10.5)
#fig1.savefig('test2png.png', dpi=100)
#
#a.set_title('Maggy Readings for each magnetometer in the array')