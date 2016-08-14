import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
from min_box import minimum_bounding_rectangle
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

df = pd.read_csv('160465_20160227_SIT_3m_PORT.csv').dropna()

df.MAG3_NORTH -= 6000000
df.MAG3_EAST -= 200000

# Doing calculations as vectorised operations

# Just an example, these formulae are prob wrong.
MAG3pos = df[['MAG3_NORTH', 'MAG3_EAST']].values

# we want to get the gradient per timestep then add them up
deltas = np.gradient(MAG3pos, axis=0)

deltaN = np.diff(MAG3pos[:,0])
deltaE = np.diff(MAG3pos[:,1])

#THIS FUNCTIONS DOES SOMETHING A LITTLE DIFFERENT
gradients = (deltas[:,0]/deltas[:,1])
#IS THERE A BETTER WAY TO CONDITIONALLY CHANGE VALUE IN AN ARRAY
heading = (np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi)
heading = np.where(heading<0,(np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi) + 360,(np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi) )
#heading = (np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi)
#heading = np.where(heading<0,(np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi) + 360,(np.arctan2(deltas[:,0],deltas[:,1]) * 180/np.pi) )

#CALCULATING POSITION OF MAGNETOMETERS
MAG2E =  MAG3pos[:,1] + 1.25 * np.sin(np.radians(heading+90))
MAG2N = MAG3pos[:,0] + 1.25 * np.sin(np.radians(heading+90))
MAG1E = MAG3pos[:,1] + 2.5 * np.sin(np.radians(heading+90))
MAG1N = MAG3pos[:,0] + 2.5 * np.sin(np.radians(heading+90))
MAG5E = MAG3pos[:,1] + 2.5 * np.sin(np.radians(heading-90))
MAG5N = MAG3pos[:,0] + 2.5 * np.sin(np.radians(heading-90))
MAG4E = MAG3pos[:,1] + 1.25 * np.sin(np.radians(heading-90))
MAG4N = MAG3pos[:,0] + 1.25 * np.sin(np.radians(heading-90))

# get distance on each timestep and we add them all
distances = (deltas[:,0]**2 + deltas[:, 1]**2)**0.5

transverse = (df.MAG3 - df.MAG1) / 2.5
MidMag = ((df.MAG3 + df.MAG1) / 2.0)
longitudinal = (MidMag-MidMag.shift()).shift(-1) / distances
vertical = (((df.MAG3 + df.MAG1) / 2.0)-df.MAG2) / 0.5

df['longitudinal'] = longitudinal
df['transverse'] = transverse
df['vertical'] = vertical
df['mid1-3'] = MidMag

AS = np.sqrt(df.longitudinal**2 + df.transverse**2 + df.vertical**2)

df['AnalyticalSignal'] = AS
df['MAG1_NORTH'] = MAG1N
df['MAG1_EAST'] = MAG1E
df['MAG2_NORTH'] = MAG2N
df['MAG2_EAST'] = MAG2E
df['MAG4_NORTH'] = MAG4N
df['MAG4_EAST'] = MAG4E
df['MAG5_NORTH'] = MAG5N
df['MAG5_EAST'] = MAG5E

#
#Delete last value of each group as the computation is incorrect due to being the transition of two lines
#THINK I NEED TO DELETE THE FIRST ELEMENT AS WELL (Zooming in on the plots, I noticed that values at the end and beginning of the lines are incorrect)
#Deleting first and last 10 points gets rid of most wobbly starts to the survey line
df = df[df.groupby('LINE').cumcount(ascending=False) > 20]
df = df[df.groupby('LINE').cumcount(ascending=True) > 20]

#PLOTTING
#ax = df.plot(kind='scatter', x='MAG3_EAST', y='MAG3_NORTH', color='DarkBlue', label='MAG3')
#df.plot(kind='scatter', x='MAG2_EAST', y='MAG2_NORTH',color='DarkGreen', label='MAG2', ax=ax)
#df.plot(kind='scatter', x='MAG4_EAST', y='MAG4_NORTH',color='DarkRed', label='MAG4', ax=ax)
#df.plot(kind='scatter', x='MAG5_EAST', y='MAG5_NORTH',color='DarkOrange', label='MAG5', ax=ax)
#df.plot(kind='scatter', x='MAG1_EAST', y='MAG1_NORTH',color='DarkCyan', label='MAG1', ax=ax)

#HAVE TO GROUP BY LINE BEFORE COMPUTATIONS OTHERWISE WILL GET INCORRECT VALUES AT THE TRANSITION BETWEEN TWO LINES
#Northing = np.array([])
for (line, group) in df.groupby('LINE'):
    MAG1p = group[['MAG1_NORTH', 'MAG1_EAST']].values.tolist()
    MAG1v = group['MAG1'].values.tolist()
    MAG2p = group[['MAG2_NORTH', 'MAG2_EAST']].values.tolist()
    MAG2v = group['MAG2'].values.tolist()
    MAG3p = group[['MAG3_NORTH', 'MAG3_EAST']].values.tolist()
    MAG3v = group['MAG3'].values.tolist()
    MAG4p = group[['MAG4_NORTH', 'MAG4_EAST']].values.tolist()
    MAG4v = group['MAG4'].values.tolist()
    MAG5p = group[['MAG5_NORTH', 'MAG5_EAST']].values.tolist()
    MAG5v = group['MAG5'].values.tolist()
    MAGp = np.asarray(MAG1p + MAG2p + MAG3p+ MAG4p + MAG5p)
    MAGv = np.asarray(MAG1v + MAG2v + MAG3v + MAG4v + MAG5v)
    #different method
    corners = minimum_bounding_rectangle(MAGp)
#    Hull = ConvexHull(MAGp, incremental=True)
    plt.figure()
    plt.plot(MAGp[:,0], MAGp[:,1], 'ko',  markersize=2)
    plt.plot(corners[:,0],corners[:,1], 'bo-')
    
    #THIS GRIDS IN THE AREA BOUND BY THE CORNERS, BUT NOT THE ACTUAL RECTANGLE
    x = np.arange(min(corners[:,1]),max(corners[:,1]),1)
    y = np.arange(min(corners[:,0]),max(corners[:,0]),1)
    grid_x, grid_y = np.meshgrid(x,y)
    gridded = griddata(MAGp, MAGv, (grid_y, grid_x), method = 'linear', fill_value = 0)
#    for simplex in Hull.simplices:
#        plt.plot(MAGp[simplex, 0], MAGp[simplex, 1], 'k-')
#    plt.plot(MAGp[Hull.vertices,0], MAGp[Hull.vertices,1], 'r--', lw=2)
#    for vertex in Hull.vertices:
#        plt.plot(MAGp[vertex,0], MAGp[vertex,1], 'ro')
#    plt.show()
    
#    ax = group.plot(kind='scatter', x='MAG3_EAST', y='MAG3_NORTH', color='DarkBlue', label='MAG3')
#    group.plot(kind='scatter', x='MAG2_EAST', y='MAG2_NORTH',color='DarkGreen', label='MAG2', ax=ax)
#    group.plot(kind='scatter', x='MAG4_EAST', y='MAG4_NORTH',color='DarkRed', label='MAG4', ax=ax)
#    group.plot(kind='scatter', x='MAG5_EAST', y='MAG5_NORTH',color='DarkOrange', label='MAG5', ax=ax)
#    group.plot(kind='scatter', x='MAG1_EAST', y='MAG1_NORTH',color='DarkCyan', label='MAG1', ax=ax)
#    plt.subplot(211)
#    plt.plot(group.AnalyticalSignal, label="AS")
#    plt.legend()
#    plt.show()
    
#    #Get position of MAG3 along line
#    positions = group[['MAG3_NORTH', 'MAG3_EAST']].values
#    #Get gradient (deltaN and deltaE) along line
#    gradients = np.gradient(positions, axis=0)
#    #Get distance between points along line
#    distances = (gradients[:,0]**2 + gradients[:, 1]**2)**0.5
#    #Calculate transverse, longitudinal and vertical gradients
#    transverse = (group.MAG3 - group.MAG1) / 2.5
#    MidMag = ((group.MAG3 + group.MAG1) / 2.0)
#    longitudinal = (MidMag-MidMag.shift()).shift(-1) / distances #shift the column down by one (to align)
#    vertical = (((group.MAG3 + group.MAG1) / 2.0)-group.MAG2) / 0.5
#    
#    group['longitudinal'] = longitudinal
#    group['transverse'] = transverse
#    group['vertical'] = vertical


