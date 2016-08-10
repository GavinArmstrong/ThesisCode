import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


df = pd.read_csv('160465_20160227_SIT_3m_PORT.csv').dropna()

df.MAG3_NORTH -= 6000000
df.MAG3_EAST -= 200000

df.plot(kind='scatter', x='MAG3_EAST', y='MAG3_NORTH')

# Doing calculations as vectorised operations

# Just an example, these formulae are prob wrong.
positions1 = df[['MAG3_NORTH', 'MAG3_EAST']].values

# we want to get the gradient per timestep then add them up
gradients = np.gradient(positions1, axis=0)
        
theta = np.arctan(-1/(gradients[:,0]/gradients[:,1]))

# get distance on each timestep and we add them all
distances = (gradients[:,0]**2 + gradients[:, 1]**2)**0.5

transverse = (df.MAG3 - df.MAG1) / 2.5
MidMag = ((df.MAG3 + df.MAG1) / 2.0)
longitudinal = (MidMag-MidMag.shift()).shift(-1) / distances
vertical = (((df.MAG3 + df.MAG1) / 2.0)-df.MAG2) / 0.5

df['longitudinal'] = longitudinal
df['transverse'] = transverse
df['vertical'] = vertical
df['mid1-3'] = MidMag
#
#Delete last value of each group as the computation is incorrect due to being the transition of two lines
df = df[df.groupby('LINE').cumcount(ascending=False) > 0]

#HAVE TO GROUP BY LINE BEFORE COMPUTATIONS OTHERWISE WILL GET INCORRECT VALUES AT THE TRANSITION BETWEEN TWO LINES
#for (line, group) in df.groupby('LINE'):
    
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


