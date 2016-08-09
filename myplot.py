import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def SurveyLine(MAGNPos, MAGEPos, count):
    Northing = np.array(MAGNPos)
    Easting = np.array(MAGEPos)
    fig1 = plt.figure()
    a = fig1.add_subplot(111)
    plt.scatter(Easting, Northing)
    fig1.set_size_inches(25, 10.5)
    fig1.savefig('test1png.png', dpi=100)