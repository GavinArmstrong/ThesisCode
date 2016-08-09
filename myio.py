import csv
from itertools import groupby

class AObs(list):
    def __init__(self):
        pass

def readObsFile(filename):
    file = open(filename,'r')

    linenumber = 0

    observations = AObs()          # set with observations
    observationsnew = AObs()
    count = 0
    for line in file:
        if linenumber > 1:              # exclude header
            line = line.strip()
            if len(line) > 0:
                sp = line.split(',')
                while '' in sp:
                    sp.remove('')
                obs = {}                 # observations dictionary
                if len(sp) == 17:
                    obs['MAG1'] = float(sp[0])
                    obs['TIME'] = sp[3].strip()
                    obs['MAG2'] = float(sp[4])
                    obs['MAG3'] = float(sp[6])
                    obs['MAG4'] = float(sp[8])
                    obs['MAG5'] = float(sp[10])
                    obs['LINE'] = sp[13]
                    obs['MAG3_NORTH'] = float(sp[15])
                    obs['MAG3_EAST'] = float(sp[16])
                    obs['ALTITUDE'] = float(sp[12])
                count += 1
                observations += [obs]    # append observations dictionary to set
                
                observations = [x for x in observations if x] # remove empty dictionaries
        linenumber += 1

    file.close()

    return observations