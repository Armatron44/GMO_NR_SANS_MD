import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from bumps.dream.state import load_state
from bumps.dream.views import plot_corrmatrix
from bumps.dream.stats import var_stats, format_vars, save_vars
import bumps.dream.varplot as varplot
from pylab import figure, savefig, suptitle, rcParams
from uravu.distribution import Distribution
import arviz as az
import corner
import scipy

#figfile = None
modelname = "P1-adv"
state = load_state(modelname)
state.mark_outliers() # ignore outlier chains
with open(modelname+".par") as fid:
    state.labels = [" ".join(line.strip().split()[:-1]) for line in fid]
	
# print all variables at the start. ###
draw = state.draw()
print(draw)
all_vstats = var_stats(draw)
print(format_vars(all_vstats))

alpha = 0.001
for i in range(0, draw.points.shape[1]):
    print(f"""{draw.labels[i]}
    Is gauss?: {'Yes' if scipy.stats.normaltest(draw.state.sample()[0].T[i])[1] >= alpha else 'No'}
    Gauss_test_p: {scipy.stats.normaltest(draw.state.sample()[0].T[i])[1]:.3E}""")

raise Exception()