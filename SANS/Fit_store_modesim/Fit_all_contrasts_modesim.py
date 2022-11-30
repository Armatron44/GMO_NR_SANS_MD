import sys
import os

from bumps.names import *

sys.path.append('C:/Users/soo29949/Documents/sasview-main/src') #append src folder from sasview downloaded from github.
#this is so we can import sasmodels methods.

import sasmodels.core
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import load_data, set_beam_stop, set_top

""" IMPORT THE DATA USED """
load_data_d2 = load_data('d2.txt')
load_data_d3 = load_data('d3.txt')
load_data_d4 = load_data('d4.txt')

kernel2 = load_model(os.path.join(os.getcwd(), 'ellip_GMO_water_solv.py'))
#use the median values for the water_GMO contrasts, but use the mode for GMO contrast.


model1 = Model(kernel2,
    scale=3.561*1E-03,
    background=1.091*1E-02,
    Surf_SLD=0.21,
    Water_GMO_ratio=0,
    water_SLD=0,
    eccentricity=4.837*1E-01,
    volume=3.110*1E4,
    Solv_vf=0,
    sld_solvent=6.7
    )
    
model2 = Model(kernel2,
    scale=6.113*1E-03,
    background=8.928*1E-03,
    Surf_SLD=0.21,
    Water_GMO_ratio=5,
    water_SLD=-0.54,
    eccentricity=6.578*1E-01,
    volume=1.236*1E5,
    Solv_vf=0,
    sld_solvent=6.7
    )
    
model3 = Model(kernel2,
    scale=5.220*1E-03,
    background=1.620*1E-02,
    Surf_SLD=0.21, #model2.Surf_SLD,
    Water_GMO_ratio=5,
    water_SLD=6.37,
    eccentricity=model2.eccentricity,
    volume=model2.volume,
    Solv_vf=0,
    sld_solvent=6.7
    )

# SET THE FITTING PARAMETERS
model1.volume.name = 'GMO_volume'
model1.eccentricity.name = 'GMO_eccentricity'
model1.background.name = 'GMO_background'
model1.scale.name = 'GMO_scale'
model2.background.name = 'GMO_H2O_bkg'
model3.background.name = 'GMO_D2O_bkg'
model2.scale.name = 'GMO_H2O_scale'
model3.scale.name = 'GMO_D2O_scale'
model2.eccentricity.name = 'GMO_H2O_eccen'
model2.volume.name = 'GMO_H2O_volume'
model1.background.pmp(0.0000000001) #we allow the background to vary to such a small extent so that it doesn't change to a meaningful extent.
#this is because the --simulate method of bumps/refl1d requires a parameter to vary.

#set up the fitting objective.
M1 = Experiment(data=load_data_d2, model=model1)
M2 = Experiment(data=load_data_d3, model=model2)
M3 = Experiment(data=load_data_d4, model=model3)
problem = FitProblem([M1, M2, M3])