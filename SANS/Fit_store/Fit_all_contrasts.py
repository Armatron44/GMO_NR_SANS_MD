import sys

sys.path.append('C:/Users/soo29949/Documents/sasview-main/src')

from bumps.names import *

import sasmodels.core
from sasmodels.core import load_model
from sasmodels.bumps_model import Model, Experiment
from sasmodels.data import load_data, set_beam_stop, set_top

""" IMPORT THE DATA USED """
load_data_d2 = load_data('d2.txt')
load_data_d3 = load_data('d3.txt')
load_data_d4 = load_data('d4.txt')

kernel2 = load_model("rep_ellip_waterGMOsolv_ratio_v3")

model1 = Model(kernel2,
    scale=0.0033,
    background=0.0097489,
    Surf_SLD=0.21,
    Water_GMO_ratio=0,
    water_SLD=0,
    eccentricity=1.88,
    volume=1.3e5,
    Solv_vf=0,
    sld_solvent=6.7
    )
    
model2 = Model(kernel2,
    scale=0.01,
    background=0.0097489,
    Surf_SLD=0.21,
    Water_GMO_ratio=5,
    water_SLD=-0.54,
    eccentricity=1.5,
    volume=1.3e5,
    Solv_vf=0,
    sld_solvent=6.7
    )
    
model3 = Model(kernel2,
    scale=0.01,
    background=0.0097489,
    Surf_SLD=0.21, #model2.Surf_SLD,
    Water_GMO_ratio=5,
    water_SLD=6.37,
    eccentricity=model2.eccentricity,
    volume=model2.volume,
    Solv_vf=0,
    sld_solvent=6.7
    )

# SET THE FITTING PARAMETERS
model1.volume.range(0, 1E8)
model1.volume.name = 'GMO_volume'
model1.eccentricity.range(0.333, 3)
model1.eccentricity.name = 'GMO_eccentricity'
model1.background.range(1E-3, 1)
model1.background.name = 'GMO_background'
model1.scale.range(1E-6, 1)
model1.scale.name = 'GMO_scale'
model2.background.range(1E-3, 1)
model2.background.name = 'GMO_H2O_bkg'
model3.background.range(1E-3, 1)
model3.background.name = 'GMO_D2O_bkg'
model2.scale.range(1E-6, 1)
model2.scale.name = 'GMO_H2O_scale'
model3.scale.range(1E-6, 1)
model3.scale.name = 'GMO_D2O_scale'
model2.eccentricity.range(0.333, 3)
model2.eccentricity.name = 'GMO_H2O_eccen'
model2.volume.range(0, 1E8)
model2.volume.name = 'GMO_H2O_volume'

## constraints ##
# def const():
    # m2vs, m2vt = model2.vol_solv.value, model2.volume.value
    # return 0 if (m2vt-m2vs) >= 0 else 1e15

#set up the fitting objective.
M1 = Experiment(data=load_data_d2, model=model1)
M2 = Experiment(data=load_data_d3, model=model2)
M3 = Experiment(data=load_data_d4, model=model3)
problem = FitProblem([M1, M2, M3])# constraints=const, soft_limit=1e15)