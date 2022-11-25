import sys
import os

sys.path.append(os.getcwd())

import dynesty
import pickle
import numpy as np
from refnx.dataset import ReflectDataset
from refnx.analysis import Transform, CurveFitter, Objective, GlobalObjective, Parameter, process_chain, Parameters
from refnx.reflect import SLD, ReflectModel, MixedReflectModel, MixedSlab, Slab, Structure
from refnx.reflect.interface import Interface, Erf, Step
import refnx
import scipy

from vfp_M1 import VFP
import MixedMagSlabs2 #load in custom model for magnetic layers in non-polarised instrument.

# load data

file_path_GMO_H2O_ddod25 = '61115_17.txt'
file_path_GMO_D2O_ddod25 = '61146_48.txt'
file_path_GMO_D2O_hdod25 = '61150_52.txt'

# define refnx datasets

L1_GMO_H2O_ddod_25 = ReflectDataset(file_path_GMO_H2O_ddod25)
L1_GMO_D2O_ddod_25 = ReflectDataset(file_path_GMO_D2O_ddod25)
L1_GMO_D2O_hdod_25 = ReflectDataset(file_path_GMO_D2O_hdod25)

#now define some SLDs we will use...

"""
Model 1 as discussed in the main article.
"""
#LAYERS

Si = SLD(2.07, name='Si')
SiO2 = SLD(3.47, name='SiO2')
Fe = SLD(8.02, name='Fe')
FeOx = SLD(7.0, name='FeOx')
GMO = SLD(0.211387, name='GMO')
D2O = SLD(6.37097, name='D2O')
H2O = SLD(-0.558, name='H2O')
GMO_H2O_25_dd = SLD(0.21, name='GMO_H2O_25_dd')
GMO_D2O_25_dd = SLD(0.21, name='GMO_D2O_25_dd')
GMO_D2O_25_hd = SLD(0.21, name='GMO_D2O_25_hd')

#SOLVENTS

hdod_SLD_1 = SLD(-0.462, name='hdod1')
ddod_par_d2o = Parameter(6.7, name="ddodSLDpar_d2o", vary=True, bounds=(5, 6.7))
ddod_SLD_1 = SLD(ddod_par_d2o, name='ddod1') #use this for ddod-d2o

ddod_par = Parameter(6.7, name="ddodSLDpar", vary=True, bounds=(5, 6.7))
ddod_SLD_2 = SLD(ddod_par, name='ddod2') #use this for ddod-h2o

SiO2_lay = SiO2(15, 3)

SiO2_lay.thick.setp(vary=True, bounds=(1, 25))

##Fe##
# share the same Fe thickness, SiO2-Fe roughness - we specify variation of parameters here.
SiO2_Fe_R = Parameter(4, 'SiO2_Fe_R', vary=True, bounds=(1, 15))
Fe.real.setp(vary=True, bounds=(7.5, 8.1))
Fe_thick = Parameter(190, 'Fe_t', vary=True, bounds=(190, 230))
Fe_magmom = Parameter(2.1, 'Fe_magmom', vary=True, bounds=(1.5, 2.2))
Fe_ScattLen = Parameter(0.0000945, 'Fe_SL', vary=False)

FeUp_lay = MixedMagSlabs2.USM_Slab(Fe_thick, Fe, Fe_magmom, Fe_ScattLen, SiO2_Fe_R)
FeDown_lay = MixedMagSlabs2.DSM_Slab(Fe_thick, Fe, Fe_magmom, Fe_ScattLen, SiO2_Fe_R)

#FeOx
Fe_FeOx_R = Parameter(5, 'FeOx_Fe_R', vary=True, bounds=(1, 20))
FeOx.real.setp(vary=True, bounds=(5.0, 7.2))
FeOx_thick = Parameter(30, 'FeOx_t', vary=True, bounds=(20, 60))
FeOx_m = Parameter(0.5, 'FeOx_m', vary=True, bounds=(0, 1.3))

FeOxUp_lay = MixedMagSlabs2.USM_nomagmom_Slab(FeOx_thick, FeOx, FeOx_m, Fe_FeOx_R)
FeOxDown_lay = MixedMagSlabs2.DSM_nomagmom_Slab(FeOx_thick, FeOx, FeOx_m, Fe_FeOx_R)

# INTERFACIAL LAYERS#

FeOx_R = Parameter(1, 'FeOx_R', vary=True, bounds=(1, 15))

# comb GMO_25 #
GMO_25_t = Parameter(25, 'GMO_25_t', vary=True, bounds=(10, 40))
GMO_wat_vol_frac = Parameter(0, 'water_vol_frac', vary=True, bounds=(0, 1))
Solvation = Parameter(0, 'solvation', vary=True, bounds=(0, 1))

GMO_25_H2O_dd_lay = GMO_H2O_25_dd(GMO_25_t, FeOx_R)
GMO_25_D2O_dd_lay = GMO_D2O_25_dd(GMO_25_t, FeOx_R)
GMO_25_D2O_hd_lay = GMO_D2O_25_hd(GMO_25_t, FeOx_R)

GMO_25_H2O_dd_lay.sld.real.setp(constraint=(GMO_wat_vol_frac*H2O.real+(1-GMO_wat_vol_frac)*GMO.real))
GMO_25_H2O_dd_lay.vfsolv.setp(constraint=Solvation)

GMO_25_D2O_dd_lay.sld.real.setp(constraint=(GMO_wat_vol_frac*D2O.real+(1-GMO_wat_vol_frac)*GMO.real))
GMO_25_D2O_dd_lay.vfsolv.setp(constraint=Solvation)

GMO_25_D2O_hd_lay.sld.real.setp(constraint=(GMO_wat_vol_frac*D2O.real+(1-GMO_wat_vol_frac)*GMO.real))
GMO_25_D2O_hd_lay.vfsolv.setp(constraint=Solvation)

###INTF-SOLV ROUGHNESS###

GMO_25_R = Parameter(7, 'GMO_25_R', vary=True, bounds=(1, 15))

### SOLVENT PARAMS FOR ADV_LAYER SOLVENT

hd_d2o_GMO25_lay = hdod_SLD_1(0, GMO_25_R)
dd_d2o_GMO25_lay = ddod_SLD_1(0, GMO_25_R)
dd_h2o_GMO25_lay = ddod_SLD_2(0, GMO_25_R)

"""
This is where we deviate from ordinary models.
"""
### now define thicknesses, roughnesses, parameters for constraints & slds to send to the VFPs.

list_of_thickness = (0, SiO2_lay.thick, Fe_thick, FeOx_thick, GMO_25_t) #first must be zero for Si thickness.
list_of_roughness = (SiO2_lay.rough, SiO2_Fe_R, Fe_FeOx_R, FeOx_R, GMO_25_R)

"""
Pass the SLDs to the VFPs. 
The SLDs are ordered as the following:
Si, SiO2, FeUp, FeDown, FeOxUp, FeOxDown, 
    GMO_dd_h2o, GMO_dd_d2o, GMO_hd_d2o, 
    dd_d2o, dd_h2o, hd_d2o
"""
SLDs = np.array([Si.real, SiO2.real, FeUp_lay.sld_n.real+(((FeUp_lay.sld_n.real*1E-6)/Fe_ScattLen)*Fe_magmom*0.00002699*1E6),
        FeDown_lay.sld_n.real-(((FeUp_lay.sld_n.real*1E-6)/Fe_ScattLen)*Fe_magmom*0.00002699*1E6), (FeOxUp_lay.sld_n.real+FeOx_m),
        (FeOxDown_lay.sld_n.real-FeOx_m), (GMO_25_H2O_dd_lay.sld.real*(1-Solvation)+Solvation*dd_h2o_GMO25_lay.sld.real),
        (GMO_25_D2O_dd_lay.sld.real*(1-Solvation)+Solvation*dd_d2o_GMO25_lay.sld.real), 
        (GMO_25_D2O_hd_lay.sld.real*(1-Solvation)+Solvation*hd_d2o_GMO25_lay.sld.real),
        dd_d2o_GMO25_lay.sld.real, dd_h2o_GMO25_lay.sld.real, hd_d2o_GMO25_lay.sld.real])

#Create an extent parameter. 
#This is the length of the whole volume fraction profiles & the resulting SLD curve so that the erfs approach 1.
#we have to include the roughness of the first interface and the roughness of the last interface (similar to how the SLD curves of a normal structure is calculated).
VFP_tot_thick = Parameter(name='VFP_tot_thick', constraint=5+(4 * SiO2_lay.rough)+SiO2_lay.thick+Fe_thick+FeOx_thick+GMO_25_t+5+(4 * GMO_25_R))

#create VFPs

dd_d2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='dd_d2o_up')
dd_d2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='dd_d2o_down')
dd_h2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='dd_h2o_up')
dd_h2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='dd_h2o_down')
hd_d2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='hd_d2o_up')
hd_d2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, contrast='hd_d2o_down')

#we need the VFP components to be placed within a structure with a fronting and backing.
#although we already calculate the fronting and backing in the VFP, we need to specify them here.
#this doesn't really complicate the situation.
dd_d2o_up_stack = Si | dd_d2o_up_vfp | ddod_SLD_1
dd_d2o_down_stack = Si | dd_d2o_down_vfp | ddod_SLD_1
dd_h2o_up_stack = Si | dd_h2o_up_vfp | ddod_SLD_2
dd_h2o_down_stack = Si | dd_h2o_down_vfp | ddod_SLD_2
hd_d2o_up_stack = Si | hd_d2o_up_vfp | hdod_SLD_1
hd_d2o_down_stack = Si | hd_d2o_down_vfp | hdod_SLD_1

#intensities
L1_GMO25dd_d2o_int = Parameter(0.5, 'L1_GMO25dd_d2o_int', vary=True, bounds=(0.5*0.8, 0.5*1.2))
L1_GMO25dd_h2o_int = Parameter(0.5, 'L1_GMO25dd_h2o_int', vary=True, bounds=(0.5*0.8, 0.5*1.2))
L1_GMO25hd_d2o_int = Parameter(0.5, 'L1_GMO25hd_d2o_int', vary=True, bounds=(0.5*0.8, 0.5*1.2))

#backgrounds
L1_GMO25dd_d2o_bkg = Parameter(1e-6, 'L1_GMO25dd_d2o_bkg', vary=True, bounds=(1e-7, 2e-5))
L1_GMO25dd_h2o_bkg = Parameter(1e-6, 'L1_GMO25dd_h2o_bkg', vary=True, bounds=(1e-7, 2e-5))
L1_GMO25hd_d2o_bkg = Parameter(1e-6, 'L1_GMO25hd_d2o_bkg', vary=True, bounds=(1e-7, 2e-5))

STD_to_FWHM = 2*(2*np.log(2))**(1/2)

#models
L1_GMO25dd_d2o_M = MixedReflectModel((dd_d2o_up_stack, dd_d2o_down_stack), scales=(L1_GMO25dd_d2o_int, L1_GMO25dd_d2o_int), bkg=L1_GMO25dd_d2o_bkg, dq=3*STD_to_FWHM)
L1_GMO25dd_h2o_M = MixedReflectModel((dd_h2o_up_stack, dd_h2o_down_stack), scales=(L1_GMO25dd_h2o_int, L1_GMO25dd_h2o_int), bkg=L1_GMO25dd_h2o_bkg, dq=3*STD_to_FWHM)
L1_GMO25hd_d2o_M = MixedReflectModel((hd_d2o_up_stack, hd_d2o_down_stack), scales=(L1_GMO25hd_d2o_int, L1_GMO25hd_d2o_int), bkg=L1_GMO25hd_d2o_bkg, dq=3*STD_to_FWHM)

#make sure to pass the normal layer parameters to the auxiliary params of the objectives.
auxiliary_parameters = (Solvation, GMO_wat_vol_frac, SiO2_lay.thick, SiO2_Fe_R, Fe_thick, Fe.real, Fe_magmom,
                        Fe_ScattLen, Fe_FeOx_R, FeOx.real, FeOx_thick, FeOx_m, FeOx_R, GMO_25_t, GMO_25_R,
                        ddod_par_d2o, ddod_par)

#objectives
L1GMOdd_d2o_25_obj = Objective(L1_GMO25dd_d2o_M, L1_GMO_D2O_ddod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)
L1GMOdd_h2o_25_obj = Objective(L1_GMO25dd_h2o_M, L1_GMO_H2O_ddod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)
L1GMOhd_d2o_25_obj = Objective(L1_GMO25hd_d2o_M, L1_GMO_D2O_hdod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)

#glob_objective
glob_obj = GlobalObjective([L1GMOdd_d2o_25_obj, L1GMOdd_h2o_25_obj, L1GMOhd_d2o_25_obj])

pars = (SiO2_lay.thick, SiO2_Fe_R)

class LogpExtra(object):
    def __init__(self, pars):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.pars = pars

    def __call__(self, model, data):
        if float(self.pars[0]-2*self.pars[1]) >= 0:
            return 0
        return -1E15

lpe = LogpExtra(pars)

L1GMOdd_d2o_25_obj.logp_extra = lpe

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    from schwimmbad import MPIPool
    with MPIPool() as p:
        if not p.is_master():
            p.wait()
            sys.exit(0)
        nested_sampler = dynesty.NestedSampler(glob_obj.logl, glob_obj.prior_transform, ndim=len(glob_obj.varying_parameters()), nlive=500, pool=p)
        nested_sampler.run_nested(dlogz=0.509, print_progress=True)
        p.close()
    res = nested_sampler.results
    print(res.logz[-1], res.logzerr[-1])
    print('Keys:', res.keys(),'\n')  # print accessible keys
    res.summary()  # print a summary
    with open(os.path.join(os.getcwd(), 'M1_GO.pkl'), 'wb+') as f:
        pickle.dump(glob_obj, f)
    
    with open(os.path.join(os.getcwd(), 'M1_NS.pkl'), 'wb+') as f:
        pickle.dump(res, f)
