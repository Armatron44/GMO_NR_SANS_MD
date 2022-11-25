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
import matplotlib.pyplot as plt

from vfp_M2 import VFP
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
Model 2 as discussed in the main article.
"""
#LAYERS

Si = SLD(2.07, name='Si')
SiO2 = SLD(3.47, name='SiO2')
Fe = SLD(8.02, name='Fe')
FeOx = SLD(7.0, name='FeOx')
D2O = SLD(6.37097, name='D2O')
GMO = SLD(0.211387, name='<GMO>')
GMO_head_s_d2o = SLD(0.377787, name='GMO_head_s_d2o')
GMO_head_s_h2o = SLD(0.377787, name='GMO_head_s_h2o')
H2O = SLD(-0.558, name='H2O')
GMO_tail_s_d2o = SLD(-0.16640, name='GMO_tail_s_d2o')
GMO_tail_s_h2o = SLD(-0.16640, name='GMO_tail_s_h2o')
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

# GMO_head
b_head = 0.00023623 # Å
GMO_head_par = Parameter(1, 'GMO_head_par', vary=True, bounds=(0, 2))# for surface excess we have to fit the sld of the head and tail group (volumes vary)
GMO_head = SLD(GMO_head_par, name='GMO_head')

GMO_vol_frac = Parameter(0.5, 'GMO_vol_frac', vary=True, bounds=(0, 1))
head_solvation = Parameter(0.05, 'head_solvation', vary=True, bounds=(0, 1))

G_head_t = Parameter(5, 'GMO_head_t', vary=True, bounds=(1, 6.6))
G_head_r = Parameter(1, 'GMO_head_r', vary=True, bounds=(1, 6.6))

GMO_head_lay_d2o = GMO_head_s_d2o(G_head_t, FeOx_R)
GMO_head_lay_d2o.sld.real.setp(constraint=(GMO_vol_frac*GMO_head.real+(1-GMO_vol_frac)*D2O.real))

GMO_head_lay_h2o = GMO_head_s_h2o(G_head_t, FeOx_R)
GMO_head_lay_h2o.sld.real.setp(constraint=(GMO_vol_frac*GMO_head.real+(1-GMO_vol_frac)*H2O.real))

GMO_head_lay_d2o.vfsolv.setp(constraint=head_solvation)
GMO_head_lay_h2o.vfsolv.setp(constraint=head_solvation)

# GMO_tail
b_tail = -0.00010405 #Å
Solvation = Parameter(0.2, 'solvation', vary=True, bounds=(0, 0.99))
GMOav_vol_frac_tail = Parameter(0.05, 'GMOav_vol_frac_tail', vary=True, bounds=(0, 1))

G_tail_t = Parameter(10, name='GMO_tail_t', vary=True, bounds=(1, 25)) #could include physisorbed GMO not adsorbed at iron oxide.
G_tail_r = Parameter(1, 'GMO_tail_r', vary=True, bounds=(1, 10))

GMO_tail_par = Parameter(-0.16640, 'GMO_tail_par', constraint=1E6*b_tail/((1/0.001599235)-(b_head/(GMO_head.real*1E-6))))
GMO_tail = SLD(GMO_tail_par, name='GMO_tail') #send this GMO tail SLD through to the spline.

GMO_t_lay_H2O = GMO_tail_s_h2o(G_tail_t, G_head_r) #these are dummy layers, with dummy slds which will be sent through to VFP, where they'll be modified.
GMO_t_lay_D2O = GMO_tail_s_d2o(G_tail_t, G_head_r)

### SOLVENT PARAMS FOR ADV_LAYER SOLVENT

hd_d2o_GMO25_lay = hdod_SLD_1(0, G_tail_r)
dd_d2o_GMO25_lay = ddod_SLD_1(0, G_tail_r)
dd_h2o_GMO25_lay = ddod_SLD_2(0, G_tail_r)

"""
This is where we deviate from ordinary refnx models.
"""
### now define thicknesses, roughnesses, parameters for constraints & slds to send to the VFPs.

list_of_thickness = (0, SiO2_lay.thick, Fe_thick, FeOx_thick, G_head_t, G_tail_t) #first must be zero for Si thickness.
list_of_roughness = (SiO2_lay.rough, SiO2_Fe_R, Fe_FeOx_R, FeOx_R, G_head_r, G_tail_r)
list_of_pars_for_cons = (GMO_head.real, GMO_vol_frac, head_solvation, GMO_tail.real, Solvation, GMOav_vol_frac_tail)

#Create an extent parameter. This is the length of the whole volume fraction profiles & the resulting SLD curve so that the erfs approach 1.
#we have to include the roughness of the first interface and the roughness of the last interface (similar to how the SLD curves of a normal structure is calculated).
VFP_tot_thick = Parameter(name='VFP_tot_thick', constraint=5+(4 * SiO2_lay.rough)+SiO2_lay.thick+Fe_thick+FeOx_thick+G_head_t+G_tail_t+5+(4 * G_tail_r))

"""
Pass the SLDs to the VFPs. 
The SLDs are ordered as the following:
Si, SiO2, FeUp, FeDown, FeOxUp, FeOxDown, 
    GMO_head_dd_h2o, GMO_head_dd_d2o, GMO_head_hd_d2o, 
    GMO_tail_dd_h2o, GMO_tail_dd_d2o, GMO_tail_hd_d2o, 
    dd_d2o, dd_h2o, hd_d2o
"""
SLDs = np.array([Si.real, SiO2.real, FeUp_lay.sld_n.real+(((FeUp_lay.sld_n.real*1E-6)/Fe_ScattLen)*Fe_magmom*0.00002699*1E6),
        FeDown_lay.sld_n.real-(((FeDown_lay.sld_n.real*1E-6)/Fe_ScattLen)*Fe_magmom*0.00002699*1E6), (FeOxUp_lay.sld_n.real+FeOx_m),
        (FeOxDown_lay.sld_n.real-FeOx_m), (GMO_head_lay_h2o.sld.real*(1-head_solvation)+head_solvation*dd_h2o_GMO25_lay.sld.real),
        (GMO_head_lay_d2o.sld.real*(1-head_solvation)+head_solvation*dd_d2o_GMO25_lay.sld.real), 
        (GMO_head_lay_d2o.sld.real*(1-head_solvation)+head_solvation*hd_d2o_GMO25_lay.sld.real),
        (GMO_t_lay_H2O.sld.real), #dummy slds.
        (GMO_t_lay_D2O.sld.real), #dummy
        (GMO_t_lay_D2O.sld.real), #dummy
        dd_d2o_GMO25_lay.sld.real, dd_h2o_GMO25_lay.sld.real, hd_d2o_GMO25_lay.sld.real])

#create VFPs

dd_d2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='dd_d2o_up')
dd_d2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='dd_d2o_down')
dd_h2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='dd_h2o_up')
dd_h2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='dd_h2o_down')
hd_d2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='hd_d2o_up')
hd_d2o_down_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='hd_d2o_down')

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

#make sure to pass the parameters to the auxiliary params of the objectives.
auxiliary_parameters = (GMO_vol_frac, GMOav_vol_frac_tail, head_solvation, Solvation, SiO2_lay.thick,
                        SiO2_Fe_R, Fe_thick, Fe.real, Fe_magmom, Fe_ScattLen, Fe_FeOx_R, FeOx.real, FeOx_thick, FeOx_m, FeOx_R,
                        G_head_t, G_head_r, G_tail_t, G_tail_r, ddod_par_d2o, ddod_par, GMO_head_par)

#objectives
L1GMOdd_d2o_25_obj = Objective(L1_GMO25dd_d2o_M, L1_GMO_D2O_ddod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)
L1GMOdd_h2o_25_obj = Objective(L1_GMO25dd_h2o_M, L1_GMO_H2O_ddod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)
L1GMOhd_d2o_25_obj = Objective(L1_GMO25hd_d2o_M, L1_GMO_D2O_hdod_25, transform=Transform('logY'), auxiliary_params=auxiliary_parameters)

#glob_objective
glob_obj = GlobalObjective([L1GMOdd_d2o_25_obj, L1GMOdd_h2o_25_obj, L1GMOhd_d2o_25_obj])

#inequality constraints.
pars = (G_head_t, G_head_r, G_tail_t, G_tail_r, SiO2_lay.thick, SiO2_Fe_R, dd_d2o_up_vfp.create_constraints()[1], dd_d2o_up_vfp.create_constraints()[0])

class LogpExtra(object):
    def __init__(self, pars):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.pars = pars

    def __call__(self, model, data):
        if (float(self.pars[0]-2*self.pars[1]) >= 0 and float(self.pars[2]-2*self.pars[3]) >= 0 and float(self.pars[6]) >= 0 and float(self.pars[6]) <= 1 
        and float(self.pars[4]-2*self.pars[5]) >= 0 and float(self.pars[7]) >= 0 and float(self.pars[7]) <= 1):
            return 0
        return -1E15

lpe = LogpExtra(pars)

L1GMOdd_d2o_25_obj.logp_extra = lpe

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    fitter = CurveFitter(glob_obj, ntemps=10)
    fitter.initialise(pos='prior', random_state=None)
    from schwimmbad import MPIPool
    with MPIPool() as p:
        if not p.is_master():
            p.wait()
            sys.exit(0)
        fitter.sample(35000, nthin=1, pool=p.map, verbose=True) #Do the burn and thin after sampling!
        processed_chain = process_chain(glob_obj, fitter.chain, nburn=10000, nthin=250, flatchain=False)
        print(glob_obj)
    
    with open(os.path.join(os.getcwd(), 'M2_FIT_GO.pkl'), 'wb+') as f:
        pickle.dump(glob_obj, f)
