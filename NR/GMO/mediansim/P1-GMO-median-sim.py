from refl1d.names import *
from copy import copy
import numpy as np

from types import MethodType
from numpy import inf

### Tims POLREF Loader i.e. Generic TOF loader for const dQ/Q resolution - for D17 style data with specific dQ error bars this needs to be modified slightly
from refl1d.probe import make_probe # AJC 4/10/2017 - needed to construct the instrument (POLREF in our case)

def QT2L(Q,T): # Q = 4 pi sin(T) / L   ==>  L = 4 pi sin(T) / Q
    return 4 * numpy.pi * numpy.sin(numpy.radians(T)) / Q
def Ourload(Fname=None, T=0.25, dQoQ=0.047096401, Q_sim_range=[0.005,0.2], instrument='INTER', **kw): #dQoQ in FWHM. FWHM = SD * 2.35482
    if instrument in ['OFFSPEC', 'POLREF', 'INTER']:
        if Fname == None:
            Q = numpy.logspace(np.log10(Q_sim_range[0]),np.log10(Q_sim_range[1]),170)
        else:
            data = numpy.loadtxt(Fname, skiprows=0).T
            Q, R, dR = data
        dQ = Q * dQoQ
        L = QT2L(Q,T)  # print L
        dT, dL = 0, dQ*L/Q # Set dT = 0 and dL = dQ/Q * L in order to preserve dQ

        if Fname == None:
            p = make_probe(T=T, dT=dT, L=L, dL=dL, data=None, radiation = 'neutron', **kw)
        else:
            p = make_probe(T=T, dT=dT, L=L, dL=dL, data=(R,dR), radiation = 'neutron', **kw)
            p.title = Fname
    return p

def OurProbes(DataList):
	for i in range(len(DataList)):
		if DataList[i]['data'] =='data' and DataList[i]['source'] =='PNR':
			if DataList[i]['inst'] == 'POLREF':
				pp = Ourload(r'./'+DataList[i]['Fname']+'_u.dat', T=DataList[i]['angle'], dQoQ=0.047096401,intensity=1.0,background=1e-7,instrument=DataList[i]['inst']) # Spin up data file
				mm = Ourload(r'./'+DataList[i]['Fname']+'_d.dat', T=DataList[i]['angle'], dQoQ=0.047096401,intensity=1.0,background=1e-7,instrument=DataList[i]['inst']) # Spin up data file
			elif DataList[i]['inst'] == 'OFFSPEC':
				pp = Ourload(r'./'+DataList[i]['Fname']+'_u.dat', T=DataList[i]['angle'], dQoQ=0.047096401,intensity=1.0,background=1e-7,instrument=DataList[i]['inst']) # Spin up data file
				mm = Ourload(r'./'+DataList[i]['Fname']+'_d.dat', T=DataList[i]['angle'], dQoQ=0.047096401,intensity=1.0,background=1e-7,instrument=DataList[i]['inst']) # Spin up data file
			elif DataList[i]['inst'] == 'D17':
				pp = Ourload(r'./'+'pol00_'+DataList[i]['Fname']+'.lam', T=DataList[i]['angle'],intensity=1.0,background=1e-7,instrument=DataList[i]['inst']) # Spin up data file
				mm = Ourload(r'./'+'pol11_'+DataList[i]['Fname']+'.lam', T=DataList[i]['angle'],intensity=1.0,background=1e-7,instrument=DataList[i]['inst'])  # Spin down data file
			data = [mm,None,None,pp]
			probes[DataList[i]['name']] = PolarizedNeutronProbe(data, Aguide=270, H=0.7,name=DataList[i]['name']); probes[DataList[i]['name']].xs[1],probes[DataList[i]['name']].xs[2] = None, None
			probes[DataList[i]['name']].view = "log"
		elif DataList[i]['data'] =='data' and DataList[i]['source'] =='NR':
			data = Ourload(DataList[i]['Fname']+'.txt',T=DataList[i]['angle'], dQoQ=0.047096401,intensity=1.0,background=1e-7,name=DataList[i]['name'],instrument=DataList[i]['inst']) # Looks for a .txt file.
			probes[DataList[i]['name']] = data
	return probes

DataList=[]
DataList = [{'Fname':'61167_69','angle':0.7,'source':'NR','data':'data','inst':'INTER','contrast':'ddod_25','name':'P1_GMO_ddod_25'},
			{'Fname':'61203_05','angle':0.7,'source':'NR','data':'data','inst':'INTER','contrast':'CMdod_25','name':'P1_GMO_CMdod_25'},
			{'Fname':'61215_17','angle':0.7,'source':'NR','data':'data','inst':'INTER','contrast':'hdod_25','name':'P1_GMO_hdod_25'}
            ]

"""
Premable for model:

This model was run previously to find the approximate point of the stationary distribution.
Therefore, some of the initialised parameter values may seem arbitary, 
but they have been chosen to be at the approximate point of equilibrium.
"""

## Load data and define probes:
probes = {}
OurProbes(DataList)

## Define all materials/base SLDs used:
hdod_SLD_1 = SLD(name='h_dod_1', rho=-0.462)
ddod_SLD_2 = SLD(name='d_dod_2', rho=6.319170727392011)
SiOx_SLD = SLD(name='SiOx', rho=3.47)
Fe_SLD = SLD(name='Fe', rho=7.914020640275218)
FeOx_SLD = SLD(name='FeOx', rho=5.637174877851259)
GMO_SLD_25 = SLD(name='GMO_25', rho=0.21)

## Define extra parameters for magnetic layers:
##Fe##
Fe_magmom = Parameter.default(2.09144280057776, name="Fe_magmom") #magnetic moment of iron in bohr magneton
Fe_scattlen = 0.0000945 #Å
MagneticSLDconstant = 0.00002699 #Å per bohr magneton.
Fe_mag = (((Fe_SLD.rho*1E-6)/Fe_scattlen)*MagneticSLDconstant*Fe_magmom)*1E6

##FeOx##
FeOx_mag=Parameter.default(0.11482357649760785, name="FeOx_m")

## And associated SLD values:
Fe_plus_SLD = SLD(name='Fe_plus', rho = Fe_SLD.rho + Fe_mag)
Fe_minus_SLD = SLD(name='Fe_minus', rho = Fe_SLD.rho - Fe_mag)
FeOx_plus_SLD = SLD(name='FeOx_plus', rho = FeOx_SLD.rho + FeOx_mag)
FeOx_minus_SLD = SLD(name='FeOx_minus', rho = FeOx_SLD.rho - FeOx_mag)

## Contrast matched SLD values:
CMdod_vf = Parameter.default(0.352, name="CMdod_vf") #0.352 is the volume fraction of d-dod in the mixture of d-dod and h-dod.
CMdod_25_mix = ddod_SLD_2.rho*CMdod_vf + (1-CMdod_vf)*hdod_SLD_1.rho #mix the slds with the volume fraction to get SLD of CMdod.
CMdod_25 = SLD(name='CMdod_25', rho = CMdod_25_mix)

## Define extra parameters for solvation:
GMO_solv_25 = Parameter.default(8.575932307850207,name='GMO_solv_25') ##  solvation in %.
GMO_d_25 = Mixture.byvolume(GMO_SLD_25, ddod_SLD_2, GMO_solv_25, name='GMO_d_25')
GMO_CM_25 = Mixture.byvolume(GMO_SLD_25, CMdod_25, GMO_solv_25, name='GMO_CM_25')
GMO_h_25 = Mixture.byvolume(GMO_SLD_25, hdod_SLD_1, GMO_solv_25, name='GMO_h_25')

Ddod_lay_GMO_25 = Slab(material=ddod_SLD_2, thickness=0, interface=5.977790092814816)
Hdod_lay_GMO_25 = Slab(material=hdod_SLD_1, thickness=0, interface=5.977790092814816)
CMdod_lay_GMO_25 = Slab(material=CMdod_25, thickness=0, interface=5.977790092814816)
Ddod_lay_GMO_25.interface = Hdod_lay_GMO_25.interface
CMdod_lay_GMO_25.interface = Hdod_lay_GMO_25.interface

## Then magnetic layers:
Fe_layer_plus = Slab(material=Fe_plus_SLD, thickness=190, interface=4)
Fe_layer_minus = Slab(material=Fe_minus_SLD, thickness=190, interface=4)
FeOx_layer_plus = Slab(material=FeOx_plus_SLD, thickness=30, interface=5)
FeOx_layer_minus = Slab(material=FeOx_minus_SLD, thickness=30, interface=5)

## Make sure these layers are the same except the magnetic part. This is more involved but makes clear which parameter
## to fit.
FeOx_thick=Parameter.default(34.50993281701116, name="FeOx_thick")
FeOx_rough=Parameter.default(11.921389455674188, name="FeOx_rough")
Fe_thick=Parameter.default(189.72727587199935, name="Fe_thick")
Fe_rough=Parameter.default(4.084097015355201, name="Fe_rough")
####
Fe_layer_plus.thickness = Fe_thick
Fe_layer_plus.interface = Fe_rough
Fe_layer_minus.thickness = Fe_thick
Fe_layer_minus.interface = Fe_rough
FeOx_layer_plus.thickness = FeOx_thick
FeOx_layer_plus.interface = FeOx_rough
FeOx_layer_minus.thickness = FeOx_thick
FeOx_layer_minus.interface = FeOx_rough
####
####GMO layers####
#GMO_25#
GMO_ddod_lay_25 = Slab(material=GMO_d_25, thickness=19.385059253096728, interface=4.796275012579236)
GMO_CMdod_lay_25 = Slab(material=GMO_CM_25, thickness=19.385059253096728, interface=4.796275012579236)
GMO_hdod_lay_25 = Slab(material=GMO_h_25, thickness=19.385059253096728, interface=4.796275012579236)

GMO_ddod_lay_25.thickness = GMO_hdod_lay_25.thickness
GMO_CMdod_lay_25.thickness = GMO_hdod_lay_25.thickness
GMO_ddod_lay_25.interface = GMO_hdod_lay_25.interface
GMO_CMdod_lay_25.interface = GMO_hdod_lay_25.interface

SiOx_layer = Slab(material=SiOx_SLD, thickness=18.31889753030951, interface=3)
	
## Now define how the layers stack together as a series of samples:
GMO25ddod_U = Ddod_lay_GMO_25 | GMO_ddod_lay_25 | FeOx_layer_plus | Fe_layer_plus | SiOx_layer | Si
GMO25ddod_D = Ddod_lay_GMO_25 | GMO_ddod_lay_25 | FeOx_layer_minus | Fe_layer_minus | SiOx_layer | Si
GMO25CMdod_U = CMdod_lay_GMO_25 | GMO_CMdod_lay_25 | FeOx_layer_plus | Fe_layer_plus | SiOx_layer | Si
GMO25CMdod_D = CMdod_lay_GMO_25 | GMO_CMdod_lay_25 | FeOx_layer_minus | Fe_layer_minus | SiOx_layer | Si
GMO25hdod_U = Hdod_lay_GMO_25 | GMO_hdod_lay_25 | FeOx_layer_plus | Fe_layer_plus | SiOx_layer | Si
GMO25hdod_D = Hdod_lay_GMO_25 | GMO_hdod_lay_25 | FeOx_layer_minus | Fe_layer_minus | SiOx_layer | Si

ratio_val=0.5 ## Ratio of mixing between the magnetic states.

## === Fit parameters ===
## "range" specifies a fitting range in terms of min/max value
## "pmp" specifies fitting range in terms of +/-  %
## "pm" specifies fitting range in terms of +/- value
## "dev" specifies a gaussian distribution prior for the parameter - M.thickness.dev(0.6)
## Anything not specified will be held.

## SiO2 ##

## Fe ##
Fe_SLD.rho.pmp(0.00001)

## FeOx ##

## GMO ## -- hold GMO sld at 0.21
## 25 GMO ##

## Solvent ##

## === Problem and Instrument definition ===
## zed is the step size in Angstroms to be used for rendering the profile
## increase zed to speed up the calculation
zed = 0.5   # <-- slicing thickness

## step = True corresponds to a calculation of the reflectivity from an actual profile
## with microslabbed interfaces.  When step = False, the Nevot-Croce
## approximation is used to account for roughness.  This approximation speeds up
## the calculation tremendously, and is reasonably accuarate as long as the
## roughness is much less than the layer thickness
step = False

## Example other parameters:
I0 = 1.0
bkg = [4.850123436459141e-06, 9.209947937386724e-06, 6.061751473879793e-06]
IOs = [0.9483492378072225, 0.9398289553404213, 0.9470334545593921]
bkgnd = 1e-6
SampBroad = 0.00
tth_offset = 0.0
for i in range(len(DataList)):
    probes[DataList[i]['name']].background.value = bkg[i]
    probes[DataList[i]['name']].background.name = DataList[i]['name']+' Bkg'
    #probes[DataList[i]['name']].background.range(1e-7, 2e-5)
    probes[DataList[i]['name']].intensity.value = IOs[i]
    probes[DataList[i]['name']].intensity.name = DataList[i]['name']+' I0'
    #probes[DataList[i]['name']].intensity.range(0.8, 1.2)
    probes[DataList[i]['name']].sample_broadening.value = SampBroad
    probes[DataList[i]['name']].theta_offset.value = tth_offset

models_dict={}
for i in range(0, 3):
    if DataList[i]['contrast'] =='ddod_25':
        models_dict[DataList[i]['name']]=MixedExperiment(samples=[GMO25ddod_U,GMO25ddod_D], ratio=[ratio_val, 1-ratio_val], probe=probes[DataList[i]['name']], dz=zed, dA=1, step_interfaces=None)
    elif DataList[i]['contrast'] =='CMdod_25':
        models_dict[DataList[i]['name']]=MixedExperiment(samples=[GMO25CMdod_U,GMO25CMdod_D], ratio=[ratio_val, 1-ratio_val], probe=probes[DataList[i]['name']], dz=zed, dA=1, step_interfaces=None)
    elif DataList[i]['contrast'] =='hdod_25':
        models_dict[DataList[i]['name']]=MixedExperiment(samples=[GMO25hdod_U,GMO25hdod_D], ratio=[ratio_val, 1-ratio_val], probe=probes[DataList[i]['name']], dz=zed, dA=1, step_interfaces=None)

## simultaneous fitting: if you define two models
models = list(models_dict.values())

if len(models_dict.keys()) == 1:
    problem = FitProblem(models[0])
else:
    problem = MultiFitProblem(models=models)