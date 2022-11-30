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
## DataList definition changed. May need to play with which entries exist in library and clean up how it is defined.
DataList = [{'Fname':'61112_14','angle':0.7,'source':'NR','data':'data','inst':'INTER','contrast':'hdod','name':'P1_hdod'},
			{'Fname':'61143_45','angle':0.7,'source':'NR','data':'data','inst':'INTER','contrast':'ddod','name':'P1_ddod'}
			]

## Load data and define probes:
probes = {}
OurProbes(DataList)

## Define all materials/base SLDs used:
ddod_SLD_1 = SLD(name='d_dod_1', rho=6.198712125879337)
hdod_SLD_1 = SLD(name='h_dod_1', rho=-0.462) #use for both hdod and hdod_gmo_25 datasets but not 60.
SiOx_SLD = SLD(name='SiOx', rho=3.47)
Fe_SLD = SLD(name='Fe', rho=7.875578976533852)
FeOx_SLD = SLD(name='FeOx', rho=5.7084410460950545)
adv_SLD_J = SLD(name='Adv', rho=0.22103496340963838)

## Define extra parameters for magnetic layers:
##Fe##
Fe_magmom = Parameter.default(2.112783984761999,name="Fe_magmom") #magnetic moment of iron in bohr magneton
Fe_scattlen = 0.0000945 #Å
MagneticSLDconstant = 0.00002699 #Å per bohr magneton.
Fe_mag = (((Fe_SLD.rho*1E-6)/Fe_scattlen)*MagneticSLDconstant*Fe_magmom)*1E6

##FeOx##
FeOx_mag=Parameter.default(0.09965432321365415,name="FeOx_m")

## And associated SLD values:
Fe_plus_SLD = SLD(name='Fe_plus', rho = Fe_SLD.rho + Fe_mag)
Fe_minus_SLD = SLD(name='Fe_minus', rho = Fe_SLD.rho - Fe_mag)
FeOx_plus_SLD = SLD(name='FeOx_plus', rho = FeOx_SLD.rho + FeOx_mag)
FeOx_minus_SLD = SLD(name='FeOx_minus', rho = FeOx_SLD.rho - FeOx_mag)

Adv_solv = Parameter.default(18.08544400938213,name='Adv_solv') ## In this definition is 0-100 range.
Adv_d = Mixture.byvolume(adv_SLD_J,ddod_SLD_1,Adv_solv,name='Adv_d')
Adv_h = Mixture.byvolume(adv_SLD_J,hdod_SLD_1,Adv_solv,name='Adv_h')

## Define all the slabs or splines used:
## Start with bulk layers - note the thickness is set to 0

Ddod_lay_1 = Slab(material=ddod_SLD_1, thickness=0, interface=2.5375794050066194)
Hdod_lay_1 = Slab(material=hdod_SLD_1, thickness=0, interface=2.5375794050066194)
Ddod_lay_1.interface = Hdod_lay_1.interface

## ADV layer:
adv_lay_d = Slab(material=Adv_d, thickness=13.7259081012633, interface=3.0999311418291904)
adv_lay_h = Slab(material=Adv_h, thickness=13.7259081012633, interface=3.0999311418291904)
adv_lay_d.thickness = adv_lay_h.thickness
adv_lay_d.interface = adv_lay_h.interface

## Then magnetic layers:
Fe_layer_plus = Slab(material=Fe_plus_SLD, thickness=189.8660539470065, interface=6.07660414524676)
Fe_layer_minus = Slab(material=Fe_minus_SLD, thickness=189.8660539470065, interface=6.07660414524676)
FeOx_layer_plus = Slab(material=FeOx_plus_SLD, thickness=32.76544572663599, interface=11.327616958407493)
FeOx_layer_minus = Slab(material=FeOx_minus_SLD, thickness=32.76544572663599, interface=11.327616958407493)

## Make sure these layers are the same except the magnetic part. This is more involved but makes clear which parameter
## to fit. It may slow the optimisation so should possibly test this.
FeOx_thick=Parameter.default(32.76544572663599,name="FeOx_thick")
FeOx_rough=Parameter.default(11.327616958407493,name="FeOx_rough")
Fe_thick=Parameter.default(189.8660539470065,name="Fe_thick")
Fe_rough=Parameter.default(6.07660414524676,name="Fe_rough")
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

SiOx_layer = Slab(material=SiOx_SLD, thickness=19.483562365493256, interface=3)
	
## Now define how the layers stack together as a series of samples:
#SOLVENT#
ddod_U = Ddod_lay_1 | adv_lay_d | FeOx_layer_plus | Fe_layer_plus | SiOx_layer | Si
ddod_D = Ddod_lay_1 | adv_lay_d | FeOx_layer_minus | Fe_layer_minus | SiOx_layer | Si
hdod_U = Hdod_lay_1 | adv_lay_h | FeOx_layer_plus | Fe_layer_plus | SiOx_layer | Si
hdod_D = Hdod_lay_1 | adv_lay_h | FeOx_layer_minus | Fe_layer_minus | SiOx_layer | Si

ratio_val=0.5 ## Ratio of mixing between the magnetic states.

## === Fit parameters ===
## "range" specifies a fitting range in terms of min/max value
## "pmp" specifies fitting range in terms of +/-  %
## "pm" specifies fitting range in terms of +/- value
## "dev" specifies a gaussian distribution prior for the parameter - M.thickness.dev(0.6)
## Anything not specified will be held.

Fe_thick.pmp(0.0000001)

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
bkg = [6.821211505650636e-06, 5.658635806509439e-06] ##hdod then ddod
IOs = [0.9406753876606535, 0.9503075723090361]
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

## This may need some optimisation but allows to specify model based on contrast, may be better removed:
models_dict={}
for i in range(0, 2):
	if DataList[i]['contrast'] =='ddod':
	    models_dict[DataList[i]['name']]=MixedExperiment(samples=[ddod_U,ddod_D], ratio=[ratio_val, 1-ratio_val], probe=probes[DataList[i]['name']], dz=zed, dA=1, step_interfaces=None)
	elif DataList[i]['contrast'] =='hdod':
	    models_dict[DataList[i]['name']]=MixedExperiment(samples=[hdod_U,hdod_D], ratio=[ratio_val, 1-ratio_val], probe=probes[DataList[i]['name']], dz=zed, dA=1, step_interfaces=None)

## simultaneous fitting: if you define two models
models = list(models_dict.values())

if len(models_dict.keys()) == 1:
    problem = FitProblem(models[0])
else:
    problem = MultiFitProblem(models=models)