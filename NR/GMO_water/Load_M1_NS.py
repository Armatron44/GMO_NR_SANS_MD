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

from vfp_M1 import VFP
import MixedMagSlabs2 #load in custom model for magnetic layers in non-polarised instrument.
import pickle
import copy

from dynesty import plotting as dyplot
from dynesty import utils as dyfunc
import matplotlib as mpl
from cycler import cycler
from functools import reduce

#we need to redefine the LogpExtra class used in the model script to load the GO file.
class LogpExtra(object):
    def __init__(self, pars):
        # we'll store the parameters and objective in this object
        # this will be necessary for pickling in the future
        self.pars = pars

    def __call__(self, model, data):
        if float(self.pars[0]-2*self.pars[1]) >= 0:
            return 0
        return -1E15
        
with open(os.path.join(os.getcwd(), 'M1_NS.pkl'),"rb") as fp:
	res = pickle.load(fp)

with open(os.path.join(os.getcwd(), 'M1_GO.pkl'),"rb") as f:
	GO = pickle.load(f)
    
print(f'The log evidence is {res.logz[-1]:.1E} +/- {res.logzerr[-1]:.1E}')
    
samples = res.samples  # samples explored by nested sampling.
weights = np.exp(res.logwt - res.logz[-1])  # normalized weights. These go from low to high.

labelz = GO.varying_parameters().names() #names of the varying parameters.	

# Compute 2.5%-97.5% quantiles.
quantiles = [dyfunc.quantile(samps, [0.025, 0.5, 0.975], weights=weights)
             for samps in samples.T]

#lets print out the posterior found by NS.
for i, j in enumerate(quantiles):
    print(f"""{labelz[i]}
    Median: {j[1]:.2E}
    Lower: {j[0]:.2E}
    Upper: {j[2]:.2E}
    Minus: {j[1]-j[0]:.2E}
    Plus: {j[2]-j[1]:.2E}""")

#function to return the posterior of the samples.
def get_posterior(quantiles, samples, sample_len=300):
    """
    This function finds the samples from the nested sampling that lie within the 95 % CI for all parameters.
    First, the indexes of the parameter values between the individual 95 % CI are found.
    Then the indexes where all parameter values fall within the overall posterior.
    The full posterior sample is then returned. Finally, we take a sub-sample of the posterior.
    """
    indiv_index = []
    for i, j in enumerate(samples.T): #samples.T is (i x n) where i is the number of parameters & n is number of samples.
        indexes = np.where((j >= quantiles[i][0]) & (j <= quantiles[i][2]))[0] # return just the array.
        indiv_index.append(indexes) # a list of indexes of different sizes.
    
    #now find those samples that all within the posterior.
    intersect_tuple = tuple(i for i in indiv_index) # make tuple out of list of indexes.
    posterior_indexes = reduce(np.intersect1d, intersect_tuple) # now we have array of indicies of samples that are all within posterior.
    full_posterior = samples[posterior_indexes] # full posterior.
    
    if sample_len > len(posterior_indexes):
        sample_len = len(posterior_indexes)
        
    random_number = np.random.choice(posterior_indexes, size=sample_len, replace=False)
    sampled_posterior = samples[random_number]
    return sampled_posterior

#return 300 of the posterior samples.
posterior_samples = get_posterior(quantiles, samples, 300)

#set the varying_parameters stored in GO to the median values found in quantiles.
for i, j in enumerate(labelz):
	if j == GO.varying_parameters()[i].name:
		GO.varying_parameters()[i].value = quantiles[i][1]

#lets plot the reflectivity, sld profiles and volume fraction profiles.
#first reflectivity:
dummy_obj = copy.deepcopy(GO)
chi2 = GO.chisqr()
chi2red = chi2/(GO.npoints)

fig, ax = plt.subplots()
for i in posterior_samples: #plot the posterior
    dummy_obj.varying_parameters().pvals = i #change the values of the varying parameters to the posterior samples.
    ax.plot(GO.objectives[0].data.x, dummy_obj.objectives[0].generative(), color='tab:blue', alpha=0.01) #then plot the dummy reflectivity
    ax.plot(GO.objectives[1].data.x, dummy_obj.objectives[1].generative()/10, color='tab:orange', alpha=0.01)
    ax.plot(GO.objectives[2].data.x, dummy_obj.objectives[2].generative()/100, color='tab:green', alpha=0.01)
#plot the median reflectivity
ax.plot(GO.objectives[0].data.x, GO.objectives[0].generative(), color='k', zorder=2, linewidth=1.5)
ax.plot(GO.objectives[1].data.x, GO.objectives[1].generative()/10, color='k', zorder=2, linewidth=1.5)
ax.plot(GO.objectives[2].data.x, GO.objectives[2].generative()/100, color='k', zorder=2, linewidth=1.5)
#plot the data stored in the original objective.
ax.errorbar(GO.objectives[0].data.x, GO.objectives[0].data.y, yerr=GO.objectives[0].data.y_err, linestyle='None', marker='.', color='#2B3787', zorder=1, label='dd_d2o')
ax.errorbar(GO.objectives[1].data.x, GO.objectives[1].data.y/10, yerr=GO.objectives[1].data.y_err/10, linestyle='None', marker='.', color='#A36E33', zorder=1, label='dd_h2o')
ax.errorbar(GO.objectives[2].data.x, GO.objectives[2].data.y/100, yerr=GO.objectives[2].data.y_err/100, linestyle='None', marker='.', color='#37652D', zorder=1, label='hd_d2o')
#formatting.
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'$R\left(Q\right)$')
ax.set_xlabel(r'$Q$ / $\mathrm{\AA}^{-1}$')
ax.text(1E-2, 1E-7, f'$\chi^{2}$ = {chi2:.2E},' + r' $\chi^{2}_{\mathrm{p}}$ = ' + f'{chi2red:.2E}')
ax.legend(frameon=False)
plt.show()

#now nuclear + magnetic sld & volume fraction profiles.
fig2, ax2 = plt.subplots(2, figsize=(9, 9), sharex=True)
for i in posterior_samples:
    dummy_obj.varying_parameters().pvals = i
    ax2[0].plot(
                dummy_obj.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
                (dummy_obj.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[1]+dummy_obj.objectives[0].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
                color='tab:blue', alpha=0.01
                )
    ax2[0].plot(
                dummy_obj.objectives[1].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
                (dummy_obj.objectives[1].model.structures[0].sld_profile(max_delta_z=0.01)[1]+dummy_obj.objectives[1].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
                color='tab:orange', alpha=0.01
                )
    ax2[0].plot(
                dummy_obj.objectives[2].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
                (dummy_obj.objectives[2].model.structures[0].sld_profile(max_delta_z=0.01)[1]+dummy_obj.objectives[2].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
                color='tab:green', alpha=0.01
                )
    ax2[0].plot(
                dummy_obj.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
                (dummy_obj.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[1]-dummy_obj.objectives[0].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
                color='tab:red', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[0].T,
                color='tab:blue', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[1].T,
                color='tab:orange', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[2].T,
                color='tab:green', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[3].T,
                color='tab:red', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[4].T,
                color='tab:purple', alpha=0.01
                )
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[5].T,
                color='tab:brown', alpha=0.01
                )
ax2[0].plot(
            GO.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
            (GO.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[1]+GO.objectives[0].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
            color='k'
            )
ax2[0].plot(
            GO.objectives[1].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
            (GO.objectives[1].model.structures[0].sld_profile(max_delta_z=0.01)[1]+GO.objectives[1].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
            color='k'
            )
ax2[0].plot(
            GO.objectives[2].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
            (GO.objectives[2].model.structures[0].sld_profile(max_delta_z=0.01)[1]+GO.objectives[2].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
            color='k'
            )
ax2[0].plot(
            GO.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[0]-(5 + 4 * 3), 
            (GO.objectives[0].model.structures[0].sld_profile(max_delta_z=0.01)[1]-GO.objectives[0].model.structures[1].sld_profile(max_delta_z=0.01)[1])/2, 
            color='k'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[0].T,
            color='k', linestyle='--'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[1].T,
            color='k', linestyle='--'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[2].T,
            color='k', linestyle='--'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[3].T,
            color='k', linestyle='--'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[4].T,
            color='k', linestyle='--'
            )
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[5].T,
            color='k', linestyle='--'
            )
#formatting

import matplotlib.lines as mlines
r_dd_d2o = mlines.Line2D([], [], color='tab:blue',  label=r'dd_d2o_nuclear')
r_dd_h2o = mlines.Line2D([], [], color='tab:orange',  label=r'dd_h2o_nuclear')
r_hd_d2o = mlines.Line2D([], [], color='tab:green',  label=r'hd_d2o_nuclear')
r_mag = mlines.Line2D([], [], color='tab:red',  label=r'magnetic')

vf_Si = mlines.Line2D([], [], color='tab:blue',  label=r'Si')
vf_SiO2 = mlines.Line2D([], [], color='tab:orange',  label=r'SiO$_{2}$')
vf_Fe = mlines.Line2D([], [], color='tab:green',  label=r'Fe')
vf_FeOx = mlines.Line2D([], [], color='tab:red',  label=r'FeO$_{x}$')
vf_GMO_water_solvent = mlines.Line2D([], [], color='tab:purple',  label=r'GMO+water+solvent')
vf_Solv = mlines.Line2D([], [], color='tab:brown',  label=r'Solv')

ax2[0].legend(handles=[r_dd_d2o, r_dd_h2o, r_hd_d2o, r_mag], frameon=False, loc='best')
ax2[0].set_ylabel(r'$\beta$ / $\mathrm{\AA}^{-2} \times 10^{-6}$')

ax2[1].set_xlabel(r'Distance over Interface / $\mathrm{\AA}$')
ax2[1].set_ylabel(r'Volume Fraction')
ax2[1].legend(handles=[vf_Si, vf_SiO2, vf_Fe, vf_FeOx, vf_GMO_water_solvent, vf_Solv], frameon=False, loc='best')
plt.show()

#corner plot.
cfig, caxes = dyplot.cornerplot(res,
                                max_n_ticks=2,
                                use_math_text=True, 
                                labels=labelz,
                                show_titles=True)
plt.show()

raise Exception()