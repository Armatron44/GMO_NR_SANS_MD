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
        if (float(self.pars[0]-2*self.pars[1]) >= 0 and float(self.pars[2]-2*self.pars[3]) >= 0 and float(self.pars[6]) >= 0 and float(self.pars[6]) <= 1 
        and float(self.pars[4]-2*self.pars[5]) >= 0 and float(self.pars[7]) >= 0 and float(self.pars[7]) <= 1):
            return 0
        return -1E15
        
with open(os.path.join(os.getcwd(), 'M2_NS.pkl'),"rb") as fp:
	res = pickle.load(fp)

with open(os.path.join(os.getcwd(), 'M2_GO.pkl'),"rb") as f:
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
    for i, j in enumerate(samples.T): #samples.T is (i x n) where is the number of parameters & n is number of samples.
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

# #lets plot the reflectivity, sld profiles and volume fraction profiles.
# #first reflectivity:
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
    ax2[1].plot(
                dummy_obj.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
                dummy_obj.objectives[0].model.structures[0].components[1].vfs_for_display()[6].T,
                color='tab:grey', alpha=0.01
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
ax2[1].plot(
            GO.objectives[0].model.structures[0].components[1].get_x_and_y_scatter()[0],
            GO.objectives[0].model.structures[0].components[1].vfs_for_display()[6].T,
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
vf_inner = mlines.Line2D([], [], color='tab:purple',  label=r'inner')
vf_outer = mlines.Line2D([], [], color='tab:brown',  label=r'outer')
vf_solv = mlines.Line2D([], [], color='tab:grey',  label=r'solv')

ax2[0].legend(handles=[r_dd_d2o, r_dd_h2o, r_hd_d2o, r_mag], frameon=False, loc='best')
ax2[0].set_ylabel(r'$\beta$ / $\mathrm{\AA}^{-2} \times 10^{-6}$')

ax2[1].set_xlabel(r'Distance over Interface / $\mathrm{\AA}$')
ax2[1].set_ylabel(r'Volume Fraction')
ax2[1].legend(handles=[vf_Si, vf_SiO2, vf_Fe, vf_FeOx, vf_inner, vf_outer, vf_solv], frameon=False, loc='best')
plt.show()

corner plot.
cfig, caxes = dyplot.cornerplot(res,
                                max_n_ticks=2,
                                use_math_text=True, 
                                labels=labelz,
                                show_titles=True)
plt.show()

#calculate volume fraction profiles of GMO, water & solvent + surface excesses.
b_head = 0.00023623
b_tail = -0.00010405
GMO_SE = []
water_SE = []

GMO_t_av_SE = []
GMO_head_SE = []
GMO_tail_SE = []

fig3, ax3 = plt.subplots()
for i in posterior_samples:
    dummy_obj.varying_parameters().pvals = i
    
    #label parameters for easy reading:
    GMO_vol_frac_head = dummy_obj.varying_parameters()[0] #defined without including solvation.
    GMOav_vol_frac_tail = dummy_obj.varying_parameters()[1] #defined without including solvation.
    head_solvation = dummy_obj.varying_parameters()[2]
    solvation = dummy_obj.varying_parameters()[3]
    SiO2_thick = dummy_obj.varying_parameters()[4]
    SiO2_Fe_R = dummy_obj.varying_parameters()[5]
    Fe_thick = dummy_obj.varying_parameters()[6]
    Fe_SLD = dummy_obj.varying_parameters()[7]
    Fe_magmom = dummy_obj.varying_parameters()[8]
    FeOx_Fe_R = dummy_obj.varying_parameters()[9]
    FeOx_SLD = dummy_obj.varying_parameters()[10]
    FeOx_thick = dummy_obj.varying_parameters()[11]
    FeOx_m = dummy_obj.varying_parameters()[12]
    FeOx_R = dummy_obj.varying_parameters()[13]
    GMO_head_thick = dummy_obj.varying_parameters()[14]
    GMO_head_rough = dummy_obj.varying_parameters()[15]
    GMO_tail_thick = dummy_obj.varying_parameters()[16]
    GMO_tail_rough = dummy_obj.varying_parameters()[17]
    dd_d2o_sld = dummy_obj.varying_parameters()[18]
    dd_h2o_sld = dummy_obj.varying_parameters()[19]
    GMO_head = dummy_obj.varying_parameters()[20]
    GMO_tail = 1E6*b_tail/((1/0.001599235)-(b_head/(GMO_head*1E-6)))
    
    GMO_head_lay_h2o = GMO_head*GMO_vol_frac_head+(1-GMO_vol_frac_head)*-0.558
    GMO_head_lay_d2o = GMO_head*GMO_vol_frac_head+(1-GMO_vol_frac_head)*6.37097
    
    SLDs = np.array([
                    2.07, 
                    3.47, 
                    Fe_SLD+(((Fe_SLD*1E-6)/0.0000945)*Fe_magmom*0.00002699*1E6),
                    Fe_SLD-(((Fe_SLD*1E-6)/0.0000945)*Fe_magmom*0.00002699*1E6), 
                    (FeOx_SLD+FeOx_m),
                    (FeOx_SLD-FeOx_m), 
                    (GMO_head_lay_h2o*(1-head_solvation)+head_solvation*dd_h2o_sld),
                    (GMO_head_lay_d2o*(1-head_solvation)+head_solvation*dd_d2o_sld), 
                    (GMO_head_lay_d2o*(1-head_solvation)+head_solvation*-0.462),
                    0.211387, #dummy
                    0.211387, #dummy
                    0.211387, #dummy
                    dd_d2o_sld, 
                    dd_h2o_sld, 
                    -0.462
                    ])
    
    list_of_thickness = (0, SiO2_thick, Fe_thick, FeOx_thick, GMO_head_thick, GMO_tail_thick)
    list_of_roughness = (3, SiO2_Fe_R, FeOx_Fe_R, FeOx_R, GMO_head_rough, GMO_tail_rough)
    list_of_pars_for_cons = (GMO_head, GMO_vol_frac_head, head_solvation, GMO_tail, solvation, GMOav_vol_frac_tail)

    VFP_tot_thick = Parameter(name='VFP_tot_thick', constraint=5+(4 * 3)+SiO2_thick+Fe_thick+FeOx_thick+GMO_head_thick+GMO_tail_thick+5+(4 * GMO_tail_rough))

    ### have a look at SLDs from splines ###
    dd_d2o_up_vfp = VFP(extent=VFP_tot_thick, SLDs=SLDs, thicknesses=list_of_thickness, roughnesses=list_of_roughness, pcons=list_of_pars_for_cons, contrast='dd_d2o_up')

    volume_fracs = dd_d2o_up_vfp.vfs_for_display()
    
    GMO_tail_vf = dd_d2o_up_vfp.create_constraints()[0]
    
    GMO_vf_components = (
                     dd_d2o_up_vfp.vfs_for_display()[4]*float(GMO_vol_frac_head)*(1-float(head_solvation)) 
                     + dd_d2o_up_vfp.vfs_for_display()[5]*(GMO_tail_vf+(GMOav_vol_frac_tail*(1-float(solvation))))
                     )
    GMO_vf = np.reshape(GMO_vf_components, (1, -1))

    water_vf_components = (
                       dd_d2o_up_vfp.vfs_for_display()[4]*(1-float(GMO_vol_frac_head))*(1-float(head_solvation))
                       + dd_d2o_up_vfp.vfs_for_display()[5]*(1-((GMO_tail_vf/(1-float(solvation)))+GMOav_vol_frac_tail))*(1-float(solvation))
                       )
    water_vf = np.reshape(water_vf_components, (1, -1))

    solvent_vf_components = (
                         dd_d2o_up_vfp.vfs_for_display()[4]*float(head_solvation)
                         + dd_d2o_up_vfp.vfs_for_display()[5]*float(solvation)
                         + dd_d2o_up_vfp.vfs_for_display()[6]
                         )
    solvent_vf = np.reshape(solvent_vf_components, (1, -1))

    sum_of_vfs = np.sum(np.vstack((volume_fracs[0:4], GMO_vf, water_vf, solvent_vf)).T, 1)
    
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], dd_d2o_up_vfp.vfs_for_display()[0].T, color='tab:blue', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], dd_d2o_up_vfp.vfs_for_display()[1].T, color='tab:orange', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], dd_d2o_up_vfp.vfs_for_display()[2].T, color='tab:green', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], dd_d2o_up_vfp.vfs_for_display()[3].T, color='tab:red', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], GMO_vf.T, color='tab:purple', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], water_vf.T, color='tab:brown', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], solvent_vf.T, color='tab:pink', alpha=0.01)
    ax3.plot(dd_d2o_up_vfp.get_x_and_y_scatter()[0], sum_of_vfs, color='tab:gray', alpha=0.01)
    ax3.set_xlabel(r'Distance over Interface / $\mathrm{\AA}$')
    ax3.set_ylabel(r'Volume Fraction')
    
    #water_SE
    water_vf_components = np.array([float(i) for i in water_vf_components])
    integrate_over = dd_d2o_up_vfp.get_x_and_y_scatter()[0]
    area_under_water = np.trapz(water_vf_components, x=integrate_over)
    water_SE.append(float(((6.37097*1E-6)/(6.0221409*1E23*1.914E-04))*area_under_water))
    
    #GMO_headandtail_SE
    GMO_head_SE.append(float(1E-6*dd_d2o_up_vfp.calc_GMO_head_tail_SE()[0]/(6.0221409*1E23)))
    GMO_tail_SE.append(float(1E-6*dd_d2o_up_vfp.calc_GMO_head_tail_SE()[1]/(6.0221409*1E23)))
    
    #GMO_tail_av_SE
    GMO_av_vf_components = np.array([float(i) for i in dd_d2o_up_vfp.vfs_for_display()[5]])
    area_under_GMO_av_vf = np.trapz(GMO_av_vf_components, x=integrate_over)
    GMO_t_av_SE_indiv = float(((0.211387*1E-6*GMOav_vol_frac_tail*(1-float(solvation)))/(6.0221409*1E23*0.00013218))*area_under_GMO_av_vf)
    GMO_t_av_SE.append(GMO_t_av_SE_indiv)
    
    #complete GMO SE
    GMO_total_SE = GMO_t_av_SE_indiv+float(1E-6*dd_d2o_up_vfp.calc_GMO_head_tail_SE()[1]/(6.0221409*1E23))
    GMO_SE.append(GMO_total_SE) #only add one of the head or tail SEs.

#plot median volume fractions.
GMO_vol_frac_head_median = GO.varying_parameters()[0] #defined without including solvation.
GMOav_vol_frac_tail_median = GO.varying_parameters()[1] #defined without including solvation.
head_solvation_median = GO.varying_parameters()[2]
solvation_median = GO.varying_parameters()[3]
SiO2_thick_median = GO.varying_parameters()[4]
SiO2_Fe_R_median = GO.varying_parameters()[5]
Fe_thick_median = GO.varying_parameters()[6]
Fe_SLD_median = GO.varying_parameters()[7]
Fe_magmom_median = GO.varying_parameters()[8]
FeOx_Fe_R_median = GO.varying_parameters()[9]
FeOx_SLD_median = GO.varying_parameters()[10]
FeOx_thick_median = GO.varying_parameters()[11]
FeOx_m_median = GO.varying_parameters()[12]
FeOx_R_median = GO.varying_parameters()[13]
GMO_head_thick_median = GO.varying_parameters()[14]
GMO_head_rough_median = GO.varying_parameters()[15]
GMO_tail_thick_median = GO.varying_parameters()[16]
GMO_tail_rough_median = GO.varying_parameters()[17]
dd_d2o_sld_median = GO.varying_parameters()[18]
dd_h2o_sld_median = GO.varying_parameters()[19]
GMO_head_median = GO.varying_parameters()[20]
GMO_tail_median = 1E6*b_tail/((1/0.001599235)-(b_head/(GMO_head_median*1E-6)))

GMO_head_lay_h2o_median = GMO_head_median*GMO_vol_frac_head_median+(1-GMO_vol_frac_head_median)*-0.558
GMO_head_lay_d2o_median = GMO_head_median*GMO_vol_frac_head_median+(1-GMO_vol_frac_head_median)*6.37097
    
SLDs_median = np.array([
                2.07, 
                3.47, 
                Fe_SLD_median+(((Fe_SLD_median*1E-6)/0.0000945)*Fe_magmom_median*0.00002699*1E6),
                Fe_SLD_median-(((Fe_SLD_median*1E-6)/0.0000945)*Fe_magmom_median*0.00002699*1E6), 
                (FeOx_SLD_median+FeOx_m_median),
                (FeOx_SLD_median-FeOx_m_median), 
                (GMO_head_lay_h2o_median*(1-head_solvation_median)+head_solvation_median*dd_h2o_sld_median),
                (GMO_head_lay_d2o_median*(1-head_solvation_median)+head_solvation_median*dd_d2o_sld_median), 
                (GMO_head_lay_d2o_median*(1-head_solvation_median)+head_solvation_median*-0.462),
                0.211387, #dummy
                0.211387, #dummy
                0.211387, #dummy
                dd_d2o_sld_median, 
                dd_h2o_sld_median, 
                -0.462
                ])
    
list_of_thickness_median = (0, SiO2_thick_median, Fe_thick_median, FeOx_thick_median, GMO_head_thick_median, GMO_tail_thick_median)
list_of_roughness_median = (3, SiO2_Fe_R_median, FeOx_Fe_R_median, FeOx_R_median, GMO_head_rough_median, GMO_tail_rough_median)
list_of_pars_for_cons_median = (GMO_head_median, GMO_vol_frac_head_median, head_solvation_median, GMO_tail_median, solvation_median, GMOav_vol_frac_tail_median)

VFP_tot_thick_median = Parameter(name='VFP_tot_thick_median', constraint=5+(4 * 3)+SiO2_thick_median+Fe_thick_median+FeOx_thick_median+GMO_head_thick_median+GMO_tail_thick_median+5+(4 * GMO_tail_rough_median))

### have a look at SLDs from splines ###
dd_d2o_up_vfp_median = VFP(extent=VFP_tot_thick_median, SLDs=SLDs_median, thicknesses=list_of_thickness_median, roughnesses=list_of_roughness_median, pcons=list_of_pars_for_cons_median, contrast='dd_d2o_up')

GMO_tail_vf_median = dd_d2o_up_vfp_median.create_constraints()[0]

GMO_vf_median_components = (
                    dd_d2o_up_vfp_median.vfs_for_display()[4]*float(GMO_vol_frac_head_median)*(1-float(head_solvation_median)) 
                    + dd_d2o_up_vfp_median.vfs_for_display()[5]*(GMO_tail_vf_median+(GMOav_vol_frac_tail_median*(1-float(solvation_median))))
                    )
GMO_vf_median = np.reshape(GMO_vf_median_components, (1, -1))

water_vf_median_components = (
                       dd_d2o_up_vfp_median.vfs_for_display()[4]*(1-float(GMO_vol_frac_head_median))*(1-float(head_solvation_median))
                       + dd_d2o_up_vfp_median.vfs_for_display()[5]*(1-((GMO_tail_vf_median/(1-float(solvation_median)))+GMOav_vol_frac_tail_median))*(1-float(solvation_median))
                       )
water_vf_median = np.reshape(water_vf_median_components, (1, -1))

solvent_vf_median_components = (
                         dd_d2o_up_vfp_median.vfs_for_display()[4]*float(head_solvation_median)
                         + dd_d2o_up_vfp_median.vfs_for_display()[5]*float(solvation_median)
                         + dd_d2o_up_vfp_median.vfs_for_display()[6]
                         )
solvent_vf_median = np.reshape(solvent_vf_median_components, (1, -1))

median_volume_fracs = dd_d2o_up_vfp_median.vfs_for_display()
sum_of_median_vfs = np.sum(np.vstack((median_volume_fracs[0:4], GMO_vf_median, water_vf_median, solvent_vf_median)).T, 1)

ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], dd_d2o_up_vfp_median.vfs_for_display()[0].T, color='#840200ff', linestyle='--', label='Si')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], dd_d2o_up_vfp_median.vfs_for_display()[1].T, color='#cf9200ff', linestyle='--', label=r'SiO$_{2}$')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], dd_d2o_up_vfp_median.vfs_for_display()[2].T, color='#036d00ff', linestyle='--', label='Fe')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], dd_d2o_up_vfp_median.vfs_for_display()[3].T, color='#9e0005ff', linestyle='--', label=r'FeO$_{x}$')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], GMO_vf_median.T, color='#76008eff', linestyle='--', label='GMO')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], water_vf_median.T, color='#5b2b00ff', linestyle='--', label='water')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], solvent_vf_median.T, color='#e500e1ff', linestyle='--', label='Solvent')
ax3.plot(dd_d2o_up_vfp_median.get_x_and_y_scatter()[0], sum_of_median_vfs, color='#464546ff', linestyle='--', label='total')
ax3.legend(frameon=False)
plt.show()

#now lets describe the distributions.
from uravu.distribution import Distribution
GMO_head_SE_dscatt = Distribution(np.array(GMO_head_SE)*1E20, name='GMO_head_SE', ci_points=(2.5, 97.5))
GMO_tail_SE_dscatt = Distribution(np.array(GMO_tail_SE)*1E20, name='GMO_tail_SE', ci_points=(2.5, 97.5))
GMO_tail_av_SE_dscatt = Distribution(np.array(GMO_t_av_SE)*1E20, name='GMO_t_av_SE', ci_points=(2.5, 97.5))
GMO_SE_dscatt = Distribution(np.array(GMO_SE)*1E20, name='GMO_SE', ci_points=(2.5, 97.5))

print(f'GMO head surface excess = {GMO_head_SE_dscatt.n:.2E} + {(GMO_head_SE_dscatt.con_int[1]-GMO_head_SE_dscatt.n):.2E} - {(GMO_head_SE_dscatt.n-GMO_head_SE_dscatt.con_int[0]):.2E}')
print(f'GMO tail surface excess = {GMO_tail_SE_dscatt.n:.2E} + {(GMO_tail_SE_dscatt.con_int[1]-GMO_tail_SE_dscatt.n):.2E} - {(GMO_tail_SE_dscatt.n-GMO_tail_SE_dscatt.con_int[0]):.2E}')
print(f'GMO tail surface excess = {GMO_tail_av_SE_dscatt.n:.2E} + {(GMO_tail_av_SE_dscatt.con_int[1]-GMO_tail_av_SE_dscatt.n):.2E} - {(GMO_tail_av_SE_dscatt.n-GMO_tail_av_SE_dscatt.con_int[0]):.2E}')
print(f'GMO surface excess = {GMO_SE_dscatt.n:.2E} + {(GMO_SE_dscatt.con_int[1]-GMO_SE_dscatt.n):.2E} - {(GMO_SE_dscatt.n-GMO_SE_dscatt.con_int[0]):.2E}')

water_SE_dscatt = Distribution(np.array(water_SE)*1E20, name='water_SE', ci_points=(2.5, 97.5))
print(f'Water surface excess = {water_SE_dscatt.n:.2E} + {(water_SE_dscatt.con_int[1]-water_SE_dscatt.n):.2E} - {(water_SE_dscatt.n-water_SE_dscatt.con_int[0]):.2E}')

ratio = Distribution(np.array(water_SE)/np.array(GMO_SE), name='ratio', ci_points=(2.5, 97.5))
print(f'Water:GMO ratio = {ratio.n:.2E} + {(ratio.con_int[1]-ratio.n):.2E} - {(ratio.n-ratio.con_int[0]):.2E}')

raise Exception()