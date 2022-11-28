import sys
import os

sys.path.append(os.getcwd())

from refnx.dataset import ReflectDataset
from refnx.analysis import Transform, CurveFitter, Objective, GlobalObjective, Parameter, autocorrelation_chain, process_chain, integrated_time
from refnx.reflect import SLD, ReflectModel, MixedReflectModel, MixedSlab, Slab, Structure
import MixedMagSlabs2
from vfp_M2 import VFP
import refnx
import scipy
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import pickle
import copy
import h5py
from uravu.distribution import Distribution

Si = SLD(2.07, name='Si')
SiO2 = SLD(3.47, name='SiO2')
Fe = SLD(8.02, name='Fe')
FeOx = SLD(7.0, name='FeOx')
D2O = SLD(6.37097, name='D2O')
GMO = SLD(0.211387, name='<GMO>')
H2O = SLD(-0.558, name='H2O')

hdod = SLD(-0.462, name='hdod')

b_head = 0.00023623
b_tail = -0.00010405

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
        
with open(os.path.join(os.getcwd(), 'M2_FIT_GO.pkl'),"rb") as f:
	GO = pickle.load(f)

#process the chains of parameter values.
chain_arrays = [GO.varying_parameters().flattened()[i].chain for i in range(0, len(GO.varying_parameters()))] # get the chains of each parameter.
chain_processed = np.stack(chain_arrays, axis=2) #shape it so that its (nsteps, nwalkers, ndim) format.
print(chain_processed.shape)

#determine the rhat statistic (gelman-rubin statistic) value for each parameter (should be v close to 1)
rhat = az.rhat(az.convert_to_dataset(chain_processed))
print(rhat.to_array())

#calculate effective sample size for each parameter.
ess = az.ess(az.convert_to_dataset(chain_processed))
print(ess.to_array())

#create a list of parameters
variables = []
for i in range(0, len(GO.varying_parameters())):
    variables.append(Distribution(GO.varying_parameters().flattened()[i].chain.flatten(), name=GO.varying_parameters().flattened()[i].name, ci_points=(2.5, 97.5))) #(2.5, 97.5) #(16, 84)

#print out the median & 95 % CI for parameter.
alpha = 0.001
for i, j in enumerate(variables):
    print(f"""{variables[i].name}
    Median: {j.n:.3E}
    Lower: {j.con_int[0]:.3E}
    Upper: {j.con_int[1]:.3E}
    Minus: {j.n-j.con_int[0]:.3E}
    Plus: {j.con_int[1]-j.n:.3E}
    Is gauss?: {'Yes' if scipy.stats.normaltest(j.samples)[1] >= alpha else 'No'}
    Gauss_test_p: {scipy.stats.normaltest(j.samples)[1]:.3E}
    MCSE_median: {az.mcse(j.samples, method="quantile", prob=0.5):.3E}
    MCSE_lb: {az.mcse(j.samples, method="quantile", prob=0.025):.3E}
    MCSE_ub: {az.mcse(j.samples, method="quantile", prob=0.975):.3E}""")

#plot corner
import corner

#get the label for each parameter.

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
plt.rc('font', size=11)

#make labels on corner look nice.
labels = [
    r'inner' '\n' r'$\phi_{\mathrm{GMO}}$ / \%',
    r'outer' '\n' r'$\phi_{\mathrm{GMO}}$ / \%',
    r'inner $\phi_{\mathrm{s}}$ /' '\n' r'\%',
    r'outer $\phi_{\mathrm{s}}$ /' '\n' r'\%',
    r'SiO$_{2}$ $d$ / $\mathrm{\AA{}}$',
    r'SiO$_{2}$ $\sigma$ / $\mathrm{\AA{}}$',
    r'Fe $d$ / $\mathrm{\AA{}}$',
    r'Fe $\beta_{\mathrm{n}}$ /' '\n' r'$\mathrm{\AA{}}^{-2}\times{}10^{-6}$',
    r'Fe $\mu$ / $\mu_{\mathrm{B}}$',
    r'Fe $\sigma$ / $\mathrm{\AA{}}$',
    r'FeO$_{x}$ $\beta_{\mathrm{n}}$ /' '\n' r'$\mathrm{\AA{}}^{-2}\times{}10^{-6}$',
    r'FeO$_{x}$ $d$ / $\mathrm{\AA{}}$',
    r'FeO$_{x}$ $\beta_{\mathrm{m}}$ /' '\n' r'$\mathrm{\AA{}}^{-2}\times{}10^{-6}$',
    r'FeO$_{x}$ $\sigma$ / $\mathrm{\AA{}}$',
    r'inner $d$ / $\mathrm{\AA{}}$',
    r'inner $\sigma$ / $\mathrm{\AA{}}$',
    r'outer $d$ / $\mathrm{\AA{}}$',
    r'outer $\sigma$ / $\mathrm{\AA{}}$',
    r'dod-d$_{26}$-D$_{2}$O' '\n' r'/ $\mathrm{\AA{}}^{-2}\times{}10^{-6}$',
    r'dod-d$_{26}$-H$_{2}$O' '\n' r'/ $\mathrm{\AA{}}^{-2}\times{}10^{-6}$',
    r'inner $\beta_{\mathrm{head}}$ /' '\n' r'$\mathrm{\AA{}}^{-2}\times{}10^{-6}$'
]

#get the chain of samples for each parameter and reshape it for the corner plot.
var_pars = GO.varying_parameters()
chain_corner = np.array([par.chain for par in var_pars])
chain_corner = chain_corner.reshape(len(chain_corner), -1).T #this ends up with a matrix that is
# n x m - n = number of samples per parameter and m is the number of parameters.

#scale volume fractions to percentages as in the tables.
chain_corner[:, 0] = chain_corner[:, 0]*100
chain_corner[:, 1] = chain_corner[:, 1]*100
chain_corner[:, 2] = chain_corner[:, 2]*100
chain_corner[:, 3] = chain_corner[:, 3]*100

#make some grayscale colors for the contour map we'll be looking at...
color_cont = {'colors':['k', 'k', 'k', 'k'], 'linewidths': 0.87}
color_cont_fill = {'colors':['w', '#00000033', '#00000059', '#00000080', '#000000a6']}

title_font = {'fontsize': 12}

#now plot
corner.corner(chain_corner[:, :-6], titles=labels, show_titles=True, quantiles=[0.025, 0.5, 0.975], 
              plot_contours=True, color='k', plot_datapoints=False, plot_density=False, title_fmt=None,
              max_n_ticks=2, fill_contours=True, contour_kwargs=color_cont, contourf_kwargs=color_cont_fill,
              title_kwargs={'pad':'10', 'fontdict':title_font})

plt.show()

#calculate volume fractions
inner_GMO_vf = Distribution(variables[0].samples*(1-variables[2].samples), name='inner_GMO_vf', ci_points=(2.5, 97.5))
inner_water_vf = Distribution((1-variables[0].samples)*(1-variables[2].samples), name='inner_water_vf', ci_points=(2.5, 97.5))
sum_inner_vf = Distribution((variables[0].samples*(1-variables[2].samples)+(1-variables[0].samples)*(1-variables[2].samples)+variables[2].samples), name='sum_inner_vf', ci_points=(2.5, 97.5)) #check we add up to 1.

print(f'inner_GMO_vf = {inner_GMO_vf.n*100:.1f} + {(inner_GMO_vf.con_int[1]*100-inner_GMO_vf.n*100):.1f} - {(inner_GMO_vf.n*100-inner_GMO_vf.con_int[0]*100):.1f}')
print(f'inner_water_vf = {inner_water_vf.n*100:.1f} + {(inner_water_vf.con_int[1]*100-inner_water_vf.n*100):.1f} - {(inner_water_vf.n*100-inner_water_vf.con_int[0]*100):.1f}')
print(f'inner_solvent_vf = {variables[2].n*100:.1f} + {(variables[2].con_int[1]*100-variables[2].n*100):.1f} - {(variables[2].n*100-variables[2].con_int[0]*100):.1f}')
print(f'tot_solv_vf = {sum_inner_vf.n*100:.1f} + {(sum_inner_vf.con_int[1]*100-sum_inner_vf.n*100):.1f} - {(sum_inner_vf.n*100-sum_inner_vf.con_int[0]*100):.1f}')

#calculate volume fractions!
#won't plot volume fractions here.
vf_obj = copy.deepcopy(GO)

SMALL_SIZE = 10*3
MEDIUM_SIZE = 12*3
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the ax[0]es title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
fig, ax = plt.subplots(figsize=(84.6*0.0393701*3, 60*0.0393701*3))

GMO_SE = []
water_SE = []

GMO_t_av_SE = []
GMO_head_SE = []
GMO_tail_SE = []
GMO_tail_vf_list = []
GMO_wat_vol_frac_list = []
GMOav_vol_frac_tail_list = []
solvation_list = []

#note - this step takes a long time (~ 1 hour)
for i in GO.varying_parameters().pgen(ngen=20000):
    vf_obj.varying_parameters().pvals = i
    
    #label parameters for easy reading:
    GMO_vol_frac_head = vf_obj.varying_parameters()[0] #defined without including solvation.
    GMOav_vol_frac_tail = vf_obj.varying_parameters()[1] #defined without including solvation.
    head_solvation = vf_obj.varying_parameters()[2]
    solvation = vf_obj.varying_parameters()[3]
    SiO2_thick = vf_obj.varying_parameters()[4]
    SiO2_Fe_R = vf_obj.varying_parameters()[5]
    Fe_thick = vf_obj.varying_parameters()[6]
    Fe_SLD = vf_obj.varying_parameters()[7]
    Fe_magmom = vf_obj.varying_parameters()[8]
    FeOx_Fe_R = vf_obj.varying_parameters()[9]
    FeOx_SLD = vf_obj.varying_parameters()[10]
    FeOx_thick = vf_obj.varying_parameters()[11]
    FeOx_m = vf_obj.varying_parameters()[12]
    FeOx_R = vf_obj.varying_parameters()[13]
    GMO_head_thick = vf_obj.varying_parameters()[14]
    GMO_head_rough = vf_obj.varying_parameters()[15]
    GMO_tail_thick = vf_obj.varying_parameters()[16]
    GMO_tail_rough = vf_obj.varying_parameters()[17]
    dd_d2o_sld = vf_obj.varying_parameters()[18]
    dd_h2o_sld = vf_obj.varying_parameters()[19]
    GMO_head = vf_obj.varying_parameters()[20]
    GMO_tail = 1E6*b_tail/((1/0.001599235)-(b_head/(GMO_head*1E-6)))
    
    #calc slds & vfs.
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
    GMO_wat_vol_frac = dd_d2o_up_vfp.create_constraints()[1]
    GMO_tail_vf_list.append(float(GMO_tail_vf))
    GMO_wat_vol_frac_list.append(float(GMO_wat_vol_frac))
    GMOav_vol_frac_tail_list.append(float(GMOav_vol_frac_tail))
    solvation_list.append(float(solvation))
    
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
    
    #water_SE
    water_vf_components = np.array([float(i) for i in water_vf_components])
    integrate_over = dd_d2o_up_vfp.get_x_and_y_scatter()[0]
    area_under_water = np.trapz(water_vf_components, x=integrate_over)
    water_SE.append(float(((D2O.real*1E-6)/(6.0221409*1E23*1.914E-04))*area_under_water))
    
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

from uravu.distribution import Distribution

GMO_head_SE_dscatt = Distribution(np.array(GMO_head_SE)*1E20, name='GMO_head_SE', ci_points=(2.5, 97.5))
GMO_tail_SE_dscatt = Distribution(np.array(GMO_tail_SE)*1E20, name='GMO_tail_SE', ci_points=(2.5, 97.5))
GMO_tail_av_SE_dscatt = Distribution(np.array(GMO_t_av_SE)*1E20, name='GMO_t_av_SE', ci_points=(2.5, 97.5))
GMO_SE_dscatt = Distribution(np.array(GMO_SE)*1E20, name='GMO_SE', ci_points=(2.5, 97.5))

water_SE_dscatt = Distribution(np.array(water_SE)*1E20, name='water_SE', ci_points=(2.5, 97.5))

GMO_APM = Distribution((1/(np.array(GMO_SE)*6.0221409*1E23)), name='GMO_APM', ci_points=(2.5, 97.5))
water_APM = Distribution((1/(np.array(water_SE)*6.0221409*1E23)), name='water_APM', ci_points=(2.5, 97.5))
ratio = Distribution(np.array(water_SE)/np.array(GMO_SE), name='ratio', ci_points=(2.5, 97.5))

print(f'GMO head surface excess = {GMO_head_SE_dscatt.n:.2E} + {(GMO_head_SE_dscatt.con_int[1]-GMO_head_SE_dscatt.n):.2E} - {(GMO_head_SE_dscatt.n-GMO_head_SE_dscatt.con_int[0]):.2E}')
print(f'GMO tail surface excess = {GMO_tail_SE_dscatt.n:.2E} + {(GMO_tail_SE_dscatt.con_int[1]-GMO_tail_SE_dscatt.n):.2E} - {(GMO_tail_SE_dscatt.n-GMO_tail_SE_dscatt.con_int[0]):.2E}')
print(f'GMO tail surface excess = {GMO_tail_av_SE_dscatt.n:.2E} + {(GMO_tail_av_SE_dscatt.con_int[1]-GMO_tail_av_SE_dscatt.n):.2E} - {(GMO_tail_av_SE_dscatt.n-GMO_tail_av_SE_dscatt.con_int[0]):.2E}')
print(f'GMO surface excess = {GMO_SE_dscatt.n:.2E} + {(GMO_SE_dscatt.con_int[1]-GMO_SE_dscatt.n):.2E} - {(GMO_SE_dscatt.n-GMO_SE_dscatt.con_int[0]):.2E}')

print(f'Water surface excess = {water_SE_dscatt.n:.2E} + {(water_SE_dscatt.con_int[1]-water_SE_dscatt.n):.2E} - {(water_SE_dscatt.n-water_SE_dscatt.con_int[0]):.2E}')
print(f'Water:GMO ratio = {ratio.n:.2E} + {(ratio.con_int[1]-ratio.n):.2E} - {(ratio.n-ratio.con_int[0]):.2E}')

print(f'GMO surface excess = {GMO_SE_dscatt.n:.2E} + {(GMO_SE_dscatt.con_int[1]-GMO_SE_dscatt.n):.2E} - {(GMO_SE_dscatt.n-GMO_SE_dscatt.con_int[0]):.2E}')
print(f"Is gauss?: {'Yes' if scipy.stats.normaltest(GMO_SE_dscatt.samples)[1] >= alpha else 'No'}")
print(f"Gauss_test_p: {scipy.stats.normaltest(GMO_SE_dscatt.samples)[1]:.3E}")

print(f'GMO area per molecule = {GMO_APM.n:.2E} + {(GMO_APM.con_int[1]-GMO_APM.n):.2E} - {(GMO_APM.n-GMO_APM.con_int[0]):.2E}')
print(f"Is gauss?: {'Yes' if scipy.stats.normaltest(GMO_APM.samples)[1] >= alpha else 'No'}")
print(f"Gauss_test_p: {scipy.stats.normaltest(GMO_APM.samples)[1]:.3E}")

print(f'Water surface excess = {water_SE_dscatt.n:.2E} + {(water_SE_dscatt.con_int[1]-water_SE_dscatt.n):.2E} - {(water_SE_dscatt.n-water_SE_dscatt.con_int[0]):.2E}')
print(f"Is gauss?: {'Yes' if scipy.stats.normaltest(water_SE_dscatt.samples)[1] >= alpha else 'No'}")
print(f"Gauss_test_p: {scipy.stats.normaltest(water_SE_dscatt.samples)[1]:.3E}")

print(f'Water area per molecule = {water_APM.n:.2E} + {(water_APM.con_int[1]-water_APM.n):.2E} - {(water_APM.n-water_APM.con_int[0]):.2E}')
print(f"Is gauss?: {'Yes' if scipy.stats.normaltest(water_APM.samples)[1] >= alpha else 'No'}")
print(f"Gauss_test_p: {scipy.stats.normaltest(water_APM.samples)[1]:.3E}")

print(f'Water:GMO ratio = {ratio.n:.2E} + {(ratio.con_int[1]-ratio.n):.2E} - {(ratio.n-ratio.con_int[0]):.2E}')
print(f"Is gauss?: {'Yes' if scipy.stats.normaltest(ratio.samples)[1] >= alpha else 'No'}")
print(f"Gauss_test_p: {scipy.stats.normaltest(ratio.samples)[1]:.3E}")

print(f'inner_GMO_vf = {inner_GMO_vf.n*100:.1f} + {(inner_GMO_vf.con_int[1]*100-inner_GMO_vf.n*100):.1f} - {(inner_GMO_vf.n*100-inner_GMO_vf.con_int[0]*100):.1f}')
print(f'inner_water_vf = {inner_water_vf.n*100:.1f} + {(inner_water_vf.con_int[1]*100-inner_water_vf.n*100):.1f} - {(inner_water_vf.n*100-inner_water_vf.con_int[0]*100):.1f}')
print(f'inner_solvent_vf = {variables[2].n*100:.1f} + {(variables[2].con_int[1]*100-variables[2].n*100):.1f} - {(variables[2].n*100-variables[2].con_int[0]*100):.1f}')

GMO_wat_vol_frac_dist = Distribution(np.array(GMO_wat_vol_frac_list), name='GMO_wat_vol_frac_dist', ci_points=(2.5, 97.5))
GMO_tail_vf_dist = Distribution(np.array(GMO_tail_vf_list), name='GMO_tail_vol_frac_dist', ci_points=(2.5, 97.5))

outer_GMO_av_vf_dist = Distribution(np.array(GMOav_vol_frac_tail_list), name='GMOav_vol_frac_tail_list', ci_points=(2.5, 97.5))
outer_solvation_dist = Distribution(np.array(solvation_list), name='solvation_list', ci_points=(2.5, 97.5))

print(f'outer_GMO_av_vf = {outer_GMO_av_vf_dist.n*100:.1f} + {(outer_GMO_av_vf_dist.con_int[1]*100-outer_GMO_av_vf_dist.n*100):.1f} - {(outer_GMO_av_vf_dist.n*100-outer_GMO_av_vf_dist.con_int[0]*100):.1f}')
print(f'outer_GMO_av_vf_from_variables = {variables[1].n*100:.1f} + {(variables[1].con_int[1]*100-variables[1].n*100):.1f} - {(variables[1].n*100-variables[1].con_int[0]*100):.1f}')

print(f'outer_solvation = {outer_solvation_dist.n*100:.1f} + {(outer_solvation_dist.con_int[1]*100-outer_solvation_dist.n*100):.1f} - {(outer_solvation_dist.n*100-outer_solvation_dist.con_int[0]*100):.1f}')
print(f'outer_solvation_from_variables = {variables[3].n*100:.1f} + {(variables[3].con_int[1]*100-variables[3].n*100):.1f} - {(variables[3].n*100-variables[3].con_int[0]*100):.1f}')

outer_GMO_av_vf_fl = Distribution(outer_GMO_av_vf_dist.samples*(1-outer_solvation_dist.samples), name='outer_GMO_av_vf_fl', ci_points=(2.5, 97.5))
outer_GMO_av_vf = Distribution((variables[1].samples*(1-variables[3].samples)), name='GMO_av_vf', ci_points=(2.5, 97.5))

print(f'outer_GMO_av_withsolv = {outer_GMO_av_vf_fl.n*100:.1f} + {(outer_GMO_av_vf_fl.con_int[1]*100-outer_GMO_av_vf_fl.n*100):.1f} - {(outer_GMO_av_vf_fl.n*100-outer_GMO_av_vf_fl.con_int[0]*100):.1f}')
print(f'outer_GMO_av_withsolv_from_variables = {outer_GMO_av_vf.n*100:.1f} + {(outer_GMO_av_vf.con_int[1]*100-outer_GMO_av_vf.n*100):.1f} - {(outer_GMO_av_vf.n*100-outer_GMO_av_vf.con_int[0]*100):.1f}')

outer_water_vf = Distribution((GMO_wat_vol_frac_dist.samples*(1-outer_solvation_dist.samples)), name='outer_water_vf', ci_points=(2.5, 97.5))

outer_GMO_vf = Distribution(GMO_tail_vf_dist.samples+(outer_GMO_av_vf_dist.samples*(1-outer_solvation_dist.samples)), name='outer_GMO_vf', ci_points=(2.5, 97.5))
sum_outer_vf = Distribution(outer_water_vf.samples+outer_GMO_vf.samples+outer_solvation_dist.samples, name='sum_outer_vf', ci_points=(2.5, 97.5))

print(f'tail_GMO_vf = {GMO_tail_vf_dist.n*100:.1f} + {(GMO_tail_vf_dist.con_int[1]*100-GMO_tail_vf_dist.n*100):.1f} - {(GMO_tail_vf_dist.n*100-GMO_tail_vf_dist.con_int[0]*100):.1f}')
print(f'tail_GMO_av_vf = {outer_GMO_av_vf.n*100:.1f} + {(outer_GMO_av_vf.con_int[1]*100-outer_GMO_av_vf.n*100):.1f} - {(outer_GMO_av_vf.n*100-outer_GMO_av_vf.con_int[0]*100):.1f}')
print(f'outer_GMO_vf = {outer_GMO_vf.n*100:.1f} + {(outer_GMO_vf.con_int[1]*100-outer_GMO_vf.n*100):.1f} - {(outer_GMO_vf.n*100-outer_GMO_vf.con_int[0]*100):.1f}')
print(f'outer_water_vf = {outer_water_vf.n*100:.1f} + {(outer_water_vf.con_int[1]*100-outer_water_vf.n*100):.1f} - {(outer_water_vf.n*100-outer_water_vf.con_int[0]*100):.1f}')
print(f'sum_outer_vf = {sum_outer_vf.n*100:.1f} + {(sum_outer_vf.con_int[1]*100-sum_outer_vf.n*100):.1f} - {(sum_outer_vf.n*100-sum_outer_vf.con_int[0]*100):.1f}')