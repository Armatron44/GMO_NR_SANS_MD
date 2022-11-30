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

#load the model & chain.
modelname = "Fit_all_contrasts"
state = load_state(modelname)
with open(modelname+".par") as fid:
    state.labels = [" ".join(line.strip().split()[:-1]) for line in fid]
	
#there is a dependancy on the initial state of the chain in the first 40000 steps
chainz = state.chains()[1] #this is in samples, walkers, parameters
print(chainz.shape)
chainz_burned = chainz[400:, :, :] #if we want the first 40000 steps to be removed, we want to ignore the first 40000/w = 400 samples of each walker.

chainz_rhat = np.reshape(chainz_burned, (chainz_burned.shape[1], chainz_burned.shape[0], chainz_burned.shape[2])) #reshape to be walkers, samples, parameters
rhat = az.rhat(az.convert_to_dataset(chainz_rhat)) #calculate rhat statistic, these should be around 1.0
print(rhat.to_array())

#now reshape so that we have array of samples for each parameter.
samples = np.reshape(chainz_burned, (chainz_burned.shape[0]*chainz_burned.shape[1], chainz_burned.shape[2]))

variables = []
for i in range(0, samples.shape[1]): #create a list of Distributions that describe the posterior of the parameters.
    variables.append(Distribution(samples[:, i], name=state.labels[i], ci_points=(2.5, 97.5)))

alpha = 0.001 #sets criteria on the normal distribution test.
#print out distribution characteristics.
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

#derive Re, Rp and Rg for the GMO+water contrasts. These are unimodal distributions.
Re_GMO_water = Distribution((variables[7].samples/variables[5].samples*(3/4)*(1/np.pi))**(1/3), name="Re_GMO_water", ci_points=(2.5, 97.5))
Rp_GMO_water = Distribution(Re_GMO_water.samples*variables[5].samples, name="Rp_GMO_water", ci_points=(2.5, 97.5))
Rg_GMO_water = Distribution(np.sqrt((2*Re_GMO_water.samples**2+Rp_GMO_water.samples**2)/5), name="Rg_GMO_water", ci_points=(2.5, 97.5))

#derive other parameters - aggregation number of GMO, volume fraction of GMO + water etc.
vey = 5*(0.0016/0.0333)
Agg_GMO_GMO_water = Distribution((variables[7].samples - vey*(variables[7].samples-(variables[7].samples*0))/(1+vey))/(1/0.0016), name="Agg_GMO_water", ci_points=(2.5, 97.5))
VGMO_GMO_water = Distribution((variables[7].samples - vey*(variables[7].samples-(variables[7].samples*0))/(1+vey)), name="VGMO_GMO_water", ci_points=(2.5, 97.5))
Vwater_GMO_water = Distribution((vey*(variables[7].samples-(variables[7].samples*0))/(1+vey)), name="Vwater_GMO_water", ci_points=(2.5, 97.5))
GMO_vf = Distribution((variables[7].samples - vey*(variables[7].samples-(variables[7].samples*0))/(1+vey))/variables[7].samples, name="phiGMO", ci_points=(2.5, 97.5))
water_vf = Distribution((vey*(variables[7].samples-(variables[7].samples*0))/(1+vey))/variables[7].samples, name="phiwater", ci_points=(2.5, 97.5))
tot_vf = Distribution(GMO_vf.samples+water_vf.samples, name="tot_vf", ci_points=(2.5, 97.5))

d_params = [Re_GMO_water, Rp_GMO_water, Rg_GMO_water, Agg_GMO_GMO_water, VGMO_GMO_water, Vwater_GMO_water, GMO_vf, water_vf, tot_vf]

alpha = 0.001
for i, j in enumerate(d_params):
    print(f"""{d_params[i].name}
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

### CORNER PLOTS ###
### Dry GMO contrast parameters
### define labels for corner plot
GMO_bkg_label = r'$B$ / cm$^{-1}$'
GMO_ecc_label = r'$e$'
GMO_scale_label = r'$C$'
GMO_vol_label = r'$V$ / $\mathrm{\AA}^{3}$'
GMO_labels = [GMO_bkg_label, GMO_ecc_label, GMO_scale_label, GMO_vol_label]

figureo = plt.figure(figsize=(120*0.0393701*2, 120*0.0393701*2))
list_range = [0.999, 0.999, 0.999, 0.999, 0.999, 0.999]
SMALL_SIZE = 10*2
MEDIUM_SIZE = 12*2
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the ax[0]es title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
colorme = {'colors':['k', 'k', 'k', 'k'], 'linewidths': 0.87}
colormef = {'colors':['w', '#00000033', '#00000059', '#00000080', '#000000a6']}
k = corner.corner(samples[:, 0:4], 
                  plot_contours=True, 
                  color='k',
                  bins=100,
                  plot_datapoints=False, 
                  plot_density=False, 
                  titles=GMO_labels, 
                  show_titles=True, 
                  max_n_ticks=2, 
                  fill_contours=True, 
                  title_fmt=None, 
                  contourf_kwargs=colormef, 
                  fig=figureo, 
                  contour_kwargs=colorme)
plt.show()      
            
### CORNER PLOTS ### 
### wet GMO contrast parameters
### define labels for corner plot
GMOwater_bkgH_label = r'\noindent \begin{center} $B_{\mathrm{H}_{2}\mathrm{O}}$ \\ / cm$^{-1}$ \end{center}'
GMOwater_ecc_label = r'$e$'
GMOwater_Hscale_label = r'$C_{\mathrm{H}_{2}\mathrm{O}}$'
GMOwater_vol_label = r'$V$ / $\mathrm{\AA}^{3}$'
GMOwater_bkgD_label = r'\noindent \begin{center} $B_{\mathrm{D}_{2}\mathrm{O}}$ \\ / cm$^{-1}$ \end{center}'
GMOwater_Dscale_label = r'$C_{\mathrm{D}_{2}\mathrm{O}}$'

GMOwater_labels = [GMOwater_bkgH_label, GMOwater_ecc_label, GMOwater_Hscale_label, GMOwater_vol_label, GMOwater_bkgD_label, GMOwater_Dscale_label]

figureo = plt.figure(figsize=(130*0.0393701*2, 130*0.0393701*2))
list_range = [0.999, 0.999, 0.999, 0.999, 0.999, 0.999]
SMALL_SIZE = 10*2
MEDIUM_SIZE = 12*2
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the ax[0]es title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', serif='Computer Modern Roman')
colorme = {'colors':['k', 'k', 'k', 'k'], 'linewidths': 0.87}
colormef = {'colors':['w', '#00000033', '#00000059', '#00000080', '#000000a6']}
k = corner.corner(samples[:, 4:10], 
                  plot_contours=True, 
                  color='k',
                  bins=50,
                  range=list_range,
                  plot_datapoints=False, 
                  plot_density=False, 
                  titles=GMOwater_labels, 
                  show_titles=True, 
                  max_n_ticks=2, 
                  fill_contours=True, 
                  title_fmt=None, 
                  contourf_kwargs=colormef, 
                  fig=figureo, 
                  contour_kwargs=colorme)
plt.show()

# Find HDIs of each parameter.

for i in range(0, samples.shape[1]):
    if i == 1: #the eccentricity parameter is bi modal & multimodal so we have to get the hdi twice.
        low_hdi, up_hdi = az.hdi(variables[i].samples, hdi_prob=.95, multimodal=True)[0][0], az.hdi(variables[i].samples, hdi_prob=.95, multimodal=True)[0][1]
        centre = np.average(np.array([az.hdi(variables[i].samples, hdi_prob=.01)[0], az.hdi(variables[i].samples, hdi_prob=.01)[1]]))
        low_hdi2, up_hdi2 = az.hdi(variables[i].samples, hdi_prob=.95, multimodal=True)[1][0], az.hdi(variables[i].samples, hdi_prob=.95, multimodal=True)[1][1]
        centre2 = np.average(np.array([az.hdi(variables[i].samples[variables[i].samples > 1], hdi_prob=.01)[0], az.hdi(variables[i].samples[variables[i].samples > 1], hdi_prob=.01)[1]]))
        print(f""" Pname: {variables[i].name}
        Centre: {centre:.5E}
        Lower: {centre-low_hdi:.5E}
        Upper: {up_hdi-centre:.5E}
        Centre2: {centre2:.5E}
        Lower2: {centre2-low_hdi2:.5E}
        Upper2: {up_hdi2-centre2:.5E}""")
    else: #the rest are either unimodal or bimodal but not seperated by probability.
        low_hdi, up_hdi = az.hdi(variables[i].samples, hdi_prob=.95, multimodal=False)[0], az.hdi(variables[i].samples, hdi_prob=.95, multimodal=False)[1]
        centre = np.average(np.array([az.hdi(variables[i].samples, hdi_prob=.01)[0], az.hdi(variables[i].samples, hdi_prob=.01)[1]]))
        print(f"""Pname: {variables[i].name}
        Centre: {centre:.5E}
        Lower: {centre-low_hdi:.5E}
        Upper: {up_hdi-centre:.5E}""")

#calculate Re and Rp of dry GMO contrast.
Re_GMO = ((variables[3].samples/variables[1].samples)*(3/4)*(1/np.pi))**(1/3)
Rp_GMO = (Re_GMO*variables[1].samples)

#then find the hdi description of these distributions.
Re_GMO_low_hdi, Re_GMO_up_hdi = az.hdi(Re_GMO, hdi_prob=.95, multimodal=True)[0][0], az.hdi(Re_GMO, hdi_prob=.95, multimodal=True)[0][1]
Re_GMO_centre = np.average(np.array([az.hdi(Re_GMO, hdi_prob=.01)[0], az.hdi(Re_GMO, hdi_prob=.01)[1]]))       
Re_GMO_low_hdi2, Re_GMO_up_hdi2 = az.hdi(Re_GMO, hdi_prob=.95, multimodal=True)[1][0], az.hdi(Re_GMO, hdi_prob=.95, multimodal=True)[1][1]
Re_GMO_centre2 = np.average(np.array([az.hdi(Re_GMO[Re_GMO < 20], hdi_prob=.01)[0], az.hdi(Re_GMO[Re_GMO < 20], hdi_prob=.01)[1]]))

print(f""" Pname: Re_GMO
Centre: {Re_GMO_centre:.5E}
Lower: {Re_GMO_centre-Re_GMO_low_hdi:.5E}
Upper: {Re_GMO_up_hdi-Re_GMO_centre:.5E}
Centre2: {Re_GMO_centre2:.5E}
Lower2: {Re_GMO_centre2-Re_GMO_low_hdi2:.5E}
Upper2: {Re_GMO_up_hdi2-Re_GMO_centre2:.5E}""")

GMO_agg = variables[3].samples/(1/0.0016)
low_hdi, up_hdi = az.hdi(GMO_agg, hdi_prob=.95, multimodal=False)[0], az.hdi(GMO_agg, hdi_prob=.95, multimodal=False)[1]
centre = np.average(np.array([az.hdi(GMO_agg, hdi_prob=.01)[0], az.hdi(GMO_agg, hdi_prob=.01)[1]]))
print(f""" Pname: GMO_agg
Centre: {centre:.5E}
Lower: {centre-low_hdi:.5E}
Upper: {up_hdi-centre:.5E}""")       


Rp_GMO_low_hdi, Rp_GMO_up_hdi = az.hdi(Rp_GMO, hdi_prob=.95, multimodal=True)[0][0], az.hdi(Rp_GMO, hdi_prob=.95, multimodal=True)[0][1]
Rp_GMO_centre = np.average(np.array([az.hdi(Rp_GMO, hdi_prob=.01)[0], az.hdi(Rp_GMO, hdi_prob=.01)[1]]))       
Rp_GMO_low_hdi2, Rp_GMO_up_hdi2 = az.hdi(Rp_GMO, hdi_prob=.95, multimodal=True)[1][0], az.hdi(Rp_GMO, hdi_prob=.95, multimodal=True)[1][1]
Rp_GMO_centre2 = np.average(np.array([az.hdi(Rp_GMO[Rp_GMO > 20], hdi_prob=.01)[0], az.hdi(Rp_GMO[Rp_GMO > 20], hdi_prob=.01)[1]]))
print(f""" Pname: Rp_GMO
Centre: {Rp_GMO_centre:.5E}
Lower: {Rp_GMO_centre-Rp_GMO_low_hdi:.5E}
Upper: {Rp_GMO_up_hdi-Rp_GMO_centre:.5E}
Centre2: {Rp_GMO_centre2:.5E}
Lower2: {Rp_GMO_centre2-Rp_GMO_low_hdi2:.5E}
Upper2: {Rp_GMO_up_hdi2-Rp_GMO_centre2:.5E}""")