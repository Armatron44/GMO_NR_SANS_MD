import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bumps.fitproblem import load_problem
from bumps.cli import load_best
from bumps.dream.state import load_state
#from bumps.errplot import calc_errors_from_state
from refl1d.errors import align_profiles
from refl1d.errors import show_errors
from refl1d.errors import show_profiles

import traceback
import logging

model, store = sys.argv[1:3]

problem = load_problem(model)

load_best(problem, os.path.join(store, model[:-3]+".par"))

chisquare = 0
for i in list(problem.models):
    chisquare += i.chisq()*(i.model_points()-len(i._parameters))
reduced_chisquare = chisquare/problem.model_points()

print(f'sum is {chisquare} and reduced value is {reduced_chisquare}')

experiment = list(problem.models)

# redefine some of the functions used to get posterior traces in bumps & refl1d 
# & use here so that we can find the posterior bands for mixedexperiments.
def calc_errors_refl1d(problem, points, origp):
    """
    Align the sample profiles and compute the residual difference from the
    measured reflectivity for a set of points.

    The points should be sampled from the posterior probability
    distribution computed from MCMC, bootstrapping or sampled from
    the error ellipse calculated at the minimum.

    Each of the returned arguments is a dictionary mapping model number to
    error sample data as follows:

    Returns (profiles, slabs, Q, residuals).

    *profiles*

        Arrays of (z, rho, irho) for non-magnetic models or arrays
        of (z, rho, irho, rhoM, thetaM) for magnetic models.  There
        will be one set of arrays returned per error sample.

    *slabs*

        Array of slab thickness for the layers in the models.  There
        will be one array returned per error sample.  Using slab thickness,
        profiles can be aligned on interface boundaries and layer centers.

    *Q*

        Array of Q values for the data points in the model.  The data
        points are the same for all error samples, so only one Q array
        is needed per model.

    *residuals*

        Array of (theory-data)/uncertainty for each data point in
        the measurement.  There will be one array returned per error sample.
    """
    # Grab the individual samples
    if hasattr(problem, 'models'):
        models = [m.fitness for m in problem.models]
    else:
        models = [problem.fitness]

    experiments = []
    for m in models:
        if hasattr(m, 'parts'):
            experiments.extend(m.parts)
        else:
            experiments.append(m)

    # Find Q
    def residQ(m):
        if m.probe.polarized:
            return np.hstack([xs.Q for xs in m.probe.xs if xs is not None])
        else:
            return m.probe.Q
    Q = dict((m, residQ(m)) for m in experiments)
    profiles = dict((m, []) for m in experiments)
    reflectivities = dict((m, []) for m in experiments)
    residuals = dict((m, []) for m in experiments)
    slabs = dict((m, []) for m in experiments)
    def record_point():
        problem.chisq() # Force reflectivity recalculation
        for m in experiments:
            D = m.residuals()
            residuals[m].append(D+0)
            slabs_i = [L.thickness.value for L in m.sample[1:-1]]
            slabs[m].append(np.array(slabs_i))
            if m.ismagnetic:
                z, rho, irho, rhoM, thetaM = m.magnetic_step_profile()
                profiles[m].append((z+0, rho+0, irho+0, rhoM+0, thetaM+0))
            else:
                z, rho, irho = m.smooth_profile()
                profiles[m].append((z, rho, irho))
            Qs, Rs = m.reflectivity()
            reflectivities[m].append((Qs, Rs))
    record_point() # Put best at slot 0, no alignment
    origp_keys = list(origp.keys())
    indexlist = []
    for i in problem.labels():
        indexlist.append(origp_keys.index(i))
    for p in points:
        parr = p[indexlist]
        problem.setp(parr)
        record_point()
    # Turn residuals into arrays
    residuals = dict((k, np.asarray(v).T) for k, v in residuals.items())
    return profiles, slabs, Q, residuals, reflectivities

def calc_errors(problem, points, origp):
    """
    Align the sample profiles and compute the residual difference from the
    measured data for a set of points.

    The return value is arbitrary.  It is passed to the :func:`show_errors`
    plugin for the application.
    Returns *errs* for :func:`show_errors`.
    """
    original = problem.getp()
    try:
        ret = calc_errors_refl1d(problem, points, origp)
    except Exception:
        info = ["error calculating distribution on model",
                traceback.format_exc()]
        logging.error("\n".join(info))
        ret = None
    finally:
        problem.setp(original)
    return ret

def calc_errors_from_state(problem, state, origp, nshown=50, random=True):
    """
    Compute confidence regions for a problem from the
    Align the sample profiles and compute the residual difference from
    the measured data for a set of points returned from DREAM.

    *nshown* is the number of samples to include from the state.

    *random* is True if the samples are randomly selected, or False if
    the most recent samples should be used.  Use random if you have
    poor mixing (i.e., the parameters tend to stay fixed from generation
    to generation), but not random if your burn-in was too short, and
    you want to select from the end.

    Returns *errs* for :func:`show_errors`.
    """
    points, _logp = state.sample()
    if points.shape[0] < nshown:
        nshown = points.shape[0]
    # randomize the draw; skip the last point since state.keep_best() put
    # the best point at the end.
    if random:
        points = points[np.random.permutation(len(points) - 1)]
    return calc_errors(problem, points[-nshown:-1], origp)

#load SLD and RQ posterior bands. 
#To get a different contrast, change experiment[0] to experiment[1].
if 1:  # Loading errors is expensive; may not want to do so all the time.
    state = load_state(os.path.join(store, model[:-3]))
    origp = dict(zip(problem.labels(), problem.getp()))
    profiles, slabs, Q, residuals, RQ = calc_errors_from_state(experiment[1], state, origp, nshown=300) #profiles is a dictionary of length n, where n is the number of models in the multifitprob.

items = list(profiles) #to access the data within each model, convert the profiles to a list to list the items. this creates a dict_keys object.

#check SLD profiles (note they are the wrong way around)
show_profiles((profiles,slabs,Q,residuals), align=0, contours=[], npoints=50)
plt.show()

## get posterior bands for SLD ##

Zs_up = []
SLDs_up = []
Zs_down = []
SLDs_down = []

for k, j in enumerate(profiles[items[0]]): #Up
    Zs_up.append(profiles[items[0]][k][0])
    SLDs_up.append(profiles[items[0]][k][1])
for k, j in enumerate(profiles[items[1]]): #Down
    Zs_down.append(profiles[items[1]][k][0])
    SLDs_down.append(profiles[items[1]][k][1])

#rearrange SLD profile so that we have columns of distances and SLDs: z1, SLD1, z2, SLD2 ... zn, SLDn
#convert lists to dataframes

dfapp_up = []
dfapp_down = []
dfapp_n = []
dfapp_m = []
for i, j in enumerate(Zs_up):
    dataassign_up = (pd.Series(Zs_up[i]), pd.Series(SLDs_up[i]))
    dataassign_down = (pd.Series(Zs_down[i]), pd.Series(SLDs_down[i]))
    dataassign_n = (pd.Series(Zs_up[i]), ((dataassign_up[1] + dataassign_down[1]) / 2))
    dataassign_m = (pd.Series(Zs_up[i]), ((dataassign_up[1] - dataassign_down[1]) / 2))
    dfapp_up.append(pd.DataFrame(dataassign_up).T)
    dfapp_down.append(pd.DataFrame(dataassign_down).T)
    dfapp_n.append(pd.DataFrame(dataassign_n).T)
    dfapp_m.append(pd.DataFrame(dataassign_m).T)

#we want the sld profiles in gaps of 0.5 Å.
SLD_up = pd.concat(dfapp_up, axis=1, ignore_index=True)[::5]
SLD_down = pd.concat(dfapp_down, axis=1, ignore_index=True)[::5]
SLD_n = pd.concat(dfapp_n, axis=1, ignore_index=True)[::5]
SLD_m = pd.concat(dfapp_m, axis=1, ignore_index=True)[::5]

"""
The SLD profiles are generated the wrong way around for the defined models.
Therefore, we must inverted the SLD profiles.
"""
# now we need to correct the z values # 
# we use the SLD++ to find the interface between SiO2 and Si.
no2 = 3.467 #SLD of layer above Sub.
no1 = 2.072 #SLD of sub layer.
halfway = ((no2 - no1)/2) + no1 #calculates the SLD value that is halfway between the two layers.
for i in range(0, (300*2), 2):
    results = []
    newdistance_up = []
    for number in SLD_up.iloc[:,(i+1)].values: #Need to get each column of SLD 
        results.append(number - halfway) #take the variable halfway from each value in c and put this value in results.
    Nearinterface = results[-200:-1] #Refine the results list down to the interface
    UpperSLD = min([i for i in Nearinterface if i>0]) #Find the minimum posistive value from the nearinterface list 
    us = results.index(UpperSLD) # us is a variable that is the index value of where the value of upperSLD is in the list
    DeltaPoints_up = ((SLD_up.iloc[:,i].values[us] - SLD_up.iloc[:,i].values[us+1])**2)**(1/2) #the magnitude of the difference between the distance values
    DeltaSLD_up = SLD_up.iloc[:,i+1].values[us] - SLD_up.iloc[:,i+1].values[us+1] #the value of difference between the SLD values
    DiffSLDratio_up = (SLD_up.iloc[:,i+1].values[us] - halfway) / DeltaSLD_up #finds ratio of where interface lies between the two data points.
    USnew_up = DeltaPoints_up * DiffSLDratio_up #uses ratio to set the correct value of distance from interface for 1 point.
    Uppernd_up = USnew_up + DeltaPoints_up * us #first value in distance list is set to correct value and for loop used below used to find others.
    for g, h in enumerate(SLD_up.iloc[:,i].values): #for numbers in list x
        newdistance_up.append(Uppernd_up - g*DeltaPoints_up) #subtracts an increasing factor of 0.5 from top value in list.
    SLD_up.iloc[:,i] = newdistance_up
    SLD_down.iloc[:,i] = newdistance_up
    SLD_n.iloc[:,i] = newdistance_up
    SLD_m.iloc[:,i] = newdistance_up

#now lets save out the posterior of the SLDs:    
SLD_up.to_csv(os.path.join(os.getcwd(), 'ddod_SLD++.csv'), index=False)
#SLD_down.to_csv(os.path.join(os.getcwd(), 'ddod_SLD--.csv'), index=False)
#SLD_n.to_csv(os.path.join(os.getcwd(), 'ddod_SLDn.csv'), index=False)
#SLD_m.to_csv(os.path.join(os.getcwd(), 'ddod_SLDm.csv'), index=False)

## GET R(Q) VARIATION ###
objs = list(RQ)
Qs = []
Rs = []
for r, t in enumerate(RQ[objs[0]]):
    Qs.append(RQ[objs[0]][r][0])
    MixedRef = ((RQ[objs[0]][r][1]*0.5)+(0.5*RQ[objs[1]][r][1]))
    Rs.append(MixedRef)
Refdata = (pd.Series(Qs[0]), pd.Series(Rs[0]))
refdf = pd.DataFrame(Refdata).T
refdfapp = []
for i, j in enumerate(Qs):
    if not i == 0:
        refdataassign = pd.Series(Rs[i]) #only save out one Q column.
        refdfapp.append(pd.DataFrame(refdataassign))
refinit = pd.concat(refdfapp, axis=1, ignore_index=True)
reffinalist = [refdf, refinit]
reffinal = pd.concat(reffinalist, axis=1, ignore_index=True)
reffinal.to_csv(os.path.join(os.getcwd(), 'ddod_RQ.csv'), index=False)

raise Exception()