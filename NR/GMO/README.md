# Scripts to analyse the NR data collected with dodecane solutions containing GMO.

## Required software
* [python 3] (https://www.python.org/downloads/)
* [Refl1d] (https://refl1d.readthedocs.io/en/latest/)
* [Pandas] (https://pandas.pydata.org/)
* [bumps] (https://bumps.readthedocs.io/en/latest/)
* [refnx] (https://refnx.readthedocs.io/en/latest/)
* [dynesty] (https://dynesty.readthedocs.io/en/stable/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)
* [uravu] (https://uravu.readthedocs.io/en/latest/)
* [arviz] (https://www.arviz.org/en/latest/)

## Nested sampling

The three nested sampling scripts (for models 1, 2 & 3) are within the M1_NS, M2_NS and M3_NS directories.
These three scripts use MixedMagSlabs2.py<br>

## MCMC fit of Model 1

The (refl1d-DREAM) fit of model 1 was conducted with P1-GMO.py <br>

The Load_posterior_bands.py script can be used to save out the reflectivity and SLD traces of the sampled posterior. This can be done within your refl1d conda environment from the GMO directory with the following command: <br>
$ refl1d Load_posterior_bands.py P1-GMO.py Fit_store<br>
An example of these bands is given in "ddod\_RQ\_300point.csv" & "ddod_SLD++\_300point.csv".

The notebook "P1_test_variables_for_gauss_and_corner" found in Fit_store directory can be used to generate a description of the posterior for each parameter.<br>

P1-GMO-median_sim.py was used to generate the median reflectivity and SLD profiles for the three solvent contrasts (these are found in the mediansim directory).
This was done with the following command: <br>
$ refl1d --simulate --noise=0.00000001 --store=mediansim P1-GMO-median_sim.py

The median SLD profiles for the up spin states are output by refl1d with the prefix -x-0-profile.dat, where x is either 1, 2 or 3 for the ddod, CMdod and hdod solvent contrasts.
The notebook 'SLDfix - refl1d' within the mediansim directory was used to correct the distances across these SLD profiles, and outputs the correct median SLD profiles in a txt file with the prefix of -corrected.txt
