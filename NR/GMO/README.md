## Scripts to analyse the NR data collected with dodecane solutions containing GMO.

python modules requirements to completely recreate:<br>
refl1d, bumps, refnx, dynesty, jupyter, uravu

The three nested sampling scripts (for models 1, 2 & 3) are within the M1_NS, M2_NS and M3_NS directories.
These three scripts use MixedMagSlabs2.py<br>

The (refl1d-DREAM) fit of model 1 was conducted with P1-GMO.py <br>

The Load_posterior_bands.py script can be used to save out the reflectivity and SLD traces of the sampled posterior.<br>
This can be done within your refl1d conda environment from the GMO directory with the following command:<br>
$ refl1d Load_posterior_bands.py P1-GMO.py Fit_store<br>
An example of these bands is given in "ddod_RQ_300point.csv" & "ddod_SLD++_300point.csv"_.<br>

The notebook "P1_test_variables_for_gauss_and_corner" found in Fit_store directory can be used to generate a description of the posterior for each parameter.<br>

P1-GMO-median_sim.py was used to generate the median reflectivity and SLD profiles for the three solvent contrasts (these are found in the mediansim directory).
This was done with the following command: <br>
$ refl1d --simulate --noise=0.00000001 --store=mediansim

The notebook SLDfix - refl1d within the mediansim directory was used to correct the distances across the profile.
