# Scripts to analyse the NR data collected with neat dodecane solutions.

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

The nested sampling scripts for modelling the interface with and without the adventitious layer
are found within the "Adv_at_interface_NS" and the "bare interface_NS" folders.
These three scripts use MixedMagSlabs2.py<br>

## MCMC fit of Adv_at_interface

The (refl1d-DREAM) fit of the "Adv_at_interface" was conducted with P1-adv.py <br>

The Load_posterior_bands.py script can be used to save out the reflectivity and SLD traces of the sampled posterior.<br>
This can be done within your refl1d conda environment from the bare directory with the following command:<br>
$ refl1d Load_posterior_bands.py P1-adv.py Fit_store<br>
These bands have been saved out in ddod_SLD++.csv, hdod_SLD++.csv, ddod_RQ.csv and hdod_RQ.csv.

The Load_data.py script in the Fit_store directory can be used to generate a description of the posterior for each parameter,<br>
using the following commands:
$ cd your_directory/GMO_NR_SANS_MD/NR/Bare/Fit_store <br>
$ refl1d Load_data.py -p <br>

P1-Adv-Med-Sim.py was used to generate the median reflectivity and SLD profiles for the two solvent contrasts.
These profiles are stored in the Med-sim directory.
This was done with the following command: <br>
$ refl1d --simulate --noise=0.00000001 --store=Med-sim P1-Adv-Med-Sim.py

The median SLD profiles for the up spin states are output by refl1d with the prefix -x-0-profile.dat, where x is either 1 or 2 for the ddod or hdod solvent contrasts.
The notebook 'SLDfix - refl1d' within the mediansim directory of the GMO folder was used to correct the distances across these SLD profiles, which output the correct median SLD profiles in a txt file with the prefix of -corrected.txt
