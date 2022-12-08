# Small angle neutron scattering analysis

## Required software
* [python 3] (https://www.python.org/downloads/)
* [bumps] (https://bumps.readthedocs.io/en/latest/)
* [Refl1d] (https://refl1d.readthedocs.io/en/latest/)
* [corner] (https://corner.readthedocs.io/en/latest/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)
* [uravu] (https://uravu.readthedocs.io/en/latest/)
* [arviz] (https://www.arviz.org/en/latest/)
* [sasview] (https://github.com/SasView/sasview) - using the git download and not the sasview.exe

## Fitting data

The data were fit using bumps and the Fit_all_contrasts.py script - the output is stored in the Fit_store directory.
This uses ellip_GMO_water_solv.py script, which is a re-parameterisation of the ellipsoid model from sasview.
With the conda prompt in the Fit_store directory, use <br> 
$ bumps -p Generate-plots.py<br>
to load the chain, generate the parameter statistics & generate the corner plots.

To generate the median and mode simulated scattered intensity use <br> 
$ bumps Gen_simSANS_modesim.py Fit_all_contrasts_modesim.py Fit_store_modesim <br>

The notebook "Compare\_fit\_to\_data" is used to compare the data to the median and mode of the posteriors.
