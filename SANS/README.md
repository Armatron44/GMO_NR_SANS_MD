# Small angle neutron scattering analysis

## Required software
* [python 3] (https://www.python.org/downloads/)
* [Pandas] (https://pandas.pydata.org/)
* [bumps] (https://bumps.readthedocs.io/en/latest/)
* [Refl1d] (https://refl1d.readthedocs.io/en/latest/)
* [corner] (https://corner.readthedocs.io/en/latest/)
* [uravu] (https://uravu.readthedocs.io/en/latest/)
* [arviz] (https://www.arviz.org/en/latest/)
* [sasview] (https://github.com/SasView/sasview) - using the git download and not the sasview.exe

## Splicing workflow

The notebook "MD\_NR\_Splice\_SLD" can be used to splice the SLD profile generated from the MD analysis (gmo\_1\_water\_0\_sld.csv) onto the SLD profile of the underlying substrate.
The spliced SLD profiles are then microsliced to create NR profiles which are compared to the original NR data.

## Convolution workflow

The notebook "MD\_NR\_convolve" loads the SLD profile generated from the MD analysis (gmo\_1\_water\_0\_sld.csv).
The MD SLD profiles are convolved with underlying substrate volume fraction profile, which is generated from the median parameter values from the refl1d fit (see files in NR/GMO/Fit_store).
These convolved SLD profiles are then added to the SLD profile of the underlying substrate. 
The complete SLD profiles are then microsliced to create NR profiles which are compared to the original NR data.


With the conda prompt in the Fit_store directory, use $ bumps -p Generate-plots.py to load the chain, generate the parameter statistics & generate the corner plots.

To generate the median and mode simulated scattered intensity use $ bumps Gen_simSANS_modesim.py Fit_all_contrasts_modesim.py Fit_store_modesim.
