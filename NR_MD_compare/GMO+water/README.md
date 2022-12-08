# Work flow to compare MD output with NR data.

## Required software
* [python 3] (https://www.python.org/downloads/)
* [Pandas] (https://pandas.pydata.org/)
* [refnx] (https://refnx.readthedocs.io/en/latest/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)

## Convolution workflow

The notebook "MD\_NR\_convolve" loads the SLD profile generated from the MD analysis (gmo_1_water_1_sld.csv) and also loads the median parameters from the PTMCMC fit of Model 2.
The MD SLD profiles are convolved with underlying substrate volume fraction profile. These convolved SLD profiles are then added to the SLD profile of the underlying substrate. The complete SLD profiles are then microsliced to create NR profiles which are compared to the original NR data.
