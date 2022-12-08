# Work flow to compare MD output with NR data.

## Required software
* [python 3] (https://www.python.org/downloads/)
* [Pandas] (https://pandas.pydata.org/)
* [refnx] (https://refnx.readthedocs.io/en/latest/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)

## Splicing workflow

The notebook "MD\_NR\_Splice\_SLD" can be used to splice the SLD profile generated from the MD analysis (gmo\_\1\_water\_\0\_sld.csv) onto the SLD profile of the underlying substrate.
The spliced SLD profiles are then microsliced to create NR profiles which are compared to the original NR data.

## Convolution workflow

The notebook "MD\_NR\_convolve" loads the SLD profile generated from the MD analysis (gmo\_\1\_water\_\0\_sld.csv).
The MD SLD profiles are convolved with underlying substrate volume fraction profile, which is generated from the median parameter values from the refl1d fit (see files in NR/GMO/Fit_store).
These convolved SLD profiles are then added to the SLD profile of the underlying substrate. 
The complete SLD profiles are then microsliced to create NR profiles which are compared to the original NR data.
