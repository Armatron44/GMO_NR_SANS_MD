# Scripts to analyse the NR data collected with dodecane solutions containing GMO and water.

## Required software
* [python 3] (https://www.python.org/downloads/)
* [refnx] (https://refnx.readthedocs.io/en/latest/)
* [dynesty] (https://dynesty.readthedocs.io/en/stable/)
* [schwimmbad] (https://schwimmbad.readthedocs.io/en/latest/)
* [jupyter] (https://jupyter-notebook.readthedocs.io/en/stable/)
* [uravu] (https://uravu.readthedocs.io/en/latest/)
* [arviz] (https://www.arviz.org/en/latest/)

## Nested sampling of single layer (model 1) and double layer (model 2) models
The two nested sampling scripts (for model 1 and 2) are named M1_NS.py and M2_NS.py.
Both of these scripts use MixedMagSlabs2.py and either vfp_M1.py or vfp_M2.py.
Note that the analysis is computationally expensive (taking ~ 1 day to complete on 64 cores of a HPC node), so recreating on a standard computer may take a long time.

However, the nested sampling results and global objective of these scripts are also available. These are named M1_GO.pkl, M1_NS.pkl, M2_GO.pkl and M2_NS.pkl.
These files can be explored using the scripts Load_M1_NS.py and Load_M2_NS.py.

## PTMCMC fit of Model 2
The PTMCMC fit of model 2 was conducted with M2_PTMCMC.py
The output of this file is given in M2_FIT_GO.pkl and can be interpreted using Load_M2_PTMCMC.py.

The NR_data_present_M2 notebook can be used to recreate Fig 10.
The Calculate_Areas_of_overlap notebook was used to calculate the fraction of area of Si contacting Fe.
