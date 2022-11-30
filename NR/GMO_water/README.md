## Scripts to analyse the NR data collected with dodecane solutions containing GMO and water.

python modules requirements to completely recreate:<br>
refnx, dynesty, schwimmbad, jupyter, uravu

The two nested sampling scripts (for model 1 and 2) are named M1_NS.py and M2_NS.py.<br>
Both of these scripts use MixedMagSlabs2.py and either vfp_M1.py or vfp_M2.py.<br>
Note that the analysis is computationally expensive (taking ~ 1 day to complete on 64 cores of a HPC),<br>
so recreating on a standard computer may take a long time.

However, the nested sampling results and global objective of these scripts are also available.<br>
These are named M1_GO.pkl, M1_NS.pkl, M2_GO.pkl and M2_NS.pkl.<br>
These files are interrogated in the scripts called Load_M1_NS.py and Load_M2_NS.py.

The PTMCMC fit of model 2 was conducted with M2_PTMCMC.py<br>
The output of this file is given in M2_FIT_GO.pkl and can be interpreted by Load_M2_PTMCMC.py.<br>

The NR_data_present_M2 notebook can be used to recreate Fig 10.<br>
The Calculate_Areas_of_overlap notebook was used to calculate the fraction of area of Si contacting Fe.
