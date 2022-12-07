#!/bin/bash

packmol < gmo_1.pack
packmol < gmo_2.pack
packmol < gmo_dod.pack
vmd gmo_dod.xyz -dispdev text -e vmd.tcl
cat lammpsparameters.txt >> data.bulk_gmo_water_dod
mergedatafiles merge.yaml
