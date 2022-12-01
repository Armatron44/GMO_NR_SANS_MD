#!/bin/bash

packmol < gmo_water_dod.pack
vmd gmo_water_dod.xyz -dispdev text -e vmd.tcl
cat lammpsparameters.txt >> data.bulk_gmo_water_dod
