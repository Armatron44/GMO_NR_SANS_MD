#!/bin/bash

packmol < gmo_dod.pack
vmd gmo_dod.xyz -dispdev text -e vmd.tcl
cat lammpsparameters.txt >> data.bulk_gmo_dod
