# Creating LAMMPS datafiles from coordinates

## Required software

* [packmol](http://leandro.iqm.unicamp.br/m3g/packmol/home.shtml)
* [VMD](https://www.ks.uiuc.edu/Research/vmd/)
* [LAMMPS](https://www.lammps.org/)

## Instructions

The bash script `create_data_file.sh` creates a LAMMPS datafile from the .xyz files in the [common files folder][cff] and the parameter files in this folder.

1. It packs the chosen number of molecules with packmol using the options in the `.pack` file.
2. It then uses vmd to set up each atom type parameters acording to the `.tcl` files in the [common files folder][cff], to set the PBC size, and to write the lammpsdatafile.
3. It finally takes the pair, bond, angle, and dihedral parameters from the `lammpsparameters.txt` file and appends it to the LAMMPS datafile.
4. The LAMMPS input script used is included as `in.run`.

[cff]: ../../../0.common_files/
