source ../../../../0.common_files/dodecane.tcl
source ../../../../0.common_files/water.tcl
source ../../../../0.common_files/gmo.tcl
package require pbctools
play write_data.tcl
topo writelammpsdata data.bulk_gmo_water_dod
exit
