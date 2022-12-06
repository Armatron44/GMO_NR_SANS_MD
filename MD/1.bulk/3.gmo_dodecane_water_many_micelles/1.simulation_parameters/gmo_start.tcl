#GMO all atom opls
set selc [atomselect top {name C}]
$selc set mass 12.011
$selc set charge -0.1200
$selc set element C
$selc set type C 
$selc set radius 1.4

set selca [atomselect top {name CA}]
$selca set mass 12.011
$selca set charge -0.1200
$selca set element C
$selca set type CA
$selca set radius 1.4

set selcd [atomselect top {name CD}]
$selcd set mass 12.011
$selcd set charge -0.1150
$selcd set element C
$selcd set type CD
$selcd set radius 1.4

set selce [atomselect top {name CE}]
$selce set mass 12.011
$selce set charge 0.7600
$selce set element C
$selce set type CE
$selce set radius 1.4

set selcoh [atomselect top {name COH}]
$selcoh set mass 12.011
$selcoh set charge 0.1450
$selcoh set element C
$selcoh set type COH
$selcoh set radius 1.4

set selcom [atomselect top {name COM}]
$selcom set mass 12.011
$selcom set charge 0.2050
$selcom set element C
$selcom set type COM
$selcom set radius 1.3

set selct [atomselect top {name CT}]
$selct set mass 12.011
$selct set charge -0.1800
$selct set element C
$selct set type CT 
$selct set radius 1.4

set selh [atomselect top {name H}]
$selh set mass 1.008
$selh set charge 0.0600
$selh set element H
$selh set type H 
$selh set radius 0.8

set selhce [atomselect top {name HCE}]
$selhce set mass 1.008
$selhce set charge 0.0600
$selhce set element H
$selhce set type HCE
$selhce set radius 0.8

set selhd [atomselect top {name HD}]
$selhd set mass 1.008
$selhd set charge 0.1150
$selhd set element H
$selhd set type HD
$selhd set radius 0.8

set selho [atomselect top {name HO}]
$selho set mass 1.008
$selho set charge 0.4180
$selho set element H
$selho set type HO
$selho set radius 0.8

set seloh [atomselect top {name OH}]
$seloh set mass 15.999
$seloh set charge -0.6830
$seloh set element O
$seloh set type OH
$seloh set radius 1.3

set selop [atomselect top {name OP}]
$selop set mass 15.999
$selop set charge -0.4300
$selop set element O
$selop set type OP
$selop set radius 1.2

set selos [atomselect top {name OS}]
$selos set mass 15.999
$selos set charge -0.3300
$selos set element O
$selos set type OS
$selos set radius 1.3

#dodecane all atom opls
set selcd [atomselect top {name DCD}]
$selcd set mass 12.011
$selcd set charge -0.1200
$selcd set element C
$selcd set type DCD
$selcd set radius 1.4

set selct [atomselect top {name DCT}]
$selct set mass 12.011
$selct set charge -0.1800
$selct set element C
$selct set type DCT 
$selct set radius 1.4

set selh [atomselect top {name DH}]
$selh set mass 1.008
$selh set charge 0.0600
$selh set element H
$selh set type DH 
$selh set radius 0.8

set selwo [atomselect top {name WO}]
$selwo set mass 15.999
$selwo set type WO
$selwo set charge -0.8340
$selwo set element O
$selwo set radius 1.2

set selwh [atomselect top {name WH}]
$selwh set mass 1.008
$selwh set type WH
$selwh set charge 0.4170
$selwh set element H
$selwh set radius 0.8

package require topotools
proc get_total_charge {{molid top}} { eval "vecadd [[atomselect $molid all] get charge]" }
mol bondsrecalc all
topo retypebonds
topo guessangles
topo guessdihedrals

puts ""
puts "Bond type names: "
puts [ topo bondtypenames ]
puts ""
puts "Angle type names: "
puts [ topo angletypenames ]
puts ""
puts "Dihedral type names: "
puts [ topo dihedraltypenames ]
puts ""
puts ""
puts [format "Number of bonds:          %s" [topo numbonds]         ]
puts [format "Number of bonds types:    %s" [topo numbondtypes]     ]
puts [format "Number of angles:         %s" [topo numangles]        ]
puts [format "Number of angles types:   %s" [topo numangletypes]    ]
puts [format "Number of dihedral:       %s" [topo numdihedrals]     ]
puts [format "Number of dihedral types: %s" [topo numdihedraltypes] ]
puts [format "Total charge:             %s" [get_total_charge]      ]
puts ""
puts "to write data file use the command:" 
puts "topo writelammpsdata data."
