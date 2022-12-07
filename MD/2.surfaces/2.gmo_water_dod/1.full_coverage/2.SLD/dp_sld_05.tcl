#!/usr/local/bin/vmd -dispdev text
# run with:
# vmd -dispdev text -e this_script.tcl

# Written by Rui Apóstolo in 2021
# ruiapostolo@gmail.com

##############################################################################
#                            EDITABLE PARAMETERS:                            #
##############################################################################
# load trajectory file
mol new ../1.simulation_parameters/gmo_npt_2.lammpstrj type lammpstrj waitfor all autobonds off
#  bin width (Angstrom)
set resolution 0.5
# dens prof step
set fstep 1
# set start
set ts_start 0
# end of last range (relative to dump file start, first timestep is always 0)
set ts_stop [expr [molinfo top get numframes] - 1]


# list of atom types to loop through, and corresponding 
# using b_c for D-dod
set atom_sets {
    { surf_fe {name 1 3}              0.00009452   }
    { surf_o  {name 2 4}              0.000058034  }
    { gmo_c   {name 5 6 7 8 9 10 11}  0.0000664601 }
    { gmo_h   {name 15 16 17 18}     -0.0000373901 }
    { gmo_o   {name 19 20 21}         0.000058034  }
    { dod_c   {name 12 13}            0.0000664601 }
    { dod_h   {name 14}               0.000066714  }
    { wat_h   {name 22}              -0.0000373901 }
    { wat_o   {name 23}               0.000058034  }
    { wat_d   {name 22}               0.000066714  }
}

##############################################################################
#                       END OF EDITABLE PARAMETERS                           #
##############################################################################

# create half bin_width
set hbin [expr $resolution / 2.0]

# define functions

# print table to stdout
proc pptable {l1 l2} { foreach i1 $l1 i2 $l2 { puts " [format %6.2f $i1]\t[format %1.8f $i2]" }  }
# usage:
# pptable [lindex $denspolc 1] [lindex $denspolc 0]

# print table to file
proc pptableout {l1 l2 b1 f3} { foreach i1 $l1 i2 $l2 { puts $f3 " [format %6.2f [expr $i1 + $b1]]\t[format %1.8f $i2]" }  }
# usage:
# pptableout [lindex $denspolc 1] [lindex $denspolc 0] half_bin_width $fout
# pptableout bins densities half_bin_width file_to_write

# sum all elements in a 1D list
proc laddele L {expr [join $L +]+0} ; # adds all list elements
# usage:
# laddele [lindex $denspolc 0]

# list add (element-wise)
proc ladd {l1 l2} {
    set result {}
    foreach x $l1 y $l2 {
        if {$x == ""} {
            lappend result $y
        } elseif {$y == ""} {
            lappend result $x
        } else {
            lappend result [expr {$x + $y} ] 
        }
    }
    return $result
}
# usage:
# set result [ladd [lindex $densprof 0] [lindex $densprof 1]]

# multiply list by variable (element-wise)
proc lmul {l1 constant} {
    set result {}
    foreach x $l1 { lappend result [expr {$x * $constant} ] }
    return $result
}
# usage:
# set result [lmul [lindex $densprof 0] constant]

# calculate density profiles
package require density_profile


puts " "
puts " "
puts "Loading done, now calculating density profiles"


# average entire range
set dir "average_05"
file mkdir $dir
set result {}
puts "Calculating averages"
puts " "
foreach atomset $atom_sets {
    set name [lindex $atomset 0]
    set atoms [lindex $atomset 1]
    puts [format "%s " $name]

    # open file and calculate dp
    set fout1 [open $dir/$name.dat w+]
    set densprof [density_profile -rho number -selection $atoms -axis z -resolution $resolution -frame_from $ts_start -frame_to $ts_stop -frame_step $fstep -average 1]
    puts $fout1 "z   dens"
    pptableout [lindex $densprof 1] [lindex $densprof 0] $hbin $fout1
    close $fout1

    # open file and write dp times scattering length density
    set fout2 [open [format "%s/%s_sld.dat" $dir $name] w+ ]
    set dp_times_csl [lmul [lindex $densprof 0] [lindex $atomset 2]]
    puts $fout2 "z   SLD"
    pptableout [lindex $densprof 1] $dp_times_csl $hbin $fout2
    close $fout2
}

exit
