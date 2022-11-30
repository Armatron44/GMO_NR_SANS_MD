#Water TIP3P
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
