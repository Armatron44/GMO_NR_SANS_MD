#dodecane all atom opls
set seldcd [atomselect top {name DCD}]
$seldcd set mass 12.011
$seldcd set charge -0.1200
$seldcd set element C
$seldcd set type DCD
$seldcd set radius 1.4

set seldct [atomselect top {name DCT}]
$seldct set mass 12.011
$seldct set charge -0.1800
$seldct set element C
$seldct set type DCT 
$seldct set radius 1.4

set seldh [atomselect top {name DH}]
$seldh set mass 1.008
$seldh set charge 0.0600
$seldh set element H
$seldh set type DH 
$seldh set radius 0.8
