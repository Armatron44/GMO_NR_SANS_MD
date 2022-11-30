from numpy import inf
from sasmodels.core import reparameterize
parameters = [
    # name, units, default, [min, max], type, description
    ["Surf_SLD", "", 1, [0, 1], "", "SLD of surfactant"],
    ["Water_GMO_ratio", "", 5, [0, 10], "", "Water_GMO_ratio"],
    ["water_SLD", "", 1, [-0.54, 6.37], "", "SLD of water"],
    ["eccentricity", "", 1, [0.333, 3], "volume", "polar/equatorial ratio"],
    ["Solv_vf", "", 0.1, [0, 1], "", "vol fraction of solvent"],
    ["volume", "", 1, [0, inf], "volume", "sphereoid total volume"],
]
translation = """
    Re = cbrt((volume/eccentricity)*(1/M_4PI_3))
    radius_polar = eccentricity*Re
    radius_equatorial = Re
    vol_ratio = Water_GMO_ratio*(0.0016/0.0333)
    vol_solv = volume*Solv_vf
    vol_water = (vol_ratio*(volume-vol_solv))/(vol_ratio+1)
    vol_GMO = volume-(vol_water+vol_solv)
    Surf_vf = vol_GMO/volume
    water_vf = vol_water/volume
    sld = Surf_vf*Surf_SLD + Solv_vf*sld_solvent + water_vf*water_SLD
    """
model_info = reparameterize('ellipsoid', parameters, translation, __file__)