"""
Components for magnetic slabs on a non-polarised beamline.
"""
__author__ = 'Sandy Armstrong'

import numpy as np

from refnx.reflect import Component, SLD, Slab
from refnx.reflect.structure import Scatterer
from refnx.reflect.reflect_model import gauss_legendre
from refnx.analysis import (Parameter, Parameters,
                            possibly_create_parameter)

class USM_Slab(Component):
    """
    Up-spin slab component has uniform SLD++ over its thickness,
    where sld++ = sld_n + sld_m.
    sld_m will be defined by (sld_n/ScattLen)*Magmom

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_n : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2) - this is the standard way of defining SLD.
    magmom : refnx.analysis.Parameter or float
        magnetic moment of this slab (bohr magneton)
    ScattLen : refnx.analysis.Parameter or float
        Scattering Length of material with magnetic and nuclear SLD (Å).
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(self, thick, sld_n, magmom, ScattLen, rough, name="", vfsolv=0, interface=None):
        super(USM_Slab, self).__init__(name=name) #what does this line do?
        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        self.magmom = possibly_create_parameter(magmom, name=f"{name} - magmom")
        self.ScattLen = possibly_create_parameter(ScattLen, name=f"{name} - ScattLen")
        self.MagSLDconstant = 0.00002699 #Å per bohr magneton. Derived value & not the 2.645 x 10-5 (errenous?) value.
        if isinstance(sld_n, Scatterer):
            self.sld_n = sld_n
        else:
            self.sld_n = SLD(sld_n)
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld_n.parameters)
        p.extend([self.magmom, self.ScattLen])
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"USM_Slab({self.thick!r}, {self.sld_n!r}, {self.magmom!r}, {self.ScattLen!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld_n)
        return np.array(
            [
                [
                    self.thick.value,
                    (sldc.real + (((sldc.real*1E-6)/self.ScattLen.value)*self.magmom.value*self.MagSLDconstant*1E6)),
                    sldc.imag,
                    # self.magmom.value,
                    # self.ScattLen.value,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )
        
class DSM_Slab(Component):
    """
    Down-spin slab component has uniform SLD-- over its thickness,
    where sld-- = sld_n - sld_m.
    sld_m will be defined by (sld_n/ScattLen)*Magmom

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_n : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2) - this is the standard way of defining SLD.
    magmom : refnx.analysis.Parameter or float
        magnetic moment of this slab (bohr magneton)
    ScattLen : refnx.analysis.Parameter or float
        Scattering Length of material with magnetic and nuclear SLD (Å).
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(self, thick, sld_n, magmom, ScattLen, rough, name="", vfsolv=0, interface=None):
        super(DSM_Slab, self).__init__(name=name) #what does this line do?
        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        self.magmom = possibly_create_parameter(magmom, name=f"{name} - magmom")
        self.ScattLen = possibly_create_parameter(ScattLen, name=f"{name} - ScattLen")
        self.MagSLDconstant = 0.00002699 #Å per bohr magneton.
        if isinstance(sld_n, Scatterer):
            self.sld_n = sld_n
        else:
            self.sld_n = SLD(sld_n)
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld_n.parameters)
        p.extend([self.magmom, self.ScattLen])
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"USM_Slab({self.thick!r}, {self.sld_n!r}, {self.magmom!r}, {self.ScattLen!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld_n)
        return np.array(
            [
                [
                    self.thick.value,
                    (sldc.real - (((sldc.real*1E-6)/self.ScattLen.value)*self.magmom.value*self.MagSLDconstant*1E6)),
                    sldc.imag,
                    # self.magmom.value,
                    # self.ScattLen.value,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )

class USM_nomagmom_Slab(Component):
    """
    Up-spin slab component has uniform SLD++ over its thickness,
    where sld++ = sld_n + sld_m.
    sld_m will be a parameter.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_n : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2) - this is the standard way of defining SLD.
    sld_m : refnx.analysis.Parameter or float
        magnetic scattering length density of this slab (Å-2 x 106)
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(self, thick, sld_n, sld_m, rough, name="", vfsolv=0, interface=None):
        super(USM_nomagmom_Slab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        self.sld_m = possibly_create_parameter(sld_m, name=f"{name} - sld_m")
        if isinstance(sld_n, Scatterer):
            self.sld_n = sld_n
        else:
            self.sld_n = SLD(sld_n)
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld_n.parameters)
        p.extend([self.sld_m])
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"USM_Slab({self.thick!r}, {self.sld_n!r}, {self.sld_m!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld_n)
        return np.array(
            [
                [
                    self.thick.value,
                    (sldc.real + self.sld_m.value),
                    sldc.imag,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )

class DSM_nomagmom_Slab(Component):
    """
    Down-spin slab component has uniform SLD-- over its thickness,
    where sld-- = sld_n - sld_m.
    sld_m will be a parameter.

    Parameters
    ----------
    thick : refnx.analysis.Parameter or float
        thickness of slab (Angstrom)
    sld_n : :class:`refnx.reflect.Scatterer`, complex, or float
        (complex) SLD of film (/1e-6 Angstrom**2) - this is the standard way of defining SLD.
    sld_m : refnx.analysis.Parameter or float
        magnetic scattering length density of this slab (Å-2 x 106)
    rough : refnx.analysis.Parameter or float
        roughness on top of this slab (Angstrom)
    name : str
        Name of this slab
    vfsolv : refnx.analysis.Parameter or float
        Volume fraction of solvent [0, 1]
    interface : {:class:`Interface`, None}, optional
        The type of interfacial roughness associated with the Slab.
        If `None`, then the default interfacial roughness is an Error
        function (also known as Gaussian roughness).
    """

    def __init__(self, thick, sld_n, sld_m, rough, name="", vfsolv=0, interface=None):
        super(DSM_nomagmom_Slab, self).__init__(name=name)
        self.thick = possibly_create_parameter(thick, name=f"{name} - thick")
        self.sld_m = possibly_create_parameter(sld_m, name=f"{name} - sld_m")
        if isinstance(sld_n, Scatterer):
            self.sld_n = sld_n
        else:
            self.sld_n = SLD(sld_n)
        self.rough = possibly_create_parameter(rough, name=f"{name} - rough")
        self.vfsolv = possibly_create_parameter(
            vfsolv, name=f"{name} - volfrac solvent", bounds=(0.0, 1.0)
        )

        p = Parameters(name=self.name)
        p.extend([self.thick])
        p.extend(self.sld_n.parameters)
        p.extend([self.sld_m])
        p.extend([self.rough, self.vfsolv])

        self._parameters = p
        self.interfaces = interface

    def __repr__(self):
        return (
            f"USM_Slab({self.thick!r}, {self.sld_n!r}, {self.sld_m!r}, {self.rough!r},"
            f" name={self.name!r}, vfsolv={self.vfsolv!r},"
            f" interface={self.interfaces!r})"
        )

    def __str__(self):
        # sld = repr(self.sld)
        #
        # s = 'Slab: {0}\n    thick = {1} Å, {2}, rough = {3} Å,
        #      \u03D5_solv = {4}'
        # t = s.format(self.name, self.thick.value, sld, self.rough.value,
        #              self.vfsolv.value)
        return str(self.parameters)

    @property
    def parameters(self):
        """
        :class:`refnx.analysis.Parameters` associated with this component

        """
        self._parameters.name = self.name
        return self._parameters

    def slabs(self, structure=None):
        """
        Slab representation of this component. See :class:`Component.slabs`
        """
        sldc = complex(self.sld_n)
        return np.array(
            [
                [
                    self.thick.value,
                    (sldc.real - self.sld_m.value),
                    sldc.imag,
                    self.rough.value,
                    self.vfsolv.value,
                ]
            ]
        )