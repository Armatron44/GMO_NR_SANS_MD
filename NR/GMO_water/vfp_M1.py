import numpy as np

from refnx.reflect import Structure, Component
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from scipy import special
import time
from methodtools import lru_cache #functools doesn't work for classmethods?

EPS = np.finfo(float).eps


class VFP(Component):
    """
    A hack of refnx's spline component.
    
    ### how does this work? ###
    
    In ReflectModel, the reflectivity function is used to calculate the generative for a given set of parameters.
    The generative is used when fitting a dataset, when carrying out posterior sampling & when estimating the model evidence.
    
    The reflectivity function requires a slab representation 
    (an array of slab (shape = 2+N, 4 where N is number of layers) parameters - thicknesses, roughnesses etc) 
    of the structure.
    
    This slab representation is returned by the structure.slab() method, 
    which uses the slab method of each component within a structure to return a concatenated array of slabs.
    
    This means, the VFP component needs a slab method which will return an array of microslabs to ReflectModel.
    In the slab method we use the __call__ method of the VFP component to do the calculation.
    
    Here, the __call__ method of Spline has been altered to calculate
    a new array of distances across the interface (zeds), and then calculate
    volume fractions profiles for all layers given the thickness and roughness parameters.
    SLDs for each layer are calculated using the SLD parameters and the
    calculated volume fractions.
    These are then added together to create an array of SLDs the same length
    as the zeds array.
    These sld values are then returned in the __call__ method, which feeds through
    to the slabs sld.
    No interpolation occurs in this class.
    The microslabs are 0.5 Å each, as defined by self.max_delta_z.
    
    Parameters
    ----------
    extent : float or Parameter
        Total length of volume fraction profiles
    SLDs : array of floats or Parameters    
        Values of SLDs
    thicknesses : tuple of floats or Parameters
        Thicknesses of layers - these are used to determine the width of the volume fraction profiles.
    roughnesses : tuple of floats or Parameters
        Roughnesses of layers - these are used to determine the width of the volume fraction profiles.
    contrast : string
        string used to select which SLDs are used to calculate the scattering length density profile.
    """

    def __init__(
        self,
        extent,
        SLDs,
        thicknesses,
        roughnesses,
        contrast,
    ):
        super().__init__() #inherit the Component class.
        self.SLDs = SLDs
        self.thicknesses = thicknesses #tuples for hashing...
        self.roughnesses = roughnesses
        self.contrast = contrast
        self.name = ""
        
        #hard code in some other values
        self.max_delta_z = 0.5
        
        #select contrasts to use
        if self.contrast == 'dd_d2o_up':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 2, 4, 7, 9]])
        elif self.contrast == 'dd_d2o_down':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 3, 5, 7, 9]])
        elif self.contrast == 'dd_h2o_up':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 2, 4, 6, 10]])
        elif self.contrast == 'dd_h2o_down':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 3, 5, 6, 10]])
        elif self.contrast == 'hd_d2o_up':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 2, 4, 8, 11]])
        elif self.contrast == 'hd_d2o_down':
            self.SLDs = np.array([SLDs[i] for i in [0, 1, 3, 5, 8, 11]])

        self.extent = possibly_create_parameter(
            extent, name="%s - VFP extent", units="Å"
        )
        
        #initialise the SLDs and zeds.
        self.vs = self.create_vs()
        
        self.dz = self.get_dzs(self.extent.value, self.max_delta_z)

    @lru_cache(maxsize=2)
    @classmethod
    def get_dzs(cls, ex, mxdz):
        """
        This function finds the thickness of each microslice, 
        while also finding the thickness of any larger slab.
        """
        def consecutive(indic, stepsize=1): #internal func for finding consecutive indices.
            """
            Splits indic into sub arrays where the difference between neighbouring units in indic is not 1.
            """
            return np.split(indic, np.where(np.diff(indic) != stepsize)[0]+1)
        
        delta_step = ex/(cls.knots-1) #gives the thickness of the microslabs (approx 0.5 Å).
        
        if not cls.ind[0].any(): #if there are no indices of where volume fraction is approx equal, all slab thicknesses = delta_step.
            dzs = np.ones(cls.knots)*delta_step
        
        else:
            indexs = consecutive(np.array(cls.ind)[0]) #list of n arrays (n is the number of zones where there is little difference)
        
            indexs_diffs = [j[-1]-j[0] for j in indexs] #find length of each zone and return in a list.
            indexs_starts = [j[0] for j in indexs] #where does each zone start?
            indexs_ends = [j[-1] for j in indexs] #where does each zone start?
        
            index_gaps = np.array([j - indexs_ends[i-1] for i, j in enumerate(indexs_starts) if i > 0])
        
            new_knots = (cls.knots) - (np.array(indexs_diffs).sum()+len(indexs)) #number of knots is now reduced.
            dzs = np.empty(int(new_knots)) #init an array for collecting dz values.
            new_indexs_starts = [indexs_starts[0] + index_gaps[:i].sum() for i, j in enumerate(indexs)]
        
            dzs = np.ones(new_knots)*delta_step #make all values delta step.
            if len(new_indexs_starts) > 1:
                for i, j in enumerate(new_indexs_starts): #find places where delta step needs to be altered.
                    dzs[j] = ((indexs_diffs[i]+1)*delta_step)+dzs[j-1]
            else:
                dzs[int(new_indexs_starts[0])] = ((indexs_diffs[0]+1)*delta_step)+dzs[int(new_indexs_starts[0]-1)] #alter dz in the one place required.
        return dzs
        
    def get_x_and_y_scatter(self):
        """ 
        Function that returns the middle z and SLD of each microslab.
        """
        y = np.array([float(i) for i in self.create_vs()])
        x = np.delete(np.array([float(i) for i in self.get_zeds(self.roughnesses, self.thicknesses)]), self.indices)
        return x, y
            
    @classmethod
    def get_zeds(cls, rough, thick):
        """
        Calculate an array of zeds for volume fraction calculations.
        """
        cls.knots = int(cls.ex/cls.mxdz) #number of points to calculate z at.
        zstart = -5 - 4 * rough[0] #where does z start?
        zend = 5 + np.cumsum(thick)[-1] + 4 * rough[-1] #where does z end?
        zed = np.linspace(float(zstart), float(zend), num=cls.knots) #now return the array.
        return zed
        
    @classmethod
    def get_erf(cls, layer_choice, loc, rough, thick):
        """
        Calculate 1-F_{i} for a given layer (defined by layer_choice)
        """
        erf = (1-np.array([0.5*(1 + special.erf((float(i)-loc[layer_choice])/
               float(rough[layer_choice])/np.sqrt(2))) for i in cls.get_zeds(rough, thick)]))
        return erf 
   
    @lru_cache(maxsize=2)
    @classmethod
    def get_vfs(cls, rough, thick, ex, mxdz):
        """
        This function creates the volume fraction profile for a given set of thicknesses, & roughnesses.
        It is a classmethod so that the result is shared between objects of the same class.
        This is useful as the different contrasts will share the same volume fraction profile
        for a given set of thickness and roughness parameters.
        As such, we can use the lru_cache capability to store the volume fraction profile so that
        it only needs calculating once per set of contrasts.
        """
        rough = np.array(rough)
        thick = np.array(thick)
        loc = np.cumsum(thick)
        #share the total length of vf profile & the microslab thickness across the class
        cls.ex = ex
        cls.mxdz = mxdz
        
        #hard code in the layers we want
        #create a vf array of length of N layers + 2.
        #the integer supplied in the brackets is the layer_choice. 0 is fronting.
        #follows Equation S9 in SI, although note that cls.get_erf(i, loc, rough, thick) = 1-F_{i}
        vfs = np.array([cls.get_erf(0, loc, rough, thick)*cls.get_erf(1, loc, rough, thick), #Si
                        (1-cls.get_erf(0, loc, rough, thick))*cls.get_erf(1, loc, rough, thick), #SiO2
                        (1-cls.get_erf(1, loc, rough, thick))*cls.get_erf(2, loc, rough, thick), #Fe
                        (1-cls.get_erf(2, loc, rough, thick))*cls.get_erf(3, loc, rough, thick), #FeOx
                        (1-cls.get_erf(2, loc, rough, thick))*(1-cls.get_erf(3, loc, rough, thick))*cls.get_erf(4, loc, rough, thick), #GMO
                        (1-cls.get_erf(2, loc, rough, thick))*(1-cls.get_erf(3, loc, rough, thick))*(1-cls.get_erf(4, loc, rough, thick))]) #Solv
        
        #There may be portions of a layer's vf profile that sits above 0 (at say 1) for an extended length.
        #Therefore, we can merge these microslabs into one to speed up calculations.
        cls.ind = np.nonzero((np.abs(np.diff(vfs[2])) < 1e-5) & (vfs[2][:-1] > 0.5)) #look for insignificant differences in Fe vf, when vf is > 0.5.
        reduced_vfs = np.delete(vfs, cls.ind, 1)
        return reduced_vfs, cls.ind
    
    def get_slds(self):
        """
        This function returns the total sld for a given contrast.
        
        The thicknesses and roughnesses are used to generate the volume fraction profile
        in combination with the total length of the volume fraction profile and the microslab width.
        
        After, the volume fraction profiles is multiplied by the sld values to create a SLD profile for a given contrast.
        """
        
        #get floats of parameters so hashing recognition works.
        thicks = tuple(np.array([float(i) for i in self.thicknesses]))
        roughs = tuple(np.array([float(i) for i in self.roughnesses]))
        
        volfracs, self.indices = self.get_vfs(roughs, thicks, self.extent.value, self.max_delta_z)
        sld_values = [float(i) for i in self.SLDs]
        
        #Equation S11 in the SI.
        self.sld_list = volfracs.T*sld_values
        tot_sld = np.sum(self.sld_list, 1)
        return tot_sld
        
    def create_vs(self): #creates parameters that are constrained to self.get_slds()
        slds_arr = self.get_slds()
        return slds_arr
        
    def __repr__(self):
        s = ("VFP({extent!r}, {SLDs!r}, {thicknesses!r}, {roughnesses!r}, {contrast!r}")
        return s.format(**self.__dict__)

    def __call__(self):
        """
        Here we get the slds from the volume fractions,
        then we find the average slds between consecutive points.
        """
        #recalculate slds.
        #caching is not shared between processes.
        self.vs = self.create_vs() #this returns the SLD profile for a given contrast.
        self.dz = self.get_dzs(self.extent.value, self.max_delta_z) #returns the thickness of each slab.
        
        average_slds = 0.5*np.diff(self.vs)+self.vs[:-1]
        return_slds = np.append(average_slds, self.vs[-1])
        return return_slds, self.dz


    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend([self.extent])
        return p
        
    def p_equivs(self):
        #as slds and dzs are not automatically returned as parameters
        #use this function to return the parameter values after fitting.
        dzs_par_list = Parameters(name='dzs')
        vs_par_list = Parameters(name='slds')
        for i, j in enumerate(self.dz):
            pdz = Parameter(value=j)
            dzs_par_list.append(pdz)
            pvs = Parameter(value=self.vs[i])
            vs_par_list.append(pvs)
        p = Parameters(name=self.name)
        p.extend([self.extent, dzs_par_list, vs_par_list])
        return p
    
    def vfs_for_display(self):
        """
        Useful function for displaying volume fractions. 
        Use in conjunction with first output of get_x_and_y_scatter to plot.
        """
        thicks = tuple(np.array([float(i) for i in self.thicknesses]))
        roughs = tuple(np.array([float(i) for i in self.roughnesses]))
        
        volfracs = self.get_vfs(roughs, thicks, self.extent.value, self.max_delta_z)[0]
        
        return volfracs
        
    def logp(self):
        return 0


    def slabs(self, structure=None):
        """
        Slab representation of the spline, as an array

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting this Component
        """
        if structure is None:
            raise ValueError("Spline.slabs() requires a valid Structure")
        
        slds, thicks = self()
        slabs = np.zeros((len(thicks), 5))
        slabs[:, 0] = thicks
        slabs[:, 1] = slds
        return slabs