from __future__ import print_function
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astromodels import Model, PointSource

from threeML.classicMLE.goodness_of_fit import GoodnessOfFit
from threeML.classicMLE.joint_likelihood import JointLikelihood
from threeML.data_list import DataList
from threeML.plugin_prototype import PluginPrototype
from threeML.utils.statistics.likelihood_functions import half_chi2
from threeML.utils.statistics.likelihood_functions import (
    poisson_log_likelihood_ideal_bkg,
)
from threeML.exceptions.custom_exceptions import custom_warnings

__instrument_name = "n.a."


class perSourceLike(PluginPrototype):
    def __init__(self, name, likelihood_model, data_list, source_names=None, quiet=False):
        """
        likelihood_model : Model object containing multiple point sources (likelihood_model.point_sources)
        lat_plugin : Array of FermiLATLike plugins

        """


        if not quiet:
            print('\nInitializing perSourceLike plugin')

        nuisance_parameters = {}

        #initialize inheritance
        super(perSourceLike, self).__init__(name,nuisance_parameters)
        
        #self._x = np.array(x,ndmin = 1)
        #self._y = np.array(y,ndmin = 1)

        self._data_list = data_list
        self._likelihood_model = likelihood_model


        self._assign_source_names(source_names)
        self.set_model(likelihood_model)
        

    def _assign_source_names(self, source_names):

        if source_names is None or len(source_names) < 1 :
            assert ( 'perSourceLike expects an array of source names which correspond with respective ROI data' ) 
        
        for source in source_names:
            assert source in self._data_list.keys(), ( 'Source %s is not contained as a plugin in the data list object' )
            assert source in self._likelihood_model.point_sources, ('Source %s is not a point source in the likelihood model ' % source)

        self._source_names = source_names

    @property
    def x(self):
        return self._x
    @property
    def y(self): # 'like' --> joint likelihood analysis. Else: Bayesian (Multinest) analysis.

        return self._y
    @property
    def source_names(self):
        return self._source_names
    @property
    def data_list(self):
        return self._data_list



    def _get_total_expectation(self):

        #redundancies
        assert( self._source_names is not None ), 'source names are undefined'
        assert( self._source_names in self._likelihood_model.point_sources ), 'A point source in self-defined source names is not present in the likelihood model'

        #expectation = self._likelihood_model.point_sources[self._source_names](self._x)=

        #TODO: figure out what to do here in reference to the expectation from FermiLatLike



    def set_model(self, model):

        #assign individual point sources in the model for the data set (uses modified FermiLatLike)
        for i,j in zip(self._data_list.values(), self._source_names): 
            i.set_model(model,j)



    def fit(self, minimizer = 'minuit', verbose = False):

        print('Fitting ... ' )
        
        self._joint_like = JointLikelihood(self._likelihood_model,self._data_list,verbose=verbose)

        self._joint_like.set_minimizer(minimizer)
        
        self._joint_like.fit()

        return self._joint_like


    def plot( flux_unit = 'erg2/(cm2 s keV)', fit_cmap = 'viridis', contour_cmap = 'viridis', contour_style_kwargs = dict(alpha=0.1), energy_unit = 'MeV', ene_min = 65, ene_max = 100000 ):

        return plot_spectra(self._joint_like.results, flux_unit, fit_cmap, contour_cmap, contour_style_kwargs, energy_unit, ene_min, ene_max)



    #def goodness_of_fit()
    #def get_number_of_data_points()

        
    def get_log_like():
        return None
    def inner_fit():
        return get_log_like()

