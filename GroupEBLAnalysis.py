
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')


from threeML import *


class GroupEBLAnalysis:
    def __init__(self, trigger_ids, dataframe=None, csv_path=None, datapath = './multipleGRBfit'):
        """
        Requires list of GRB names
        """
        
        #Setting catalog 
        assert dataframe is not None or csv_path is not None, ('Requires a dataframe catalog or the path to a csv catalog')
        self._catalog = None
        if csv_path is not None:
            self.read_dataframe_from_csv(csv_path)
        if dataframe is not None:
            self.set_catalog(dataframe)


        #Setting grouped properties
        self.set_names( trigger_ids )

        self.LAT_DATA_PATH = os.path.expandvars('${HOME}/FermiData')
        
        self.emin,self.emax = 65,100000 #MeV
        
        self.tstart = 0.0
        #tstop is pulled from catalog (TL100) and passed to doLat 

        #Begin working on model
        self.populate_model()

        #expect .fit() and .plot() to be called separately


    @property
    def get_fit_results(self): 
        return self.JOINT_FIT._analysis_results.get_data_frame() #.iloc[1,0]


    def populate_model(self):
        
        lat_plugins = []
        source_models = []

        for i in range(len(self.DATA['GRBNAME'])):

            name = self.DATA.iloc[i]['GRBNAME']
            ra = self.DATA.iloc[i]['RA']
            dec = self.DATA.iloc[i]['DEC']
            rs = self.DATA.iloc[i]['REDSHIFT']
            #Time interval called from TL100 estimated emission duration 
            tstop = self.DATA.iloc[i]['TL100']

            FT2 = self.LAT_DATA_PATH + '/%s/gll_ft2_tr_%s_v00.fit'%(name,name)
            
            self.DATA.at[self.DATA['GRBNAME'] == name,'doLAT'] = self.doLAT('%s' % name, ra, dec, [self.tstart], [tstop], data_path=self.LAT_DATA_PATH)
 
            
            self.DATA.at[self.DATA['GRBNAME'] == name,'lat_plugin'] = self.get_lat_like(self.tstart,tstop,FT2,name)
            self.DATA.at[self.DATA['GRBNAME'] == name,'source_model'] = self.setup_powerlaw_model('%s'%name, -2.0, ra, dec, REDSHIFT = rs)

        self.MODEL = Model(*self.DATA['source_model'].tolist())
        self.MODEL.display(complete=True)

        self.DATALIST = DataList(*self.DATA['lat_plugin'].tolist())

        #Verify models, datalist and sources are copacetic with one another
        for source in self.DATA['GRBNAME']: 
            assert (source) in self.MODEL.point_sources.keys(), ('Sources %s not a point source in the likelihood model '%source)
            assert (source) in self.DATALIST.keys(), ('Source %s does not have a plugin in the data list object'%source)

        #set FermiLatLike plugins' models
        for i in self.DATALIST.values():
            i.set_model(self.MODEL)

        print('Models created and linked for LAT plugins')



    def fit(self, minimizer = 'minuit', verbose = False):

        self.link_parameters()

        self.JOINT_FIT = JointLikelihood(self.MODEL,self.DATALIST,verbose=verbose)
        self.JOINT_FIT.set_minimizer(minimizer)
        self.JOINT_FIT.fit()

        return self.JOINT_FIT

    #param type in following two functions not yet implemented
    def link_parameters(self,param = 'fit'):

        fit_params = []
        for i in self.DATA['GRBNAME'].tolist():
            fit_params.append(
                    getattr(self.MODEL,i).spectrum.main.composite.fit_2)

        self.MODEL.link(*fit_params)

    def plot_param_profile(self, source_name,param = 'fit',steps=100,xmin=0,xmax=2):
        
        #---> is it possible to plot the contour of a linked parameter?
        a,b,cc,fig = self.JOINT_FIT.get_contours('%s.spectrum.main.composite.fit_2'%source_name, xmin, xmax,steps,param_2=None)
        
        result = self.JOINT_FIT._analysis_results.get_data_frame().iloc[1,0]
        fig.suptitle('fit = %s'%result)
        
        return fig


    def plot(self, flux_unit = 'erg2/(cm2 s keV)', fit_cmap = 'viridis', contour_cmap = 'viridis', contour_style_kwargs = dict(alpha=0.1), energy_unit = 'MeV', ene_min = 65, ene_max = 100000 ):

        return plot_spectra(self.JOINT_FIT.results, flux_unit, fit_cmap, contour_cmap, contour_style_kwargs, energy_unit, self.emin, self.emax)


    #helper fcns brought from multGRBfit
    def get_lat_like(self,t0, t1, ft2File, TRIGGER_ID,fermi_dir='.'):
        '''This is an helper funtion to retrieve the LAT data files saved by the doLAT step '''
        directory= '%s/interval%s-%s/' % (fermi_dir, t0, t1)
        print(directory)
        print(os.path.abspath(directory))

        eventFile = glob.glob(directory + "/*%s_*_filt.fit" % TRIGGER_ID)[0]
        expomap = glob.glob(directory + "/*%s_*_filt_expomap.fit" % TRIGGER_ID)[0] 
        ltcube = glob.glob(directory + "/*%s_*_filt_ltcube.fit" % TRIGGER_ID)[0] 
        return FermiLATLike(TRIGGER_ID, eventFile, ft2File, ltcube, 'unbinned', expomap, source_name = TRIGGER_ID)

    def setup_powerlaw_model(self,src_name,index,RA,DEC,REDSHIFT=0):
        powerlaw = Powerlaw()
        powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
        powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0e-20, upper_bound=1e-10)
        powerlaw.piv     = 5.0e+5
        powerlaw.index   = index
        if REDSHIFT>0:
            powerlaw.index.free = False

            ebl = EBLattenuation()

            ebl.set_ebl_model('kneiske')
            ebl.fit.prior = Uniform_prior(lower_bound = 0.0, upper_bound=2.0)
            
            spectrumEBL = powerlaw * ebl
            spectrumEBL.redshift_2 = REDSHIFT * u.dimensionless_unscaled
            
            source = PointSource(src_name, RA, DEC, spectral_shape=spectrumEBL)
            
        else:
            source = PointSource(src_name, RA, DEC, spectral_shape=powerlaw)
        return source

    def doLAT(self,OUTFILE,RA,DEC,TSTARTS,TSTOPS,ROI=5.0,ZMAX=105,EMIN=65,EMAX=100000,IRF='p8_transient010e', data_path='./'):
        '''
        This is a simple wrapper of the doTimeResolvedLike of gtburst
        TSTARTS,TSTOPS can be arrays if you want to run multiple intervals
        '''
        analysis_dir = '%s_analysis_%s-%s' % (OUTFILE,EMIN,EMAX) 
        os.system('mkdir -p %s' % analysis_dir)
        os.chdir(analysis_dir)
        exe='$CONDA_PREFIX/lib/python2.7/site-packages/fermitools/GtBurst/scripts/doTimeResolvedLike.py'
        #exe='doTimeResolvedLike.py'
        args={}
        args['outfile'] = OUTFILE
        args['ra']      = RA
        args['dec']     = DEC
        args['roi']     = ROI
        TSTARTS_str     = ''
        TSTOPS_str      = ''
        for t0,t1 in zip(TSTARTS,TSTOPS):
            TSTARTS_str+='%s, ' % t0
            TSTOPS_str+='%s, ' % t1
        TSTARTS_str=TSTARTS_str[:-2]
        TSTOPS_str=TSTOPS_str[:-2]
        args['tstarts'] = "'%s'" % TSTARTS_str
        args['tstops']  = "'%s'" % TSTOPS_str
        args['zmax']    = ZMAX
        args['emin']    = EMIN
        args['emax']    = EMAX
        args['irf']     = IRF
        args['galactic_model']   = "'template (fixed norm.)'"
        args['particle_model']   = "'isotr template'"
        args['tsmin']            = 25
        args['strategy']         = 'time'
        args['thetamax']         = 180
        args['spectralfiles']    = 'yes'
        args['liketype']         = 'unbinned'
        args['optimizeposition'] = 'no'
        args['datarepository']   = data_path
        args['flemin']           = 100.
        args['flemax']           = 10000
        args['fgl_mode']         = 'fast'
        triggername              = OUTFILE
        for k,i in args.items():
            exe+=' --%s %s' % (k,i)
        exe+=' %s' % triggername
        print(exe)
        os.system(exe)
        return analysis_dir




    def set_catalog(self,dataframe):
        if self._catalog is not None:
            print('Replacing catalog with new')
        self._catalog = dataframe

    def read_dataframe_from_csv(self, path):
        if self._catalog is not None:
            print('Replacing catalog with new')
        self._catalog = pd.read_csv(path)

    def set_names(self, GRBs):
        assert type(GRBs) is list, ('Group EBL analysis requires a list of GRB names to be included, regardless of whether a dataframe is applied later.')
        
        data=[]
        for i in GRBs:    
            assert int(i) in self._catalog['GRBNAME'].tolist(), ('GRB %s not found in catalog'%i)
            pick = self._catalog[ self._catalog['GRBNAME'] == int(i) ]
            pick['GRBNAME'] = 'bn%s'%i
            data.append(pick[['GRBNAME', 'RA', 'DEC', 'REDSHIFT','TL100']])
            

        self.DATA = pd.concat(data)

    #helper functions
    


