
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
import astropy.units as u

import warnings
warnings.filterwarnings('ignore')

from datetime import datetime


from threeML import *


class GroupEBLAnalysis:
    def __init__(self, trigger_ids, dataframe=None, csv_path=None, fit_type='ultranest',datapath = './multipleGRBfit', attenuation_model='kneiske', OUTDIR=None):
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


	if OUTDIR is not None:
		self.head_directory = OUTDIR
	else:
		if len(self.DATA['GRBNAME'].tolist()) < 5:
			self.head_directory = os.getcwd()+'/GroupEBLAnalysis_%s'%'_'.join(self.DATA['GRBNAME'].tolist())
		else:
			self.head_directory = os.getcwd()+'/GroupEBLAnalysis'
	os.system('mkdir -p %s' % self.head_directory)
	os.chdir(self.head_directory)
	self.DATA['GRBNAME'].to_csv('source_names.csv')

        self.LAT_DATA_PATH = os.path.expandvars('${HOME}/FermiData')

        assert fit_type == 'JointLikelihood' or fit_type == 'multinest' or fit_type == 'ultranest',('Fit type parameter supports JointLikelihood and BayesianAnalysis only')
        self.fit_type = fit_type
        
        self.emin,self.emax = 65,100000 #MeV
        
        self.saveTS = False

        self.set_attenuation_model(attenuation_model)
        
        #expect .populate_model(), then.fit() and .plot() to be called separately
        

    @property
    def get_fit_results(self):
        self.FIT.results.get_data_frame()
    
    def save_TS(self, b=True):
        assert (self.fit_type == 'JointLikelihood'), 'Computation of test statistic only available for JointLikelihood fit type.'
        self.saveTS = b

    def set_attenuation_model(self, model='kneiske'):
        self.attenuation_model = model

    def update_attenuation_model(self):
        for i in range(len(self.DATA['GRBNAME'])):
            name = self.DATA.iloc[i]['GRBNAME']
            ra = self.DATA.iloc[i]['RA']
            dec = self.DATA.iloc[i]['DEC']
            rs = self.DATA.iloc[i]['REDSHIFT']

            self.DATA.at[self.DATA['GRBNAME'] == name,'source_model'] = self.setup_powerlaw_model('%s'%name, -2.0, ra, dec, REDSHIFT = rs)

        self.MODEL = Model(*self.DATA['source_model'].tolist())
        self.MODEL.display()


    def populate_model(self, rollingSave = False):
        
        self.failed_sources = []

        for i in range(len(self.DATA['GRBNAME'])):
            
            name = self.DATA.iloc[i]['GRBNAME']
            ra = self.DATA.iloc[i]['RA']
            dec = self.DATA.iloc[i]['DEC']
            rs = self.DATA.iloc[i]['REDSHIFT']
            #Time interval called from TL100 estimated emission duration 
            tstart = self.DATA.iloc[i]['TL0']
            tstop = self.DATA.iloc[i]['TL1']

            FT2 = self.LAT_DATA_PATH + '/%s/gll_ft2_tr_%s_v00.fit'%(name,name)
             
            index = self.do_unattenuated_fit(name,ra,dec,tstart,tstop,FT2)

            
            if index is None:
                import pdb;pdb.set_trace()
                continue

            self.DATA.at[self.DATA['GRBNAME'] == name, 'prelim_index'] = index
            os.chdir(self.head_directory)

            self.DATA.at[self.DATA['GRBNAME'] == name,'doLAT'] = self.doLAT('%s' % name, ra, dec, [tstart], [tstop], data_path=self.LAT_DATA_PATH) 

            self.DATA.at[self.DATA['GRBNAME'] == name,'lat_plugin'] = self.get_lat_like(tstart,tstop,FT2,name)

            self.DATA.at[self.DATA['GRBNAME'] == name,'source_model'] = self.setup_powerlaw_model('%s'%name, index, ra, dec, REDSHIFT = rs)
            
            os.chdir(self.head_directory)

            if rollingSave is True:
                self.DATA.to_csv('DATA.csv')

        for i in self.failed_sources:
            self.DATA = self.DATA[self.DATA['GRBNAME'] != i]
        
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
        print('Successfully executed on sources: ' + ', '.join(self.DATA['GRBNAME'].tolist()))
        print('Excluded sources : ' + ', '.join(self.failed_sources))


    def do_unattenuated_fit(self,name,ra,dec,tstart,tstop,ft2,keep_fits = False):

        #Create two fitted powerlaws + a fitted spectrumEBL
        #Pwl1 : self.emin - self.emax/10
        #Pwl2 : full range
        #sEBL : draw pivot from Pwl2

        #First fit:
        emin,emax = self.emin, self.emax/100

        doLAT = self.doLAT('%s'%name,ra,dec,[tstart],[tstop],data_path=self.LAT_DATA_PATH,EMIN=emin,EMAX=emax)
        try:
             lat = self.get_lat_like(tstart,tstop,ft2,name)
             lat_plugin = DataList(lat)
        except:
                self.failed_sources.append(name)
                print('%s removed from fit to preserve execution' %name)
                return None
        powerlaw = Model(self.setup_powerlaw_model('%s'%name,-2.0,ra,dec))
        
        if self.fit_type == 'JointLikelihood':
            fit = JointLikelihood(powerlaw, lat_plugin)
            try:
                a, b = fit.fit()
                if self.saveTS is True:
                    #Clear source name for compute_TS
                    lat.clear_source_name()
                    fit._data_list = DataList(lat)
                    fit._assign_model_to_data(fit._likelihood_model)
                    TS = fit.compute_TS(source_name=name, alt_hyp_mlike_df = b)
                    print("Computed test statistics for %s"%name)
                    print(TS)
                    self.DATA.at[self.DATA['GRBNAME'] == name, 'TS'] = TS['TS'][0]
            except:
                return None
        elif self.fit_type == 'multinest':
            bayes = BayesianAnalysis(powerlaw,lat_plugin) 
            getattr(bayes.likelihood_model,'%s_GalacticTemplate_Value'%name).set_uninformative_prior(Uniform_prior) 
            getattr(bayes.likelihood_model,'%s_IsotropicTemplate_Normalization'%name).set_uninformative_prior(Uniform_prior)
            bayes.set_sampler("multinest")
            bayes.sampler.setup(n_live_points=400)
            bayes.sample(quiet = True)
            bayes.restore_median_fit()
            bayes.results.corner_plot()
            fit = bayes
        else: #is ultranest
            bayes = BayesianAnalysis(powerlaw,lat_plugin) 
            getattr(bayes.likelihood_model,'%s_GalacticTemplate_Value'%name).set_uninformative_prior(Uniform_prior) 
            getattr(bayes.likelihood_model,'%s_IsotropicTemplate_Normalization'%name).set_uninformative_prior(Uniform_prior)
            bayes.set_sampler("ultranest")
            bayes.sampler.setup()
            bayes.sample(quiet = True) 
            fit = bayes

        
        if keep_fits is True:
            self.DATA.at[self.DATA['GRBNAME'] == name,'powerlaw_%s_%s'%(emin,emax)] = fit
        
                
        index = fit.results.get_data_frame().iloc[1,0]
        
        return index
    
    
    def plot_TS(self, TS_line = None, RS_line = None):
        """
        Plots joint likelihood test statistic against redshift for all passed sources.
        """
        try:
            df = self.DATA[['GRBNAME','REDSHIFT','TS']]
        except:
            raise Exception('Either populate_model() is needed or saveTS was disabled.')

        fig, ax = plt.subplots()
        ax.scatter(df['REDSHIFT'],df['TS'])
        ax.set_xlabel('REDSHIFT')
        ax.set_ylabel('TEST STATISTIC')

        for i, txt in enumerate(df['GRBNAME']):
            ax.annotate(txt, (df['REDSHIFT'][i] , df['TS'][i] ) )

        ax.plot([RS_line,RS_line],[min(df['TS']),max(df['TS'])], label='redshift cut-off', color='r', linestyle='dashed')
        ax.plot([min(df['REDSHIFT']),max(df['REDSHIFT'])], [TS_line,TS_line], label='test significance cut-off', color='b', linestyle='dashed')

        return fig

    def do_fit(self, minimizer = 'minuit', verbose = False,quiet=True):

        self.link_parameters()

        print('Beginning fit at %s'%datetime.now().strftime('%H : %M : %S'))

        if self.fit_type == 'JointLikelihood':

            self.FIT = JointLikelihood(self.MODEL,self.DATALIST,verbose=verbose)

            for i in self.DATA['GRBNAME'].tolist():
                getattr(self.MODEL,'%s_GalacticTemplate_Value'%i).free = False

            self.FIT.set_minimizer(minimizer)
            self.FIT.fit()

        elif self.fit_type == 'multinest':

            bayes = BayesianAnalysis(self.MODEL,self.DATALIST)
            for name in self.DATA['GRBNAME'].tolist():
                getattr(bayes.likelihood_model,'%s_GalacticTemplate_Value'%name).set_uninformative_prior(Uniform_prior) 
                getattr(bayes.likelihood_model,'%s_IsotropicTemplate_Normalization'%name).set_uninformative_prior(Uniform_prior)
            bayes.set_sampler("multinest")
            bayes.sampler.setup(n_live_points=400)
            bayes.sample(quiet = quiet)
            bayes.restore_median_fit()
            bayes.results.corner_plot() 

            self.FIT = bayes

        else:
            bayes = BayesianAnalysis(self.MODEL,self.DATALIST)
            for name in self.DATA['GRBNAME'].tolist():
                getattr(bayes.likelihood_model,'%s_GalacticTemplate_Value'%name).free = False
		getattr(bayes.likelihood_model,'%s_GalacticTemplate_Value'%name).value = 1
                getattr(bayes.likelihood_model,'%s_IsotropicTemplate_Normalization'%name).set_uninformative_prior(Uniform_prior)
            bayes.set_sampler("ultranest")
            bayes.sampler.setup()
            bayes.sample(quiet = quiet) 
            fit = bayes

            self.FIT = bayes
        
        print('Sources fitted: %s'%(', '.join(self.DATA['GRBNAME'].tolist())))
        print('Sources excluded from fit: %s'%(', '.join(self.failed_sources)))
        return self.FIT

    
    def save_populated_model(self, directory = 'model'):
        os.system('mkdir -p %s'%directory)
        os.chdir(directory)
        self.MODEL.save('model.yaml',overwrite=True)

    def save_fit_results(self, filename = 'fitted_model.fits'):
        self.FIT.results.write_to(filename, overwrite=True)

    #param type in following two functions not yet implemented
    def link_parameters(self,param = 'attenuation'):

        source_names = self.DATA['GRBNAME'].tolist()

        for i in range(len(source_names)-1):
            param_1 = getattr(self.MODEL,source_names[i]).spectrum.main.composite.attenuation_2
            param_2 = getattr(self.MODEL,source_names[i+1]).spectrum.main.composite.attenuation_2
            self.MODEL.link(param_1,param_2)


    def plot(self, flux_unit = 'erg2/(cm2 s keV)', fit_cmap = 'viridis', contour_cmap = 'viridis', contour_style_kwargs = dict(alpha=0.1), energy_unit = 'MeV', ene_min = 65, ene_max = 100000 ):

        return plot_spectra(self.FIT.results, flux_unit=flux_unit, fit_cmap=fit_cmap, contour_cmap=contour_cmap, contour_style_kwargs=contour_style_kwargs, energy_unit=energy_unit, ene_min=self.emin, ene_max=self.emax)



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

            ebl.set_ebl_model(self.attenuation_model)
            ebl.attenuation.prior = Uniform_prior(lower_bound = 0.0, upper_bound=2.0)
            ebl.attenuation.fix = False
            
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
            data.append(pick[['GRBNAME', 'RA', 'DEC', 'REDSHIFT','TL0','TL1']])
            

        self.DATA = pd.concat(data)

    #helper functions
    


