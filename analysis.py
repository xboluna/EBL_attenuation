
#########
#Save plots or display them?
savePlots = True

if(savePlots==True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("Plots will be saved")
else:
    import matplotlib.pyplot as plt

##########

import pdb

#parses argument as GRB to investigate
import argparse
parser = argparse.ArgumentParser(description='run analysis on a specified GRB')
parser.add_argument('grbs',metavar='GRB id',type=str,nargs='+',default='080916009',help='format e.g. 080916009')
args = parser.parse_args()

import os
import numpy as np
import glob
import astropy.units as u 

import warnings
warnings.filterwarnings("ignore")


#import local modifications to threeML
if (True):
    import sys
    sys.path.insert(0,'threeML_repo')

from threeML import *



catalog = pd.read_csv('EBL_candidates/selectedGRBs.csv')
owd = os.path.abspath(os.curdir)

def doLAT(OUTFILE,RA,DEC,ebl_model,TSTARTS,TSTOPS,ROI=8.0,ZMAX=100,EMIN=100,EMAX=100000,IRF='p8_transient010e', data_path='./'):
    '''
    This is a simple wrapper of the doTimeResolvedLike of gtburst
    TSTARTS,TSTOPS can be arrays if you want to run multiple intervals
    '''
    analysis_dir = 'analysis_%s_%s_%s' % (EMIN,EMAX,ebl_model)
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


def get_lat_like(t0, t1, ft2File, TRIGGER_ID,fermi_dir='.'):
    '''This is an helper funtion to retrieve the LAT data files saved by the doLAT step '''
    directory= '%s/interval%s-%s/' % (fermi_dir, t0, t1)
    print(directory)
    print(os.path.abspath(directory))

    eventFile = glob.glob("%s/*_filt.fit" % directory)[0]
    expomap = glob.glob("%s/*_filt_expomap.fit" % directory)[0] 
    ltcube = glob.glob("%s/*_filt_ltcube.fit" % directory)[0]

    #FermiLatLike on 187 in plugins dir
    return FermiLATLike("bn%s"%TRIGGER_ID, eventFile, ft2File, ltcube, 'unbinned', expomap)

# -------------------------------------------------------------- #
# Spectral functions
# -------------------------------------------------------------- #

def setup_powerlaw_model(src_name,index,ebl_model='powerlaw',REDSHIFT=0):    #default if REDSHIFT parameter not given
    powerlaw = Powerlaw()
    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0e-20, upper_bound=1e-10)
    powerlaw.piv     = 5.0e+5
    powerlaw.index   = index
    if REDSHIFT>0:
        powerlaw.index.free = False

        ebl = EBLattenuation()

        ebl.set_ebl_model(ebl_model)
        ebl.fit.prior = Uniform_prior(lower_bound = 0.0, upper_bound=1.0)
        
        spectrumEBL = powerlaw * ebl
        spectrumEBL.redshift_2 = REDSHIFT * u.dimensionless_unscaled
        spectrumEBL.fit = Uniform_prior(lower_bound = 0.0, upper_bound = 1.0) * u.dimensionless_unscaled
        source = PointSource(src_name, RA, DEC, spectral_shape=spectrumEBL)
        
    else:
        source = PointSource(src_name, RA, DEC, spectral_shape=powerlaw)
    return Model(source)

def setup_exponential_model(src_name):
    spectrum      = Cutoff_powerlaw()
    spectrum.piv  = 1000.0
    spectrum.xc   = 5000.0

    spectrum.K.prior   = Log_uniform_prior(lower_bound=1E-6, upper_bound=1E3)
    spectrum.xc.prior  = Log_normal(mu=np.log10(5000), sigma=np.log10(5000))
    spectrum.xc.bounds = (None, None)
    spectrum.index.prior = Truncated_gaussian(lower_bound=-5, upper_bound=5.0,
                                                mu=1.5, sigma=0.5)
    source = PointSource(src_name, RA, DEC, spectral_shape=spectrum)
    return Model(source)
# ------------------------------------------------------------------------------ #

def do_LAT_analysis(tstart,tstop,emin,emax,TRIGGER_ID,ebl_model='powerlaw',index=-2.0,REDSHIFT=0,irf='p8_transient010e'):
    analysis_dir = doLAT('%s' % TRIGGER_ID, RA, DEC, ebl_model, TSTARTS=[tstart], TSTOPS=[tstop],
                ROI=5.0, ZMAX=105, EMIN=emin, EMAX=emax,
                IRF=irf, data_path=LAT_DATA_PATH)
    like = True # If Like is true: joint likelihood analysis. Else: Bayesian (Multinest) analysis.

    if REDSHIFT is not 0: 
        model = setup_powerlaw_model('bn%s_%s'%(TRIGGER_ID,ebl_model),index,ebl_model,REDSHIFT)
    else:
        model = setup_powerlaw_model('bn%s'%TRIGGER_ID,index)
    model.display(complete=True)

    pdb.set_trace()
    lat_plugin = get_lat_like(tstart, tstop, FT2, TRIGGER_ID)

    if like:
        jl = JointLikelihood(model, DataList(lat_plugin))
        jl.fit()
        #plot_spectra(jl.results, flux_unit='erg2/(cm2 s keV)', energy_unit='MeV', ene_min=10, ene_max=10e+4)

        os.chdir('..') #replaces you to top of the cwd
        return jl
    else:
        bayes = BayesianAnalysis(model, DataList(lat_plugin))
        bayes.likelihood_model.LAT_GalacticTemplate_Value.set_uninformative_prior(Uniform_prior)
        bayes.likelihood_model.LAT_IsotropicTemplate_Normalization.set_uninformative_prior(Uniform_prior)
        bayes.set_sampler("multinest")
        bayes.sampler.setup(n_live_points=400)
        bayes.sample()
        bayes.restore_median_fit()
        bayes.results.corner_plot()
        #  display_spectrum_model_counts(bayes, min_rate=20)
        #plot_spectra(bayes.results, flux_unit='erg2/(cm2 s keV)', energy_unit='MeV',
        #             ene_min=10, ene_max=10e+4)

        os.chdir('..') #replaces you to top of the cwd
        return bayes


# ------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------ #

#takes: trigger ID
#returns: trigger's RA and DEC from catalog
#exception: if trigger not found, print and exit
def findGRB(grb_name):
    entry = catalog[ catalog['GRBNAME'] == int(i) ]
    print('found %s matches'%(len(entry)))
    try:
        return entry.iloc[0]['RA'],entry.iloc[0]['DEC'],entry.iloc[0]['REDSHIFT']
    except IndexError:
        print('grb %i not found'%grb_name)
        exit()

#container to run analysis generally
#takes: trigger information - RA, DEC, REDSHIFT
def runAnalysis(TRIGGER_ID,RA,DEC,REDSHIFT):
    analysis=[]
    tstart = 0.0
    tstop  = 600.0
    fig, ax = plt.subplots()

    print('--------------- Running first fit ')
    emin, emax = 65, 1000    # These are MeV
    analysis.append(do_LAT_analysis(tstart, tstop, emin,emax,TRIGGER_ID))

    print('--------------- Running second fit ')
    emin, emax = 65, 100000  # These are MeV
    analysis.append(do_LAT_analysis(tstart, tstop, emin, emax,TRIGGER_ID))
    #pulls photon index of first fit for use in EBL model
    bayesIndex = getattr(analysis[0].likelihood_model,'bn%s'%(TRIGGER_ID)).spectrum.main.Powerlaw.index.value
    
    
    for i in ['dominguez']:#,'finke','gilmore','franceschini','kneiske']:

        #only if you want separate images for each model
        pwlAnalysis = list(analysis)

        print('--------------- Running ebl attenuation model %s with photon index %s'%(i,bayesIndex))
        emin, emax = 65, 100000  # These are MeV
        pwlAnalysis.append(do_LAT_analysis(tstart, tstop, emin, emax, TRIGGER_ID, ebl_model=i,index=bayesIndex, REDSHIFT=REDSHIFT))

        plt.ylabel(r"Flux (erg$^{2}$ cm$^{-2}$ s$^{-1}$ TeV$^{-1}$)")
        plt.grid(True)

        plot_spectra(*[a.results for a in pwlAnalysis[::1]], flux_unit="erg2/(cm2 s keV)", fit_cmap='viridis',
                 contour_cmap='viridis', contour_style_kwargs=dict(alpha=0.1),
                 energy_unit='MeV', ene_min=emin, ene_max=emax
                 );

        

        if savePlots == True:
            plt.savefig('%s_%s'%(TRIGGER_ID,i)+'.png')
        else:
            plt.show()


    return;


# ------------------------------------------------------------------------------- #


if __name__ == "__main__":
    
    for i in args.grbs:
        print("####################################")
        print("analyzing %s"%i)
        print("####################################")
        TRIGGER_ID = i
        RA, DEC, REDSHIFT = findGRB(i) #scans catalog for given trigger, returns RA and DEC 
        print("TRIGGER_ID: %s RA: %s   DEC: %s   RS: %s"%(TRIGGER_ID,RA,DEC,REDSHIFT))
        GBM_DATA_PATH = './GRB%s' % TRIGGER_ID
        LAT_DATA_PATH = os.path.expandvars('${HOME}/FermiData') # This has to point where the gtburst data directory points.
        FT2 = LAT_DATA_PATH + '/bn%s/gll_ft2_tr_bn%s_v00.fit' % (TRIGGER_ID, TRIGGER_ID)

        like=False
        os.system('mkdir -p %s' % GBM_DATA_PATH)
        os.chdir(GBM_DATA_PATH)

        runAnalysis(TRIGGER_ID,RA,DEC,REDSHIFT)
        print('analysis for %s completed'%i)
        os.chdir(owd)

    print('analyses complete')
    #end main


