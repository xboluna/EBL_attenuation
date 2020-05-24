
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

from threeML import *

catalog = pd.read_csv('EBL_candidates/selectedGRBs.csv')


def doLAT(OUTFILE,RA,DEC,TSTARTS,TSTOPS,ROI=8.0,ZMAX=100,EMIN=100,EMAX=100000,IRF='p8_transient010e', data_path='./'):
    '''
    This is a simple wrapper of the doTimeResolvedLike of gtburst
    TSTARTS,TSTOPS can be arrays if you want to run multiple intervals
    '''
    analysis_dir = 'analysis_%s_%s' % (EMIN,EMAX)
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


def get_lat_like(t0, t1, ft2File, fermi_dir='.'):
    '''This is an helper funtion to retrieve the LAT data files saved by the doLAT step '''
    directory= '%s/interval%s-%s/' % (fermi_dir, t0, t1)
    print(directory)
    print(os.path.abspath(directory))

    eventFile = glob.glob("%s/*_filt.fit" % directory)[0]
    expomap = glob.glob("%s/*_filt_expomap.fit" % directory)[0] 
    ltcube = glob.glob("%s/*_filt_ltcube.fit" % directory)[0] 
    return FermiLATLike("LAT", eventFile, ft2File, ltcube, 'unbinned', expomap)

# -------------------------------------------------------------- #
# Spectral functions
# -------------------------------------------------------------- #

def setup_powerlaw_model(src_name):
    powerlaw = Powerlaw()
    powerlaw.index.prior = Uniform_prior(lower_bound=-5.0, upper_bound=5.0)
    powerlaw.K.prior = Log_uniform_prior(lower_bound=1.0e-20, upper_bound=1e-10)
    powerlaw.piv     = 5.0e+5
    source = PointSource(src_name, RA, DEC, spectral_shape=powerlaw)
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

def do_LAT_analysis(tstart,tstop,emin,emax,irf='p8_transient010e'):
    analysis_dir = doLAT('%s' % TRIGGER_ID, RA, DEC, TSTARTS=[tstart], TSTOPS=[tstop],
                ROI=5.0, ZMAX=105, EMIN=emin, EMAX=emax,
                IRF=irf, data_path=LAT_DATA_PATH)
    like = False # If Like is true: joint likelihood analysis. Else: Bayesian (Multinest) analysis.
    model = setup_powerlaw_model('bn%s' % TRIGGER_ID)
    model.display(complete=True)

    lat_plugin = get_lat_like(tstart, tstop, FT2)

    if like:
        jl = JointLikelihood(model, DataList(lat_plugin))
        jl.fit()
        plot_spectra(jl.results, flux_unit='erg2/(cm2 s keV)', energy_unit='MeV',
                     ene_min=10, ene_max=10e+4)
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
        return bayes


# ------------------------------------------------------------------------------ #

def loadModels(TRIGGER_ID,RA,DEC,REDSHIFT):
    
    #define attenuated spectrum
    spectrum = Powerlaw()
    source1 = PointSource(TRIGGER_ID,ra=RA,dec=DEC,spectral_shape=spectrum)
    spectrum.piv = 1. * u.MeV
    spectrum.K = 1.e-11 / (u.MeV * u.cm**2 * u.2)
    spectrum.index = -2.2
    
    #define attenuated spectrum for Dominiguez
    ebl = EBLattenuation()
    spectrumEBL = spectrum*ebl
    source2 = PointSource(TRIGGER_ID, ra=RA, dec=DEC,spectral_hape=spectrumEBL)
    spectrumEBL.redshift_2 = REDSHIFT*u.dimensionless_unscaled

    #define attenuation for Gilmore

    return spectrum,spectrumEBL


# ------------------------------------------------------------------------------ #

#takes: trigger ID
#returns: trigger's RA and DEC from catalog
#exception: if trigger not found, print and exit
def findGRB(grb_name):
    entry = catalog[ catalog['GRBNAME'] == int(i) ]
    print('found %s matches'%(len(entry)))
    try:
        print('Ra %s :: Dec %s'%(entry.iloc[0]['RA'],entry.iloc[0]['DEC']))
        return entry.iloc[0]['RA'],entry.iloc[0]['DEC'],entry.iloc[0]['REDSHIFT']
    except IndexError:
        print('grb %i not found'%grb_name)
        exit()

def runAnalysis():
    analysis=[]
    tstart = 0.0
    tstop  = 600.0

    fig, ax = plt.subplots()
    emin,emax=65,1000 # These are MeV
    analysis.append(do_LAT_analysis(tstart, tstop, emin,emax))
    emin, emax = 65, 10000  # These are MeV
    analysis.append(do_LAT_analysis(tstart, tstop, emin, emax))

    plot_spectra(*[a.results for a in analysis[::1]], flux_unit="erg2/(cm2 s keV)", fit_cmap='viridis',
                 contour_cmap='viridis', contour_style_kwargs=dict(alpha=0.1),
                 energy_unit='MeV', ene_min=emin, ene_max=emax
                 );

    #plotting spectrum attenuation models
    spectrum, spectrumEBL = loadModels(TRIGGER_ID, RA, DEC, REDSHIFT)
    energies=np.logspace(emin,emax,100)*u.MeV
    """
    plt.loglog(energies,spectrum(energies),label="unattenuated")
    plt.loglog(energies,spectrumEBL(energies),label="Dominiguez attenuated")
    """

    if savePlots == True:
        plt.savefig(TRIGGER_ID+'.png')
    else:
        plt.show()


# ------------------------------------------------------------------------------- #


if __name__ == "__main__":
    

    for i in args.grbs:
        print("####################################")
        print("analyzing %s"%i)
        print("####################################")
        TRIGGER_ID = i
        RA, DEC, REDSHIFT = findGRB(i) #scans catalog for given trigger, returns RA and DEC 
        GBM_DATA_PATH = './GRB%s' % TRIGGER_ID
        LAT_DATA_PATH = os.path.expandvars('${HOME}/FermiData') # This has to point where the gtburst data directory points.
        FT2 = LAT_DATA_PATH + '/bn%s/gll_ft2_tr_bn%s_v00.fit' % (TRIGGER_ID, TRIGGER_ID)

        like=False
        os.system('mkdir -p %s' % GBM_DATA_PATH)
        os.chdir(GBM_DATA_PATH)

        runAnalysis()
        print('analysis for %s completed'%i)

    print('analyses complete')
    #end main


