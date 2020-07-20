
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
import copy

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
    from perSourceLike import *
from threeML import *


owd = os.path.abspath(os.curdir)
catalog = pd.read_csv('EBL_candidates/selectedGRBs.csv')


def doLAT(OUTFILE,RA,DEC,TSTARTS,TSTOPS,ROI=8.0,ZMAX=100,EMIN=100,EMAX=100000,IRF='p8_transient010e', data_path='./'):
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


def get_lat_like(t0, t1, ft2File, TRIGGER_ID,fermi_dir='.'):
    '''This is an helper funtion to retrieve the LAT data files saved by the doLAT step '''
    directory= '%s/interval%s-%s/' % (fermi_dir, t0, t1)
    print(directory)
    print(os.path.abspath(directory))

    
    eventFile = glob.glob(directory + "/*bn%s_*_filt.fit" % TRIGGER_ID)[0]
    expomap = glob.glob(directory + "/*bn%s_*_filt_expomap.fit" % TRIGGER_ID)[0] 
    ltcube = glob.glob(directory + "/*bn%s_*_filt_ltcube.fit" % TRIGGER_ID)[0] 
    return FermiLATLike("bn%s"%TRIGGER_ID, eventFile, ft2File, ltcube, 'unbinned', expomap, source_name = 'bn%s'%TRIGGER_ID)

# -------------------------------------------------------------- #
# Spectral functions
# -------------------------------------------------------------- #

def setup_powerlaw_model(src_name,index,RA,DEC,REDSHIFT=0):    #default if REDSHIFT parameter not given
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
    return source
# ------------------------------------------------------------------------------ #


#takes: trigger ID
#returns: trigger's RA and DEC from catalog
#exception: if trigger not found, print and exit
def findGRB(grb_name):
    entry = catalog[ catalog['GRBNAME'] == int(grb_name) ]
    print('found %s matches'%(len(entry)))
    try:
        return entry.iloc[0]['RA'],entry.iloc[0]['DEC'],entry.iloc[0]['REDSHIFT']
    except IndexError:
        print('%s not found'%grb_name)
        exit()

# ------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------ #


if __name__ == "__main__":
    
    #testing Bayesian analysis for multiple GRBs at once


    TRIGGER_ID=[]
    RA=[]
    DEC=[]
    REDSHIFT=[]
    sources=[]

    analysis=[]
    analysis_dir=[]
    lat_plugin = []


    GBM_DATA_PATH = './multipleGRBFit'
    os.system('mkdir -p %s' % GBM_DATA_PATH)
    os.chdir(GBM_DATA_PATH)

    irf='p8_transient010e'

    tstart = 0.0
    tstop  = 600.0
    fig, ax = plt.subplots()

    emin,emax = 65,100000 #MeV



    for i in range(0,len(args.grbs)): 
        print("####################################")
        print("retrieving information for %s"%args.grbs[i])
        print("####################################")


        TRIGGER_ID.append(args.grbs[i])
        #retrieves RA, DEC, REDSHIFT from catalogue
        RA[len(RA):], DEC[len(DEC):], REDSHIFT[len(REDSHIFT):] = tuple(zip(findGRB(args.grbs[i])))
        print("TRIGGER_ID: %s RA: %s   DEC: %s   RS: %s"%(TRIGGER_ID[i],RA[i],DEC[i],REDSHIFT[i]))


        LAT_DATA_PATH = os.path.expandvars('${HOME}/FermiData') # This has to point where the gtburst data directory points.
        FT2 = LAT_DATA_PATH + '/bn%s/gll_ft2_tr_bn%s_v00.fit' % (TRIGGER_ID[i], TRIGGER_ID[i])
        
        analysis_dir.append( doLAT('%s' % TRIGGER_ID[i], RA[i], DEC[i], 
                TSTARTS=[tstart], TSTOPS=[tstop],
                ROI=5.0, ZMAX=105, EMIN=emin, EMAX=emax,
                IRF=irf, data_path=LAT_DATA_PATH) )

        lat_plugin.append(get_lat_like(tstart,tstop,FT2,TRIGGER_ID[i]))



        #get powerlaws for each source
        sources.append(setup_powerlaw_model('bn'+TRIGGER_ID[i],-2.0,RA[i],DEC[i],REDSHIFT = REDSHIFT[i]))


    #testing with perSourceLike
    model = Model(*sources)
    model.display(complete=True)
    
    plugin = DataList(*lat_plugin)
    
    source_names=[]
    for i in TRIGGER_ID:
        source_names.append('bn%s'%i)
    
    per = perSourceLike( 'combinedSourceLikelihood', model, plugin, source_names = source_names, quiet = False )
    per.fit()
    per.plot()
    



    #collate all sources into a single model\
    """
    print('Collating models ...')
    model = Model(*sources)
    models = [Model(sources[0]),Model(sources[1])]
    for i in range(0,len(models)):
        model[i].link(models.bn080916009.spectrum.main.composite.lower_bound_3,models.bn090102122.spectrum.main.composite.lower_bound_3)
        model[i].link(models.bn080916009.spectrum.main.composite.upper_bound_3,models.bn090102122.spectrum.main.composite.upper_bound_3)
    model.display(complete=True)
    plugin = DataList(*lat_plugin)
    
    #

    #perform analysis on group of sources
    print('Performing analysis ...')
    jl = jla.JointLikelihood(model,plugin)
    jl.fit()
    jl.results()
    """ 

    #  display_spectrum_model_counts(bayes, min_rate=20)
    #plot_spectra(jl.results, flux_unit='erg2/(cm2 s keV)', fit_cmap='viridis', contour_cmap = 'viridis', contour_style_kwargs = dict(alpha=0.1),energy_unit='MeV', ene_min=emin, ene_max=emax)


    #Plot settings
    plt.grid(True)
    plt.ylabel(r"Flux (erg$^{2}$ cm$^{-2}$ s$^{-1}$ TeV$^{-1}$)")

    if savePlots == True:
        plt.savefig('%s_%s'%(TRIGGER_ID,i)+'.png')
    else:
        plt.show()



    #bayesIndex = getattr(analysis[0].likelihood_model,'bn%s_powerlaw'%(TRIGGER_ID)).spectrum.main.Powerlaw.index.value

    #end main


