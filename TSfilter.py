#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from GroupEBLAnalysis import *

import pandas as pd

import os

path = 'EBL_candidates/selectedGRBs.csv'
df = pd.read_csv(path,dtype=str)
names = df['GRBNAME'].tolist()

x = GroupEBLAnalysis(['091127976','120624933','130907904','131108862','150514774','171010792','180720598'],csv_path = path,fit_type='JointLikelihood',attenuation_model='kneiske',OUTDIR = os.getcwd()+'/GroupEBLAnalysis_TS_run')
x.save_TS()

x.populate_model(rollingSave=True)

fig = x.plot_TS()
fig.savefig('TSvRS.png')

exit()
y = x.DATA['prelim_index']
x.do_fit(quiet = False)
x.save_fit_results()

results = x.get_fit_results

#print('fit = %s +- %s'%(results.iloc[2,0],results.iloc[2,3]))

fig = x.plot()
fig.savefig('bayes_profile.png')

from matplotlib import rc
rc('text',usetex=False)
fig = x.FIT.results.corner_plot()
fig.savefig('corner_plot.png')

"""
for i in ['dominguez','finke','gilmore','franceschini']:
    
    x.set_attenuation_model(i)
    x.update_attenuation_model()
    x.fit()
    results.append(x.get_fit_results)
    

for i in results:
    print(i)
"""

x.FIT.print_results()
