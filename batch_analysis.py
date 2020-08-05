
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

if True:
    import sys
    sys.path.insert(0,'threeML_repo')
from threeML import *

from GroupEBLAnalysis import *

import pandas as pd

path = 'EBL_candidates/selectedGRBs.csv'
df = pd.read_csv(path,dtype=str)
names = df['GRBNAME'].tolist()

x = GroupEBLAnalysis(names[0:2],csv_path = path,fit_type='multinest')
y = x.DATA['prelim_index']
x.do_fit(quiet = False)
x.save_fit_results()

results = x.get_fit_results

#print('fit = %s +- %s'%(results.iloc[2,0],results.iloc[2,3]))

fig = x.plot()
fig.savefig('bayes_profile.png')

pdb.set_trace()

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

pdb.set_trace()

x.FIT.print_results()
