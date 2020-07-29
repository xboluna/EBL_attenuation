
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

from threeML import *

from GroupEBLAnalysis import *

import pandas as pd

path = 'EBL_candidates/selectedGRBs.csv'
df = pd.read_csv(path,dtype=str)
names = df['GRBNAME'].tolist()

x = GroupEBLAnalysis(names[0:12],csv_path = path)
x.do_fit()


results = x.get_fit_results
results.to_csv('results.csv')
#print('fit = %s +- %s'%(results.iloc[2,0],results.iloc[2,3]))

fig = x.plot()
fig.savefig('bayes_profile.png')

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
