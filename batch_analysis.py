
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

from threeML import *

from GroupEBLAnalysis import *

x = GroupEBLAnalysis(['080916009','090102122','090510016'],csv_path = 'EBL_candidates/selectedGRBs.csv',fit_type='BayesianAnalysis')
x.fit()


results = []
results.append(x.get_fit_results)
#print('fit = %s +- %s'%(results.iloc[2,0],results.iloc[2,3]))

fig = x.plot()
fig.savefig('bayes_profile_bn080916009+090510016.png')

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
