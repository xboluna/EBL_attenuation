
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

from threeML import *

from GroupEBLAnalysis import *

x = GroupEBLAnalysis(['080916009','090510016'],csv_path = 'EBL_candidates/selectedGRBs.csv')
x.fit()

"""
profile_fig = x.plot_param_profile('bn080916009')
profile_fig.savefig('joint_profile_bn080916009+090510016.png')
"""

results = x.get_fit_results
print('fit = %s +- %s'%(results.iloc[2,0],results.iloc[2,3]))


pdb.set_trace()
