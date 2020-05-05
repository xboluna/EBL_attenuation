#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:59:16 2020

@author: xboluna
"""


import numpy as np
import matplotlib.pyplot as plt

"""
from jupyterthemes import jtplot
%matplotlib inline
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
plt.style.use("mike")
"""

import pandas as pd
import threeML

import warnings
warnings.simplefilter("ignore")



#pulls GRB name string from CSV format
def pullGRBName(string):
    return (string.split('\'')[1])

#query fermi database function
gbm_catalog = threeML.FermiGBMBurstCatalog()
def queryCatalog(string):
    return gbm_catalog.query_sources(string)

dirpath = "/Users/xboluna/Google Drive/Resume - Portfolio - Professional/Fermi Project/GRB analysis scripts/"

#data obtained from https://www-glast.stanford.edu/pub_data/953/2FLGC/
file = pd.read_csv("selectedGRBs.csv")["GRBNAME"]


GRBs=[]
for i in file:
    GRBs.append(pullGRBName(i))


"""
test = 'GRB'+GRBs(0)
gbm_catalog.query_sources(test)
grb_info =  gbm_catalog.get_detector_information()[test]
gbm_detectors = grb_info['detectors']
source_interval = grb_info["source"]["fluence"]
background_interval = grb_info["background"]["full"]
best_fit_model = grb_info["best fit model"]["fluence"]
model =  gbm_catalog.get_model(best_fit_model, "fluence")[test]
model
"""

#dl = threeML.download_GBM_trigger_data('bn'+GRBs(0),detectors=gbm_detectors)
