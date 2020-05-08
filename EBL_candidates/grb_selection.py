#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:59:16 2020

@author: xboluna
"""

from astropy.table import Table
import pandas as pd

#data obtained from https://www-glast.stanford.edu/pub_data/953/2FLGC/
file = Table.read('gll_2flgc.fits.txt')
df = file.to_pandas()

#criterion for specific GRBs:
#usable, confirmed redshift values
redshift = df['REDSHIFT'] > 0
#zenith no greater than 105
zenith = df['ZENITH'] < 105.2
df[redshift & zenith].to_csv(r'selectedGRBs.csv')

