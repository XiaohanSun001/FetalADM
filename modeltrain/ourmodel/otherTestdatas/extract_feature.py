"""
Created on Fri Mar  5 00:34:25 2021

@author: Dell
"""

import pandas as pd
import numpy as np


def data_extract(filename,outname):

#    feature_selet = ['label','MaxMin', 'SDchr13','SDchr18' ,'SDchr21','MW',
#              'GW',	'AR', 'DR',
#             'R_UR',	'GC' ,'SDchrX',	'SDchrY',  'ChrX', 'ChrY']

    feature_selet = ['label','Chr21','MaxMin','Zchr21','DR','Zchr18','Chr18','Zchr13','Chr13',
               'ChrY','FY','R','R_UR','PT','GC']
             
    all_data = pd.read_csv(filename)
    data = all_data[feature_selet]
    
    data.to_csv(outname, index=False)            

filename1='T211_25_standard.csv'
filename2='T136_25_standard.csv'



data_extract(filename1,'T211_14_standard.csv')
data_extract(filename2,'T136_14_standard.csv')

