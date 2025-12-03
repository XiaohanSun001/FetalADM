import pandas as pd
import numpy as np


def data_extract(filename,outname):

             
    df = pd.read_csv(filename)
    #data = all_data[feature_selet]
    df['label'] = df['label'].apply(lambda x: x-1 if x != 0 else 0)

    df.to_csv(outname, index=False)            

filename='modeldata178_13.csv'
filename2='test10_13.csv'
filename3='primary38_13.csv'

data_extract(filename3,'primary38_13_label3.csv')
data_extract(filename,'modeldata178_13_label3.csv')
data_extract(filename2,'test1013_label3.csv')
