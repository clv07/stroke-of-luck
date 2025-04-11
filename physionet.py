import random
import os
import argparse
import csv
import glob

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from helper_code import *


### Read in 12SL statements per TestID
stmnts_dir = "C:\\Documents\\Code\\PhysioNet_2020-1\\12SL\\PTBXL\\"
df_stmnts = pd.read_csv(stmnts_dir + '12slv24_stt.csv')
print(df_stmnts.columns)

## Create binary MI column for 12SL diagnosis
MI_stmnts_12SL = [740,810,820, ## anterior infarct
                  700,810, ## septal infarct
                  760,820, ## lateral infarct
                  780,801,806, ## inferior infarct
                  801,802,803, ## posterior infarct
                  826,827,963,964,965,966,967,968, ## infarct - ST elevation
                  4,821,822,823,826,827,828,829,920,930,940,950,960,961,962,1361, ## acute MI or injury
                  ]
MI_stmnts_12SL = set(MI_stmnts_12SL)
df_stmnts['MI_12SL'] = 0
for MI_stmnt in MI_stmnts_12SL: 
    df_stmnts.loc[df_stmnts['Statements'].str.contains('|%d|'%MI_stmnt,regex=False), 'MI_12SL'] = 1
print('%d records in the PTBXL dataset with MI-related diagnoses from 12SL' %
      df_stmnts['MI_12SL'].sum() )
    

### Read in filenames for PhysioNet headers and recordings
trainingset_dir = 'C:\\Data\\WFDB_PTBXL\\'
header_files, recording_files = find_challenge_files(trainingset_dir)
num_recordings = len(header_files)
print(' %d records found' % num_recordings)

## Add column for diagnostic statements from physician
df_stmnts['Statements_Phys'] = 0
for header_file in header_files:
    file_id = int( header_file[-9:-4] )
    header = load_header(header_file)
    current_labels = get_labels(header)
    ## Format statements same as the 12SL Statements column
    stmnts_str = ''
    for each_label in current_labels: stmnts_str+='|%s'%each_label
    stmnts_str += '|'
    df_stmnts.loc[df_stmnts['TestID']==file_id, 'Statements_Phys'] = stmnts_str

## Create binary MI column for Physician diagnosis
MI_stmnts_Phys = [57054005, ## acute myocardial infarction
                  413444003, #acute myocardial ischemia	
                  426434006, #anterior ischemia	
                  54329005, #anterior myocardial infarction	
                  #413844008, #chronic myocardial ischemia	
                  425419005, #inferior ischaemia	
                  #704997005, #inferior ST segment depression	
                  425623009, #lateral ischaemia	
                  164865005, #myocardial infarction	
                  164861001, #myocardial ischemia	
                  #164867002, #old myocardial infarction	
                  #429622005, #st depression	
                  #164931005, #st elevation	
                  #266257000, #transient ischemic attack	
                  ]
MI_stmnts_Phys = set(MI_stmnts_Phys)
df_stmnts['MI_Phys'] = 0
for MI_stmnt in MI_stmnts_Phys: 
    df_stmnts.loc[df_stmnts['Statements_Phys'].str.contains('|%d|'%MI_stmnt,regex=False), 'MI_Phys'] = 1
print('%d records in the PTBXL dataset with MI-related diagnoses from Physician' %
      df_stmnts['MI_Phys'].sum() )

df_missedMI = df_stmnts.loc[(df_stmnts['MI_Phys']==1) & (df_stmnts['MI_12SL']==0)]
print(' %d Records with MI diagnosis from Physician that were missed by 12SL :' % len(df_missedMI))
for testID in df_missedMI['TestID'].values[:300]: 
    print(testID, ',')

   

''' 
** This code segment reads in all diagnostic classes and the mapping files. 
** Not required for the MI application, if we just read the mappings for that diagnosis. 
# Extract the classes from the dataset.
print('Extracting classes...')
classes = set()
for header_file in header_files:
    header = load_header(header_file)
    classes |= set(get_labels(header))
if all(is_integer(x) for x in classes):
    classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers
else:
    classes = sorted(classes) # Sort classes alphanumerically if not numbers
num_classes = len(classes)
print('Found %d classes' % num_classes)
print(classes)

## Read mappings of diagnostic codes
mappings_dir = "C:\\Documents\\Code\\PhysioNet_2020-1\\evaluation-2021\\"
dx_scored = pd.read_csv(mappings_dir + 'dx_mapping_scored.csv')
dx_unscored = pd.read_csv(mappings_dir + 'dx_mapping_unscored.csv')
'''
