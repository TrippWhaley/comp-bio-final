import flowkit as fk
import numpy as np
import pandas as pd
import os

FILE_BASE = '../data/'
file_names = [f.split('/')[-1] for f in os.listdir(FILE_BASE) if f.startswith('SCAN_10')]
timestamps = set([f.split('_')[-1].split('.fcs')[0] for f in file_names])
timestamp_grouped_files = {t: sorted([f for f in file_names if f.endswith(f"{t}.fcs")]) for t in timestamps}

def get_timestamp_df(timestamp: str):
  df = pd.DataFrame()
  for f in timestamp_grouped_files[timestamp]:
    flow_sample = fk.Sample(f'{FILE_BASE}/{f}', ignore_offset_error=True)
    # Maybe this arcsinh isn't necessary but I remember it being discussed in class for something, we can re-evaluate later
    # flow_sample_np = np.arcsinh(1./5 * flow_sample._get_raw_events()[:,:])
    flow_sample_np = flow_sample._get_raw_events()[:,:]
    col_names = np.array(flow_sample.pns_labels)
    # we know these for a fact, and they come in empty on pns labels
    col_names[0] = 'time'
    col_names[1] = 'event_length'
    # create df
    flow_sample_df = pd.DataFrame(flow_sample_np,columns=col_names)
    # Add patient_id to df
    patient_id = f.split('_')[1][-2:]
    flow_sample_df['patient_id'] = patient_id
    df = pd.concat([df, flow_sample_df])
  return df

def get_training_datasets(t1: str, t2: str):
  t1 = get_timestamp_df(t1)
  t2 = get_timestamp_df(t2)
  ids = set(t1['patient_id'].unique()).intersection(t2['patient_id'].unique())
  return t1.loc[t1['patient_id'].isin(ids)], t2.loc[t2['patient_id'].isin(ids)]

def get_training_dataset(t1: str):
  t1 = get_timestamp_df(t1)
  ids = set(t1['patient_id'].unique())
  return t1.loc[t1['patient_id'].isin(ids)]