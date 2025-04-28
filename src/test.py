import flowkit as fk
import numpy as np
import pandas as pd
import os

FILE_BASE = '/Users/trippwhaley/Downloads'
files = [
    'SCAN_Control_IFNa_LPS_1.fcs'
]

def get_timestamp_df(timestamp: str):
  df = pd.DataFrame()
  for f in files:
    flow_sample = fk.Sample(f'{FILE_BASE}/{f}', ignore_offset_error=True)
    # Maybe this arcsinh isn't necessary but I remember it being discussed in class for something, we can re-evaluate later
    flow_sample_np = np.arcsinh(1./5 * flow_sample._get_raw_events()[:,:])
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

get_timestamp_df('ad')