import pandas as pd

AE_train_ids = [] # some subset of patient IDs
folder_path = './'

def our_matching(iter_id, df_source, df_target, timepoints, AE_train_ids=AE_train_ids):
    source, target = timepoints
    # optimize for the perfect matching
    print(f'Iter= {iter_id}')
    df_source_target_matched = pd.DataFrame()
    for train_id in AE_train_ids:
        ## Not dealing with this cost stuff
        # cost = pd.read_csv(folder_path+'cost_matrix_5k_source_Pre_target_1hr_scaled_train_id_{}.csv'.format(train_id), index_col=0) # the first col is index
        # print('finished loading csv train id = {}'.format(train_id))
        # cost = np.array(cost.values)
        # row_ind, col_ind = linear_sum_assignment(cost)

        df_source_p = df_source[df_source['patient_id']==train_id]
        df_source_p = df_source_p.sample(n=5000, random_state=42)
        df_source_p.reset_index(drop=True, inplace=True)
        df_target_p = df_target[df_target['patient_id']==train_id]
        df_target_p = df_target_p.sample(n=5000, random_state=42)
        df_target_p.reset_index(drop=True, inplace=True)
        ## Dont need these where we're going
        # df_source_p = df_source_p.drop('timepoint', axis=1)
        # df_target_p = df_target_p.drop('timepoint', axis=1)
        df_source_p = df_source_p.drop('patient_id', axis=1)
        df_target_p = df_target_p.drop('patient_id', axis=1)

        df_source_p.columns = [col+'_{}'.format(source) for col in df_source_p.columns.values]
        df_target_p.columns = [col+'_{}'.format(target) for col in df_target_p.columns.values]

        # THis might prove to be important
        # df_target_p = df_target_p.reindex(col_ind)
        df_target_p.reset_index(drop=True, inplace=True)

        df_p_source_target_matched = pd.concat([df_source_p, df_target_p], axis=1)
        df_p_source_target_matched['patient_id'] = train_id
        df_source_target_matched = pd.concat([df_source_target_matched, df_p_source_target_matched], axis=0)

    print('finished. Writing to csv..')
    # training_data_path = f"/content/EnGen/EnGen_train_iterations/training_data/iter_{iter_id}/"
    # os.makedirs(training_data_path, exist_ok=True)
    # df_source_target_matched.to_csv(training_data_path+'Func_Pheno_45k_scaled_with_{}_tps_source_{}_target_{}_matched.csv'.format('_'.join(timepoints), source, target), index=False, header=True)

    df_source_target_matched.drop(f'time_{timepoints[0]}', axis=1, inplace=True)
    df_source_target_matched.drop(f'event_length_{timepoints[0]}', axis=1, inplace=True)
    df_source_target_matched.drop(f'time_{timepoints[1]}', axis=1, inplace=True)
    df_source_target_matched.drop(f'event_length_{timepoints[1]}', axis=1, inplace=True)
    df_source_target_matched.to_csv(folder_path+'Func_Pheno_45k_scaled_with_{}_tps_source_{}_target_{}_matched.csv'.format('_'.join(timepoints), source, target), index=False, header=True)
    return df_source_target_matched
