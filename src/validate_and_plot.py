import itertools
import os.path
from warnings import catch_warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from data_loader import get_training_dataset
from EnGen_generate_patch import GenerateEnGen

num_iters = 30  # use 30 for real run
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# iteration num hardcoded for now, sorry
num_iters_str = "0" + str(num_iters) if num_iters < 10 else num_iters
MODEL_PATH = os.path.join(BASE_PATH, '..', f'EnGen_train_iterations/engen_output/iter_{num_iters_str}/saved_model/best_model_engen.pth')


generate_ids = ['03']
def main(ts1: str, ts2: str, model_path=MODEL_PATH):

    with (catch_warnings(action='ignore')):
        source_df = get_training_dataset(ts1)
        source_df = source_df[source_df['patient_id'] == '03']
        val_df = get_training_dataset(ts2)
        val_df = val_df[val_df['patient_id'] == '03']
        val_df.drop(columns=['time', 'event_length', 'patient_id'], inplace=True)

    print("testing")

    # Always run generation for now
    if not os.path.exists('./generated.csvgenerated_03_scaled.csv'):
        generator = GenerateEnGen(model_path=model_path, test_patient_ids=generate_ids, source=source_df, iter_id=num_iters)
        generator.generate_csv()

    # Compare results similarly to EnGen Paper
    gen_df = pd.read_csv('./generated.csvgenerated_03_scaled.csv')


    combined_df = pd.concat([val_df, gen_df], keys=['val', 'gen'])

    # Compare variance with imputed data to og data
    cv = lambda x: np.abs(np.std(x, ddof=1) / np.mean(x) * 100)
    og = val_df.drop(columns='beadDist').apply(cv) #.sort_values(key=lambda x: abs(x)).index
    imp = combined_df.drop(columns='beadDist').apply(cv) #.sort_values(key=lambda x: abs(x)).index
    cols = imp.subtract(og).sort_values(key=lambda x: abs(x))

    print(cols)
    print(np.mean(np.abs(cols.values)))
    print(np.median(np.abs(cols.values)))
    print(len([x for x in cols if abs(x) > 50]))
    best_column_combos = list(itertools.combinations(cols.index[0:3], 2))
    worst_column_combos = list(itertools.combinations(cols.index[-3:], 2))
    # print(cv_values)

    gen_means = gen_df.mean()
    gen_vars = gen_df.var()
    val_means = val_df.mean()
    val_vars = val_df.var()
    diffs = {}
    for i, (genm, genv, valm, valv) in enumerate(zip(gen_means, gen_vars, val_means, val_vars)):
        marker = gen_df.columns[i]
        diffs[marker] = {'mean': abs(genm - valm), 'variance': abs(genv - valv)}
    mean_diffs = dict(sorted(diffs.items(), key=lambda v: v[1]['mean']))
    var_diffs = dict(sorted(diffs.items(), key=lambda v: v[1]['variance']))
    # best_means = list(mean_diffs.keys())[0:3]
    # best_vars = list(var_diffs.keys())[0:3]
    # worst_means = list(mean_diffs.keys())[-3:]
    # worst_vars = list(var_diffs.keys())[-3:]

    # best_column_combos = list(itertools.combinations(list(set(best_means).union(set(best_vars))), 2))
    # worst_column_combos = list(itertools.combinations(list(set(worst_means).union(set(worst_vars))), 2))
    column_combos = best_column_combos + worst_column_combos

    for combo in column_combos:
        f1, f2 = combo
        fig, (axgt, axgen) = plt.subplots(ncols=2)
        fig.tight_layout()
        gt_x, gt_y = val_df[f1].to_numpy(), val_df[f2].to_numpy()
        gen_x, gen_y = gen_df[f1].to_numpy(), gen_df[f2].to_numpy()
        # mean_diff_x = mean_diffs[f1]['mean']
        # mean_diff_y = mean_diffs[f2]['mean']
        # var_diff_x = mean_diffs[f1]['variance']
        # var_diff_y = mean_diffs[f2]['variance']
        axgt.set_xscale('log')
        axgt.set_yscale('log')
        axgen.set_xscale('log')
        axgen.set_yscale('log')
        axgt.set_xlim(3e-3, 3e3)
        axgt.set_ylim(3e-3, 3e3)
        axgen.set_xlim(3e-3, 3e3)
        axgen.set_ylim(3e-3, 3e3)
        axgt.scatter(gt_x, gt_y)
        axgen.scatter(gen_x,gen_y)
        fig.supxlabel(f1)
        fig.supylabel(f2)
        # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        # axgen.legend([extra, extra, extra], (f1, f"Mean {mean_diff_x}", f"Variance: {var_diff_x}"))
        # axgt.legend([extra, extra, extra], (f2, f"Mean {mean_diff_y}", f"Variance: {var_diff_y}"))
        plt.show()

if __name__ == "__main__":
    main('24H', '14D')
    # main('24H', '14D', True, MODEL_PATH_2)