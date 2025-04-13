import os.path
import subprocess

import numpy as np
import torch
from torch import Tensor
from warnings import catch_warnings

from gan import GAN, Generator, Discriminator
from data_loader import get_training_datasets
from EnGen.EnGen_Iteration_preprocess import our_matching
from EnGen.EnGen_model.train import train_engen

# def train(t1: Tensor, t2: Tensor, ts1: str, ts2: str):
#     default_size = t1.shape[1]
#     generator = Generator(default_size, default_size)
#     discriminator = Discriminator(2*default_size)
#     model = GAN(generator=generator, discriminator=discriminator)
#
#     model.do_train(data=torch.concat([t1, t2]), source_ts=ts1, target_ts=ts2)
#
#     return model
#
# def evaluate(model, data):
#     model.eval()
#     test_t2 = model.generator(data).detach().numpy()
#     print('hi')

def main(ts1: str, ts2: str):
    num_iters = 3  # use 30 for real run

    if not os.path.exists(f'/Users/trippwhaley/Projects/comp-bio-final/src/Func_Pheno_45k_scaled_with_{ts1}_{ts2}_tps_source_{ts1}_target_{ts2}_matched.csv'):
        with catch_warnings(action='ignore'):
            t1_df, t2_df = get_training_datasets(ts1, ts2)
        # t1 = torch.tensor(t1_df.values.astype(np.float32))
        # t2 = torch.tensor(t2_df.values.astype(np.float32))

        # Preprocessing
        # Seems like only these 3 IDs exist in both 24H and 14D
        for iter_id in range(num_iters):
            our_matching(iter_id, t1_df, t2_df, AE_train_ids=['02', '03', '12'], timepoints=[ts1, ts2])

    train_engen(num_iters)
    # subprocess.call(['python', f'/Users/trippwhaley/Projects/comp-bio-final/EnGen/EnGen_model/train.py', '--iter_id', f'{num_iters}', '--source', '{ts1}', '--target', '{ts2}'], shell=True)
    # model = train(t1, t2, ts1, ts2)
    # # later generate for all timesteps
    # evaluate(model, t1)

if __name__ == "__main__":
    main('24H', '14D')