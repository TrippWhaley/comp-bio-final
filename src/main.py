import numpy as np
import torch
from torch import Tensor
from warnings import catch_warnings

from gan import GAN, Generator, Discriminator
from data_loader import get_training_datasets

def train(t1: Tensor, t2: Tensor):
    default_size = t1.shape[1]
    generator = Generator(default_size, default_size)
    discriminator = Discriminator(2*default_size)
    model = GAN(generator=generator, discriminator=discriminator)

    model.do_train(data=torch.concat([t1, t2]), source_ts=ts1, target_ts=ts2)

    return model

def evaluate(model, data):
    model.eval()
    test_t2 = model.generator(data).detach().numpy()
    print('hi')

def main(ts1: str, ts2: str):
    with catch_warnings(action='ignore'):
        t1_df, t2_df = get_training_datasets(ts1, ts2)
    t1 = torch.tensor(t1_df.values.astype(np.float32))
    t2 = torch.tensor(t2_df.values.astype(np.float32))
    model = train(t1, t2)
    # later generate for all timesteps
    evaluate(model, t1)

if __name__ == "__main__":
    main('24H', '7D')