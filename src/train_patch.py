import os
import sys
import time
import torch
import numpy as np
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from sklearn import preprocessing
from EnGen.EnGen_model.models import EnGen

from EnGen.EnGen_model.utils import cytofDataset, GlobalsVars


def train_engen(iter_id=0, batch_size=2048, epochs=100, seed=42):
    globals_vars = GlobalsVars(iter_id)

    learning_rate = 0.005
    encoder_layer_sizes = [47, 128, 256, 256]
    decoder_layer_sizes = [256, 256, 128, 47]
    latent_size = 128
    print_every = 10

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('mps')
    # torch.cuda.set_device(device)

    cytof_dataset = cytofDataset(globals_vars=globals_vars)
    dataloader = DataLoader(cytof_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.MSELoss(reduction='mean')

    engen = EnGen(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        device=device)

    optimizer = torch.optim.Adam(
        engen.parameters(), lr=learning_rate)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, cooldown=10,
                                               min_lr=10e-3)

    logs = defaultdict(list)
    #
    # # Load checkpoint if available
    # if args.ckpt is not None:
    #     print('Loading last checkpoint..')
    #     engen.load_ckpt(args.ckpt)

    engen.to(device)
    Tensor = torch.FloatTensor

    best_model_dict = {'epoch': 0, 'min_total_loss': float('inf')}
    for epoch in range(epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        engen.train()
        total_loss = 0
        pbar = tqdm(dataloader, ascii=True, desc="Epoch: {:2d}/{}".format(epoch + 1, epochs))
        current_lr = optimizer.param_groups[0]['lr']

        for data in pbar:

            x_in = Variable(data['row_source'].type(Tensor)).to(device)
            x_out = Variable(data['row_target'].type(Tensor)).to(device)

            recon_x, z = engen(x_in)

            loss = criterion(recon_x, x_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Bookkeeping
            if total_loss == 0:
                total_loss = loss.item()
            else:
                total_loss = total_loss * .95 + loss.item() * .05

            pbar.set_postfix({'total_loss': loss.item(), 'lr': current_lr})

        scheduler.step(loss)

        logs['epoch'].append(epoch)
        logs['learning_rate'].append(current_lr)
        logs['total_loss'].append(total_loss)

        if epoch % print_every == 0 or epoch == epochs - 1 or (
                total_loss < best_model_dict['min_total_loss'] and epoch >= best_model_dict['epoch'] + 20):
            engen.eval()

            filename = globals_vars.dir_path_csv + 'epoch{}_model_params.txt'.format(epoch)
            with open(filename, 'w') as filetowrite:
                model_params = [str(param) for param in engen.parameters()]
                filetowrite.writelines(model_params)
                filetowrite.close()

            if total_loss < best_model_dict['min_total_loss'] and epoch >= best_model_dict['epoch'] + 20:
                print('********* best model found so far: total_loss={}'.format(total_loss))
                engen.save_ckpt(globals_vars.dir_path_ckpt + 'best_model_engen.pth')
                with open(globals_vars.dir_path_ckpt + 'best_model_engen.txt', 'a+') as f:
                    f.write('\n===========================================\n')
                    f.write('\n'.join("{}={}".format(key, val[-1]) for (key, val) in logs.items()))
                best_model_dict['epoch'] = epoch
                best_model_dict['min_total_loss'] = total_loss
            else:
                engen.save_ckpt(globals_vars.dir_path_ckpt + 'last_model_engen.pth')
            if epoch % 500 == 499:
                engen.save_ckpt(globals_vars.dir_path_ckpt + 'epoch_{}_model_engen.pth'.format(epoch))

            pd_logs = pd.DataFrame(logs)
            pd_logs.to_csv(globals_vars.dir_path_csv + 'Logs_engen.csv', index=False)


