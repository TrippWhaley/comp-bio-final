import numpy as np
import torch
import pandas as pd
import pickle
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable

from EnGen.EnGen_model.models import EnGen


class GenerateEnGen(object):
    def __init__(self, model_path, test_patient_ids, source, iter_id,
                 random_state=42):

        self.random_state = random_state
        self.test_patient_ids = test_patient_ids
        self.iter_id = iter_id
        self.model_path = model_path
        self.source = source
        self.csv_path = './generated.csv'
        self.device = torch.device('cuda:{}'.format(self.model_args['GPU_ID']) if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model().to(self.device)

    def get_model_arguments(self):
        with open(self.args_path, 'r') as f:
            lines = f.readlines()
        args = {}
        for l in lines:
            l = l.replace('\n', '')
            args.update({l.split('=')[0]: l.split('=')[1]})

        return args

    def loss_fn(self, gen_x, x_1hr):

        mse_loss = nn.MSELoss(reduction='mean')
        return mse_loss(gen_x, x_1hr)

    def load_model(self):

        encoder_layer_sizes = [47, 128, 256, 256]
        decoder_layer_sizes = [256, 256, 128, 47]
        latent_size = 128
        engen = EnGen(
            encoder_layer_sizes=encoder_layer_sizes,
            latent_size=latent_size,
            decoder_layer_sizes=decoder_layer_sizes,
            device=self.device)
        print('Loading the best engen model..')
        engen.load_ckpt(self.model_path)

        return engen

    ## We already do this
    # def arcsinh_transformation(self, x):
    #     a = 0
    #     b = 1 / 5
    #     c = 0
    #     return np.arcsinh(a + b * x) + c

    def generate_csv(self):

        for test_id in self.test_patient_ids:
            df_source = self.source[self.source['patient_id'] == test_id].astype(float)
            if len(df_source) == 0:
                continue
            df_source.drop(['time', 'event_length', 'patient_id'], axis=1, inplace=True)
            print('generating for test patient {}'.format(test_id))

            # This was originally using a Scaler from args['AE_scaler'], which we should actually use tbh
            scaler = StandardScaler()
            print('fitting scaler')
            scaler.fit(df_source.iloc[:, :].values)
            # scaler = self.p['AE_scaler']
            df_source.iloc[:, :] = scaler.transform(df_source.iloc[:, :].values)

            df_source = df_source.sample(frac=1, random_state=42)
            df_source.reset_index(drop=True, inplace=True)

            self.model.eval()
            x_source = torch.from_numpy(df_source.values)

            x_source = Variable(x_source.type(torch.FloatTensor))

            gen_target, z_source = self.model(x_source)

            df_gen_target = pd.DataFrame(data=np.array(torch.Tensor.cpu(gen_target).detach()),
                                         columns=df_source.columns.values)

            df_gen_target.to_csv(self.csv_path + 'generated_{}_scaled.csv'.format(test_id), header=True, index=False)
            df_gen_target.iloc[:, :] = scaler.inverse_transform(df_gen_target.iloc[:, :].values)
            df_gen_target.to_csv(self.csv_path + 'generated_{}.csv'.format(test_id), header=True, index=False)


#