import os
import numpy as np
from os.path import dirname, abspath
import datetime
from train import train_wrapper
from predict import predict

mode = 'train'

### common parameters ###
homedir = dirname(dirname(abspath(__file__)))
data_dir = homedir + '/data/'
test_data_dir = homedir + '/data/interpretation_points/F3/XL500/il_xl/'
segy_filename = homedir + '/data/F3_entire.segy'
now = datetime.datetime.now()
save_location = homedir + f'/output/dilation_cnn/{mode}_' + now.strftime('%Y-%m-%d_%H-%M') + '/'
test_files = [test_data_dir + x for x in os.listdir(test_data_dir)] if test_data_dir is not None else None
inp_res = np.float32
facies_names = ['else', 'grizzly', 'high_amp_cont', 'high_amp_dips', 'high_amplitude', 'low_amp_dips', 'low_amplitude',
                     'low_coherency', 'salt']

### predict mode parameters ###
model_path = dirname(dirname(save_location)) + '/hyper_test_riped_2020-11-15_22-53/set_0/trained.h5'

### train mode parameters ###
train_data_dir = homedir + '/data/interpretation_points/F3/IL339/il_xl/'
train_files = [train_data_dir + x for x in os.listdir(train_data_dir) if os.path.isfile(train_data_dir + x)]

n_param_samples = 5
n_epoch = 40
validation_split = 0.1

n_conv_layers_prior = [2, 3, 4, 5]
n_dilate_layers_prior = [2, 3, 4, 5]
kernel_size_prior = [3, 5, 7]
n_filters_prior = [16, 32]
lr_prior = [0.0005, 0.001]
window_size_prior = [16, 32, 64]
overlap_prior = [30, 50, 70, 80]
batch_size_prior = [4, 8, 15]
maxpool_prior = [True, False]
telescopic_prior = [True, False]

num_conv_layers_samples = np.array(n_conv_layers_prior)[np.random.randint(0, len(n_conv_layers_prior), size=n_param_samples)]
num_dil_layers_samples = np.array(n_dilate_layers_prior)[np.random.randint(0, len(n_dilate_layers_prior), size=n_param_samples)]
kernel_size_samples = np.array(kernel_size_prior)[np.random.randint(0, len(kernel_size_prior), size=n_param_samples)]
n_filters_samples = np.array(n_filters_prior)[np.random.randint(0, len(n_filters_prior), size=n_param_samples)]
lr_samples = np.array(lr_prior)[np.random.randint(0, len(lr_prior), size=n_param_samples)]
window_size_samples = np.array(window_size_prior)[np.random.randint(0, len(window_size_prior), size=n_param_samples)]
overlap_prior_samples = np.array(overlap_prior)[np.random.randint(0, len(overlap_prior), size=n_param_samples)]
batch_size_samples = np.array(batch_size_prior)[np.random.randint(0, len(batch_size_prior), size=n_param_samples)]
maxpool_samples = np.array(maxpool_prior)[np.random.randint(0, len(maxpool_prior), size=n_param_samples)]
telescopic_prior_samples = np.array(telescopic_prior)[np.random.randint(0, len(telescopic_prior), size=n_param_samples)]

train_dict = {
    'train_files': train_files,
    'test_files': test_files,
    'segy_filename': segy_filename,
    'inp_res': inp_res,
    'epochs': n_epoch,
    'validation_split': validation_split,
    'save_location': save_location,
    'facies_names': facies_names,
    'num_conv_layers_samples': num_conv_layers_samples,
    'num_dil_layers_samples': num_dil_layers_samples,
    'kernel_size_samples': kernel_size_samples,
    'n_filters_samples': n_filters_samples,
    'lr_samples': lr_samples,
    'window_size_samples': window_size_samples,
    'overlap_prior_samples': overlap_prior_samples,
    'batch_size_samples': batch_size_samples,
    'maxpool_samples': maxpool_samples,
    'telescopic_prior_samples': telescopic_prior_samples
}

if mode == 'train':
    train_wrapper(train_dict)

elif mode == 'predict':
    predict(model_path, test_files, segy_filename, inp_res, save_location, facies_names)

else:
    raise ValueError("mode should be train or predict")