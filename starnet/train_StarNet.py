import os, sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
import argparse
from starnet.nn.models.cnn_models import StarNet2017, StarNet2017DeepEnsemble, StarResNet, StarResNetDeepEnsemble

parser = argparse.ArgumentParser()
parser.add_argument('--datafile_path', type=str, required=True,
                    help='Path of training file')
parser.add_argument('--num_train', type=int, required=True,
                    help='Size of training set')
parser.add_argument('--targets', nargs='+', required=True,
                    help='Keys of h5py file to train on')
parser.add_argument('--model_to_train', type=str, default='StarNet2017',
                    help='Which model to train (StarNet2017, StarNet2017DeepEnsemble)')
parser.add_argument('--save_folder', type=str, default=None,
                    help='Folder to save trained model in (if None, folder name created based on date)')
parser.add_argument('-s', '--spec_key', type=str, default='spectra_starnetnorm',
                    help='Key of h5py referring to flux values')
parser.add_argument('-b', '--batch_size', type=int, default=32,
                    help='Number of spectra used in a single batch')
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help='Maximum number of epochs for training')
parser.add_argument('-z', '--zeros', type=float, default=0.10,
                    help='Maximum fraction of spectrum to be augmented with zeros')
parser.add_argument('-t', '--telluric_file', type=str, default=None,
                    help='Path of file containing information about telluric lines to mask')
parser.add_argument('-m', '--finetune_model', type=str, default=None,
                    help='Name of trained h5 model file to finetune')
parser.add_argument('-n', '--noise', type=float, default=None,
                    help='Maximum fraction of noise to be added to spectra (if training set does not already have'
                         'noise added in the spectra)')
parser.add_argument('-g', '--num_gpu', type=int, default=1,
                    help='Number of GPUs available for training')
args = parser.parse_args()

datafile_path = args.datafile_path
model_to_train = args.model_to_train
num_train = args.num_train
targets = args.targets
spec_key = args.spec_key
save_folder = args.save_folder
max_frac_zeros = args.zeros
max_epochs = args.epochs
batch_size = args.batch_size
telluric_mask_file = args.telluric_file
finetune_model = args.finetune_model
max_noise = args.noise
num_gpu = args.num_gpu

if model_to_train == 'StarNet2017':
    NN = StarNet2017()
elif model_to_train == 'StarNet2017DeepEnsemble':
    NN = StarNet2017DeepEnsemble()
elif model_to_train == 'StarResNet':
    NN = StarResNet()
elif model_to_train == 'StarResNetDeepEnsemble':
    NN = StarResNetDeepEnsemble()
else:
    raise ValueError('Model type {} not valid'.format(model_to_train))

if finetune_model == 'None': finetune_model = None

NN.folder_name = save_folder
NN.targetname = targets
NN.spec_name = spec_key
NN.data_filename = datafile_path
NN.telluric_mask_file = telluric_mask_file
NN.num_train = num_train
NN.spec_norm = True  # if True, the spectrum is already normalized so don't perform continuum normalization

NN.verbose = 1  # 0: nothing, 1: progress bar, 2: epoch
NN.batch_size = batch_size
NN.max_epochs = max_epochs
NN.max_added_noise = max_noise
NN.max_frac_zeros = max_frac_zeros
NN.num_gpu = num_gpu

# To enable autosave
NN.autosave = True

# Load trained model if finetuning requested
if save_folder is not None and finetune_model is not None:
    finetune_model_path = os.path.join(save_folder, finetune_model)
    if os.path.exists(finetune_model_path):
        print('Loading pre-trained model: %s' % finetune_model_path)
        NN.load_pretrained_model(finetune_model_path)
    else:
        print('Pre-trained model does not exist: %s' % finetune_model_path)
        print('Training new model...')
elif save_folder is not None and finetune_model is None:
    if os.path.exists(save_folder):
        print('Save folder provided already exists. Creating new folder to avoid overwriting')
        NN.save_folder = None  # None type results in unique foldername generation

# Start the training
NN.train()
