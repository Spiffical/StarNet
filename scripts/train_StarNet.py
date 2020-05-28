import os, sys
sys.path.insert(0, os.path.join(os.getenv('HOME'), 'StarNet'))
import argparse
from starnet.models.cnn_models import StarNet2017, StarNet2017DeepEnsemble, StarResNet, StarResNetDeepEnsemble, \
    StarResNetSmallDeepEnsemble, StarResNetSmall, StarResNetDeepEnsembleTwoOutputs, StochasticResNet

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='Path of training file')
parser.add_argument('--num_train', type=int, required=True,
                    help='Size of training set')
parser.add_argument('--targets', nargs='+', required=True,
                    help='Keys of h5py file to train on')
parser.add_argument('--val_data_path', type=str, default='',
                    help='(optional) path of validation data set')
parser.add_argument('--model_type', type=str, default='StarNet2017',
                    help='Which model type to train (StarNet2017, StarNet2017DeepEnsemble, StarResNet, etc.)')
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
parser.add_argument('-v', '--verbose', type=int, default=1,
                    help='level of output (0: nothing, 1: progress bar, 2: epoch)')
parser.add_argument("--hardmine", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Activate hard mining")


args = parser.parse_args()

data_path = args.data_path
val_data_path = args.val_data_path
model_type = args.model_type
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
use_hardmining = args.hardmine

if model_type == 'StarNet2017':
    NN = StarNet2017()
elif model_type == 'StarNet2017DeepEnsemble':
    NN = StarNet2017DeepEnsemble()
elif model_type == 'StarResNet':
    NN = StarResNet()
elif model_type == 'StarResNetSmall':
    NN = StarResNetSmall()
elif model_type == 'StarResNetDeepEnsemble':
    NN = StarResNetDeepEnsemble()
elif model_type == 'StarResNetSmallDeepEnsemble':
    NN = StarResNetSmallDeepEnsemble()
elif model_type == 'StarResNetDeepEnsembleTwoOutputs':
    NN = StarResNetDeepEnsembleTwoOutputs()
elif model_type == 'StochasticResNet':
    NN = StochasticResNet()
else:
    raise ValueError('Model type {} not valid'.format(model_type))

if finetune_model == 'None': finetune_model = None

NN.folder_name = save_folder
NN.targetname = targets
NN.spec_name = spec_key
NN.data_filename = data_path
NN.val_data_filename = val_data_path
NN.telluric_mask_file = telluric_mask_file
NN.num_train = num_train
NN.use_val_generator = 'auto'
NN.shuffle_indices = False

NN.verbose = args.verbose  # 0: nothing, 1: progress bar, 2: epoch
NN.batch_size = batch_size
NN.max_epochs = max_epochs
NN.max_added_noise = max_noise
NN.max_frac_zeros = max_frac_zeros
NN.num_gpu = num_gpu
NN.use_hard_mining = bool(use_hardmining)

# To enable autosave
NN.autosave = True

print('HARD MINING: {}'.format(use_hardmining))

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
