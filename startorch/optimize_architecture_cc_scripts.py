import os
import argparse
import numpy as np
import random

HOME_DIR = os.getenv('HOME')
SCRATCH_DIR = os.getenv('SCRATCH')
#SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
PYTHONPATH = os.getenv('PYTHONPATH')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of training file')
parser.add_argument('--data_path', type=str, required=True,
                    help='Path of training file')
parser.add_argument('--num_train', type=int, required=True,
                    help='Size of training set')
parser.add_argument('--train_script', type=str, default=os.path.join(HOME_DIR, 'StarNet/scripts/train_StarNet.py'),
                    help='Path to training script')
parser.add_argument('--targets', nargs='+', required=True,
                    help='Keys of h5py file to train on')
parser.add_argument('--spec_key', type=str, required=True,
                    help='h5py key for spectra')
parser.add_argument('--virtual_env', type=str, required=True,
                    help='Path to virtual environment to use')
parser.add_argument('--save_folder', type=str, default=None,
                    help='Folder to save trained model in (if None, folder name created based on date)')
parser.add_argument('-e', '--epochs', type=int, default=300,
                    help='Maximum number of epochs for training')
parser.add_argument('--noise_addition', type=str, default='False',
                        help='Add noise if True')
parser.add_argument('--remove_gaps', type=str, default='False',
                    help='Remove interccd gaps if True')
parser.add_argument('--remove_arm', type=str, default='False',
                    help='Randomly remove blue or green arm if true')
parser.add_argument('--weight_decay', type=float, default=0.,
                    help='Weight decay (for L2 regularization)')
parser.add_argument('--val_data_path', type=str, default='',
                        help='Path of validation file')
parser.add_argument('--min_wvl', type=float, default=None,
                        help='Minimum wavelength of spectra that should be kept')
parser.add_argument('--max_wvl', type=float, default=None,
                    help='Maximum wavelength of spectra that should be kept')
args = parser.parse_args()

output_path = args.output_path
data_path = args.data_path
num_train = args.num_train
train_script = args.train_script
targets = args.targets
save_folder = args.save_folder
spec_key = args.spec_key
max_epochs = args.epochs
virtual_env = args.virtual_env
epochs = args.epochs
noise_addition = args.noise_addition
remove_gaps = args.remove_gaps
remove_arm = args.remove_arm
weight_decay = args.weight_decay
val_data_path = args.val_data_path
min_wvl = args.min_wvl
max_wvl = args.max_wvl


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_script(output_path, data_path, train_script, virtual_env, num_train, targets, spec_key,
                 save_folder, batch_size, epochs, sizes, lr, noise_addition, remove_gaps, remove_arm, weight_decay,
                 val_data_path, min_wvl, max_wvl):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    data_filename = os.path.basename(data_path)
    #data_path_slurmtmpdir = os.path.join(SLURM_TMPDIR, data_filename)

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.7\n')
        writer.write('source {}\n'.format(os.path.join(virtual_env, 'bin/activate')))

        #writer.write('python -c "from eniric import config; config.copy_file(\'.\')"\n')
        writer.write('cp -r {} {}\n'.format(os.path.join(HOME_DIR, 'StarNet'), '$SLURM_TMPDIR'))
        #writer.write('cp {} {}\n'.format(data_path, '$SLURM_TMPDIR'))
        writer.write('export PYTHONPATH="{}:{}"\n'.format(PYTHONPATH, os.path.join('$SLURM_TMPDIR', 'StarNet/')))

        #writer.write('cp {} {}/'.format(data_path, SLURM_TMPDIR))
        writer.write('\n')

        writer.write('python {}/StarNet/startorch/{} \\\n'.format('$SLURM_TMPDIR',
                                                                        os.path.basename(train_script)))
        writer.write('--data_path $SCRATCH/spectra/%s \\\n' % data_filename)
        writer.write('--num_train %s \\\n' % num_train)
        writer.write('--spec_key %s \\\n' % spec_key)
        writer.write('--targets')
        for target in targets:
            writer.write(' %s' % target)
        writer.write(' \\\n')
        writer.write('--layers')
        for layer in sizes:
            writer.write(' %s' % layer)
        writer.write(' \\\n')
        writer.write('--save_folder %s \\\n' % save_folder)
        writer.write('--batch_size %s \\\n' % batch_size)
        writer.write('--epochs %s \\\n' % epochs)
        if noise_addition:
            writer.write('--noise_addition %s \\\n' % noise_addition)
        if remove_gaps:
            writer.write('--remove_gaps %s \\\n' % remove_gaps)
        if remove_arm:
            writer.write('--remove_arm %s \\\n' % remove_arm)
        if weight_decay:
            writer.write('--weight_decay %s \\\n' % weight_decay)
        if val_data_path:
            writer.write('--val_data_path %s \\\n' % val_data_path)
        if min_wvl:
            writer.write('--min_wvl %s \\\n' % min_wvl)
        if max_wvl:
            writer.write('--max_wvl %s \\\n' % max_wvl)
        writer.write('--learning_rate %s' % lr)
        
for sample in range(20):

    num_layers = np.random.randint(2, 5)
    sizes = np.random.randint(0,400,(num_layers))
    #sizes = np.random.randint(0, 5000, (num_layers))
    #sizes = [300 * i + x for i, x in enumerate(sorted(random.sample(range(3000), num_layers)))]
    sizes[0] += 50
    for i in range(num_layers - 1):
        sizes[i + 1] += sizes[i]
    lr = np.random.uniform(1e-3, 1e-2)
    batch_size = 16 * np.random.randint(1, 4)

    name = 'n-{}'.format(num_layers)
    for i in range(num_layers):
        name = name + '_{}'.format(sizes[i])
    name = name + '_lr-{:.4f}_batchsize-{}'.format(lr, batch_size)
    config = [sizes, lr, batch_size]

    output_job_path = args.output_path
    output_job_folder = os.path.dirname(output_job_path)
    if not os.path.exists(output_job_folder):
        os.makedirs(output_job_folder)
    if not output_job_path.endswith('.sh'):
        output_job_path += '.sh'

    output_path = output_job_path[:-3] + '_{}.sh'.format(sample)

    new_save_folder = os.path.join(save_folder, name)
    write_script(output_path, data_path, train_script, virtual_env, num_train, targets, spec_key,
                 new_save_folder, batch_size, epochs, sizes, lr, str2bool(noise_addition),
                 str2bool(remove_gaps), str2bool(remove_arm), weight_decay,
                 val_data_path, min_wvl, max_wvl)

    # num_layers = np.random.randint(3,7)
    # sizes = np.random.randint(0,400,(num_layers))
    # sizes[0] += 50
    # for i in range(num_layers-1):
    #     sizes[i+1] += sizes[i]
    # lr = np.random.uniform(1e-3,1e-2)
    # batch_size = 64*np.random.randint(1,4)
    #
    # name = 'n-{}'.format(num_layers)
    # for i in range(num_layers):
    #     name = name + '_{}'.format(sizes[i])
    # name = name + '_lr-{:.4f}_batchsize-{}'.format(lr,batch_size)
    # config = [sizes,lr,batch_size]
    #
    # output_job_path = args.output_path
    # output_job_folder = os.path.dirname(output_job_path)
    # if not os.path.exists(output_job_folder):
    #     os.makedirs(output_job_folder)
    # if not output_job_path.endswith('.sh'):
    #     output_job_path += '.sh'
    #
    # output_path = output_job_path[:-3] + '_{}.sh'.format(sample)
    #
    # new_save_folder = os.path.join(save_folder, name)
    # write_script(output_path, data_path, train_script, virtual_env, num_train, targets, spec_key,
    # new_save_folder, batch_size, epochs, sizes, lr)


