import os
import argparse

HOME_DIR = os.getenv('HOME')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of training file')
parser.add_argument('--data_path', type=str,
                    help='Path of training file')
parser.add_argument('--train_script', type=str, default=os.path.join(HOME_DIR, 'StarNet/starnet/train_StarNet.py'),
                    help='Path to training script')
parser.add_argument('--virtual_env', type=str, required=True,
                    help='Name of the virtual environment to use')
parser.add_argument('--num_train', type=int,
                    help='Size of training set')
parser.add_argument('--targets', nargs='+', required=True,
                    help='Keys of h5py file to train on')
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
parser.add_argument('--model_type', type=str, default='StarNet2017',
                    help='Type of model to train')
parser.add_argument('--ensemble', type=int, default=1,
                    help='Number of models to train in an ensemble')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='Number of GPUs available for training')
args = parser.parse_args()


def write_script(output_path, data_path, train_script, virtual_env, num_train, targets, save_folder, spec_key,
                 batch_size, epochs, zeros, telluric_file, finetune_model, num_gpu, model_type):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load python/3.6\n')
        writer.write('source {}\n'.format(os.path.join(HOME_DIR, virtual_env, 'bin/activate')))
        writer.write('\n\n')
        writer.write('python {} \\\n'.format(train_script))
        writer.write('--data_path %s \\\n' % data_path)
        writer.write('--num_train %s \\\n' % num_train)
        writer.write('--targets')
        for target in targets:
            writer.write(' %s' % target)
        writer.write(' \\\n')
        writer.write('--spec_key %s \\\n' % spec_key)
        writer.write('--save_folder %s \\\n' % save_folder)
        writer.write('--batch_size %s \\\n' % batch_size)
        writer.write('--epochs %s \\\n' % epochs)
        writer.write('--zeros %s \\\n' % zeros)
        writer.write('--telluric_file %s \\\n' % telluric_file)
        writer.write('--finetune_model %s \\\n' % finetune_model)
        writer.write('--num_gpu %s \\\n' % num_gpu)
        writer.write('--model_type %s' % model_type)


output_job_path = args.output_path
output_job_folder = os.path.dirname(output_job_path)
if not os.path.exists(output_job_folder):
    os.makedirs(output_job_folder)
if not output_job_path.endswith('.sh'):
    output_job_path += '.sh'

if args.ensemble > 1:
    for i in range(args.ensemble):
        save_folder_ensemble = os.path.join(args.save_folder, 'model{}'.format(i))
        output_path_ensemble = output_job_path[:-3] + '_{}.sh'.format(i)
        write_script(output_path_ensemble, args.data_path, args.train_script, args.virtual_env, args.num_train,
                     args.targets, save_folder_ensemble, args.spec_key, args.batch_size, args.epochs, args.zeros,
                     args.telluric_file, args.finetune_model, args.num_gpu, args.model_type)
else:
    write_script(output_job_path, args.data_path, args.train_script, args.virtual_env, args.num_train, args.targets,
                 args.save_folder, args.spec_key, args.batch_size, args.epochs, args.zeros, args.telluric_file,
                 args.finetune_model, args.num_gpu, args.model_type)
