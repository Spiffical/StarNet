import os
import argparse

HOME_DIR = os.getenv('HOME')
SCRATCH_DIR = os.getenv('SCRATCH')
#SLURM_TMPDIR = os.getenv('SLURM_TMPDIR')
PYTHONPATH = os.getenv('PYTHONPATH')

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='./jobs/todo/submit_job.sh',
                    help='Path of training file')
parser.add_argument('--data_path', type=str,
                    help='Path of training file')
parser.add_argument('--val_data_path', type=str, default='',
                    help='(optional) path of validation data set')
parser.add_argument('--train_script', type=str, default=os.path.join(HOME_DIR, 'StarNet/scripts/train_StarNet.py'),
                    help='Path to training script')
parser.add_argument('--virtual_env', type=str, required=True,
                    help='Path to virtual environment to use')
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
parser.add_argument('--global_mask', type=str, default=None,
                    help='Path of numpy file which contains a mask to apply to all spectra')
parser.add_argument('--masks', type=str, default=None,
                    help='Path of numpy file which contains a list of masks to apply')
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
parser.add_argument('--hardmine', type=bool, default=False,
                    help='Whether or not to use hard mining')
parser.add_argument('--filter_len', type=int, default=8,
                    help='Filter length for the CNN layers')
args = parser.parse_args()


def write_script(output_path, data_path, train_script, virtual_env, num_train, targets, save_folder, spec_key,
                 batch_size, epochs, zeros, telluric_file, finetune_model, num_gpu, model_type, val_data_path,
                 hardmine, global_mask_path, masks_path):

    if not output_path.endswith('.sh'):
        output_path += '.sh'

    data_filename = os.path.basename(data_path)
    #data_path_slurmtmpdir = os.path.join(SLURM_TMPDIR, data_filename)

    print('Writing file to {}'.format(output_path))
    with open(output_path, 'w') as writer:
        writer.write('#!/bin/bash\n')
        writer.write('module load StdEnv/2018.3 python/3.6 cuda cudnn\n')
        writer.write('source {}\n'.format(os.path.join(virtual_env, 'bin/activate')))

        writer.write('python -c "from eniric import config; config.copy_file(\'.\')"\n')
        writer.write('cp -r {} {}\n'.format(os.path.join(HOME_DIR, 'StarNet'), '$SLURM_TMPDIR'))
        writer.write('cp {} {}\n'.format(data_path, '$SLURM_TMPDIR'))
        writer.write('export PYTHONPATH="{}:{}"\n'.format(PYTHONPATH, os.path.join('$SLURM_TMPDIR', 'StarNet/')))

        #writer.write('cp {} {}/'.format(data_path, SLURM_TMPDIR))
        writer.write('\n')

        writer.write('python {}/StarNet/starnet/scripts/{} \\\n'.format('$SLURM_TMPDIR',
                                                                        os.path.basename(train_script)))
        writer.write('--data_path $SLURM_TMPDIR/%s \\\n' % data_filename)
        if val_data_path != '':
            writer.write('--val_data_path %s \\\n' % val_data_path)
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
        writer.write('--verbose 2 \\\n')
        writer.write('--hardmine %s \\\n' % str(hardmine))
        writer.write('--global_mask %s \\\n' % global_mask_path)
        writer.write('--masks %s \\\n' % masks_path)
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
                     args.telluric_file, args.finetune_model, args.num_gpu, args.model_type, args.val_data_path,
                     args.hardmine, args.global_mask, args.masks)
else:
    write_script(output_job_path, args.data_path, args.train_script, args.virtual_env, args.num_train, args.targets,
                 args.save_folder, args.spec_key, args.batch_size, args.epochs, args.zeros, args.telluric_file,
                 args.finetune_model, args.num_gpu, args.model_type, args.val_data_path, args.hardmine,
                 args.global_mask, args.masks)
