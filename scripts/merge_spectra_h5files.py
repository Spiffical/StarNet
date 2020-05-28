import os
import glob
import h5py
import warnings
import argparse
import time
import numpy as np

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True,
                    help='Path to batches of .h5 spectra files')
parser.add_argument('--output_path', type=str, required=True,
                    help='Location to save .h5 file')
parser.add_argument('--wave_grid_key', type=str, default='wave_grid',
                    help='Wave grid key in h5 files being merged')
args = parser.parse_args()

input_path = args.input_path
master_file = args.output_path
wave_grid_key = args.wave_grid_key
num_spectra_merged = 0

for i, file_path in enumerate(glob.glob(input_path + '/*')):
    filename = os.path.basename(file_path)

    try:
        # Collect data from temporary h5 file
        with h5py.File(file_path, "r") as f:
            # Collect h5 labels (acquiring the wave grid separately)
            labels = list(f.keys())
            labels.remove(wave_grid_key)
            wav = f[wave_grid_key][:]
            label_vals = [f[label][:] for label in labels]

        # Add data to master file
        if not os.path.exists(master_file):
            print('Master file does not yet exist. Creating it at: {}'.format(master_file))
            with h5py.File(master_file, 'w') as hf:
                for j, label in enumerate(labels):
                    if label_vals[j].ndim == 2:
                        maxshape = (None, label_vals[j].shape[1])
                    else:
                        maxshape = (None,)
                    hf.create_dataset(label, data=label_vals[j], maxshape=maxshape)
                hf.create_dataset(wave_grid_key, data=wav)
        else:
            with h5py.File(master_file, 'a') as hf:
                for j, label in enumerate(labels):
                    hf[label].resize((hf[label].shape[0]) + label_vals[j].shape[0],
                                     axis=0)
                    hf[label][-label_vals[j].shape[0]:] = label_vals[j]

        num_spectra_merged += len(label_vals[0])
        if i % 10 == 0:
            print('Number of spectra merged: {}'.format(num_spectra_merged))
    except:
        print('Problem with file %s, deleting...' % filename)
        os.system('rm %s' % file_path)
        continue

    # Delete file
    os.system('rm %s' % file_path)

#
# input_path = args.input_path
# master_file = args.output_path
# wave_grid_key = args.wave_grid_key
# num_files = len(glob.glob(input_path + '/*'))
#
# h5_params = np.array([])
# for i, file_path in enumerate(glob.glob(input_path + '/*')):
#     filename = os.path.basename(file_path)
#
#     # Turn into one master file
#     if i % 10 == 0:
#         print(i)
#     with h5py.File(file_path, "r") as f:
#         # Collect h5 labels (acquiring the wave grid separately)
#         labels = list(f.keys())
#         labels.remove(wave_grid_key)
#         print(labels)
#         wav = f[wave_grid_key][:]
#         label_vals = [f[label][:] for label in labels]
#         # Append values to placeholder list
#         h5_params = np.stack([h5_params, label_vals], -1) if np.size(h5_params) else np.array(label_vals)
#         if h5_params.ndim == 2: print(h5_params[0,:])
#
#     # Fill up placeholder array with >100 entries and then dump them into the h5 file
#     num_entries = np.shape(h5_params)[1] if h5_params.ndim == 2 else 1
#     if num_entries >= 100 or i == num_files-1:
#         if not os.path.exists(master_file):
#             print('Master file does not yet exist. Creating it at: {}'.format(master_file))
#             with h5py.File(master_file, 'w') as hf:
#                 for j, label in enumerate(labels):
#                     if h5_params[j].ndim == 2:
#                         maxshape = (None, h5_params[j].shape[1])
#                     else:
#                         maxshape = (None,)
#                     hf.create_dataset(label, data=h5_params[j], maxshape=maxshape)
#                 hf.create_dataset(wave_grid_key, data=wav)
#         else:
#             print('Appending data from {} files to {}'.format(np.shape(h5_params)[1], master_file))
#             with h5py.File(master_file, 'a') as hf:
#                 for j, label in enumerate(labels):
#                     hf[label].resize((hf[label].shape[0]) + h5_params[j].shape[0],
#                                      axis=0)
#                     hf[label][-h5_params[j].shape[0]:] = h5_params[j]
#         # Empty the placeholder list after dumping data
#         h5_params = np.array([])
#
#     # Delete file
#     os.system('rm %s' % file_path)
