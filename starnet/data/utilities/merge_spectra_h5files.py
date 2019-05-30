import os
import glob
import h5py
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True,
                    help='Path to batches of .h5 spectra files')
parser.add_argument('--output_path', type=str, required=True,
                    help='Location to save .h5 file')
parser.add_argument('--labels', nargs='+', default=['spectra_starnetnorm', 'spectra_statcont', 'teff',
                                                     'logg', 'M_H', 'a_M', 'v_rot', 'v_rad', 'VT', 'noise'],
                    help='Keys of h5py file to include')
args = parser.parse_args()

input_path = args.input_path
master_file = args.output_path
labels = args.labels

for i, file_path in enumerate(glob.glob(input_path + '/*')):
    filename = os.path.basename(file_path)

    # Turn into one master file
    if i % 10 == 0:
        print(i)
    try:
        with h5py.File(file_path, "r") as f:
            wav = f['wave_grid'][:]
            label_vals = [f[label][:] for label in labels]
    except:
        continue

    if not os.path.exists(master_file):
        print('Master file does not yet exist. Creating it at: {}'.format(master_file))
        with h5py.File(master_file, 'w') as hf:
            for j, label in enumerate(labels):
                if label_vals[j].ndim == 2:
                        maxshape = (None, label_vals[j].shape[1])
                else:
                        maxshape = (None,)
                hf.create_dataset(label, data=label_vals[j], maxshape=maxshape)
            hf.create_dataset('wave_grid', data=wav)
    else:
        with h5py.File(master_file, 'a') as hf:
            for j, label in enumerate(labels):
                hf[label].resize((hf[label].shape[0]) + label_vals[j].shape[0],
                                 axis=0)
                hf[label][-label_vals[j].shape[0]:] = label_vals[j]
    os.system('rm %s' % file_path)
