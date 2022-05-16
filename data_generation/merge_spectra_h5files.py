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
                    help='Path (including filename to save it to) of the saved file')
parser.add_argument('--wave_grid_key', type=str, default='wave_grid',
                    help='Wave grid key in h5 files being merged')
args = parser.parse_args()

input_path = args.input_path
master_file = args.output_path
wave_grid_key = args.wave_grid_key
num_spectra_merged = 0
total_num_spectra = 0

for i, file_path in enumerate(glob.glob(input_path + '/*')):
    filename = os.path.basename(file_path)

    #try:
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
                total_num_spectra = (hf[label].shape[0]) + label_vals[j].shape[0]

    num_spectra_merged += len(label_vals[0])
    if i % 10 == 0:
        print('Number of spectra merged: {}'.format(num_spectra_merged))
    #except:
    #    print('Problem with file %s, deleting...' % filename)
    #    os.system('rm %s' % file_path)
    #    continue

    # Delete file
    os.system('rm %s' % file_path)

new_filename = '{}_{}{}'.format(os.path.splitext(master_file)[0],
                               str(total_num_spectra),
                               os.path.splitext(master_file)[1])
os.system('mv {} {}'.format(master_file, new_filename))
