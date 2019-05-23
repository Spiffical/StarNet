<img src="https://github.com/Spiffical/StarNet/blob/master/full_logo.png" alt="drawing" width="600"/>

Harnessing the power of deep learning to accurately and efficiently predict crucial properties of the stars in our galaxy.


## Training StarNet

A script has been provided [`starnet/train_StarNet.py`](https://github.com/Spiffical/StarNet/blob/master/starnet/train_StarNet.py) to make training StarNet simple. An example of how to use it to train the basic StarNet2017 architecture on a dataset named `training_dataset.h5` to predict the parameters `[T_eff, logg, [M/H], [alpha/M]`:

```
python /path/to/starnet/train_StarNet.py \
--data_path /path/to/training_dataset.h5 \
--num_train 50000 \
--targets teff logg M_H a_M \
--spec_key spectra_starnetnorm \
--save_folder results_stored_here/ \
--batch_size 32 \
--epochs 35 \
--model_to_train StarNet2017
```

NOTE: the `--targets` and `--spec_key` arguments expect the keywords that are used in your `training_dataset.h5` file to store the training labels and training features, respectively.




