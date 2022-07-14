import torch
from torch.nn import Linear, ReLU, Sequential, Conv1d, MaxPool1d, Module, MSELoss
from torch import flatten
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
import functools
import operator

sys.path.insert(0, "{}/StarNet".format(os.getenv('HOME')))
from startorch.utils import get_train_valid_loader


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class StarNet(Module):
    def __init__(self, in_channels=1, out_channels=4, input_dim=(1, 43480)):
        # call the parent constructor
        super(StarNet, self).__init__()

        self.feature_extractor = Sequential(
            # initialize first set of CONV => RELU => POOL layers
            Conv1d(in_channels=in_channels, out_channels=4, kernel_size=4, padding=0),
            ReLU(),
            # initialize second set of CONV => RELU => POOL layers
            Conv1d(in_channels=4, out_channels=16, kernel_size=4, padding=0),
            ReLU(),
            MaxPool1d(kernel_size=4, stride=1)
        )

        num_features_before_fcnn = functools.reduce(operator.mul,
                                                    list(self.feature_extractor(torch.rand(1, *input_dim)).shape))

        self.fc = Sequential(
            # initialize first set of FC => RELU layers
            Linear(in_features=num_features_before_fcnn, out_features=256),
            ReLU(),
            # initialize second set of FC => RELU layers
            Linear(in_features=256, out_features=128),
            ReLU(),
            Linear(in_features=128, out_features=out_channels)
        )

    def forward(self, x):

        out = self.feature_extractor(x)
        out = flatten(out, 1)
        out = self.fc(out)

        # return the output predictions
        return out


def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    is_inf = torch.isinf(v)
    v[is_nan] = 0
    v[is_inf] = 0
    return v.sum(*args, **kwargs) / (~(is_nan&is_inf)).float().sum(*args, **kwargs)


# Defining l2 loss
def l2(y_pred,y_true):
    return torch.sqrt(nanmean((y_pred-y_true)**2))


# Defining l2 loss
def l1(y_pred,y_true):
    return (nanmean(torch.abs(y_pred-y_true)))


# Function to get a batch of data
def get_batch(filepath,indices,targets,mean,std,batch_size,n):
    indices_batch = indices[n*batch_size: (n+1)*batch_size]
    x, y = load_data(filepath, indices_batch, targets)
    x = (x - mean)/std
    x = torch.from_numpy(x).to('cuda:0').float().view(-1,1,len(targets))
    y = torch.from_numpy(y).to('cuda:0').float().view(-1,1,43480)
    return x, y


# Function to execute a training epoch
def train_epoch_generator(NN,training_generator,optimizer,device,train_steps,loss_fn):

    NN.train()
    loss = 0
    # Passing the data through the NN
    for i, (labels, spectra) in enumerate(training_generator):
        #sys.stdout.write('{}\n'.format(i))

        x = spectra
        y = labels

        # Transfer to device
        x = x.to(device).float().view(-1, 1, np.shape(x)[1])
        y_true = y.to(device).float()

        # perform a forward pass and calculate loss
        y_pred = NN(x)
        batch_loss = loss_fn(y_pred, y_true)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # add batch loss to the total training loss
        loss += batch_loss

    #scheduler.step()
    #MSE = (loss / (batch_size * (n_train // batch_size))).detach().cpu().numpy()
    avgLoss = loss / train_steps
    avgLoss = avgLoss.detach().cpu().numpy()
    return avgLoss


# Function to execute a validation epoch
def val_epoch_generator(NN,valid_generator,device,val_steps,loss_fn):
    with torch.no_grad():
        NN.eval()
        loss = 0
        # Passing the data through the NN
        for labels, spectra in valid_generator:
            x = spectra
            y = labels

            # Transfer to device
            x = x.to(device).float().view(-1, 1, np.shape(x)[1])
            y_true = y.to(device).float()

            y_pred = NN(x)
            loss += loss_fn(y_pred, y_true)#*batch_size
        #MSE = (loss/(batch_size*(n_val//batch_size))).detach().cpu().numpy()
        avgLoss = loss / val_steps
        avgLoss = avgLoss.detach().cpu().numpy()

        return avgLoss


def train_NN(lr, batch_size, num_train, data_path, targets, spec_key, save_folder, max_epochs, noise_addition,
             remove_gaps, remove_arm, weight_decay, val_data_path, min_wvl, max_wvl, starlink_mode):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        with open((os.path.join(save_folder, 'config.txt')), 'w') as f:
            f.write('{}\n'.format(data_path))
            f.write('{}\n'.format(targets))
            f.write('{}\n'.format(spec_key))
            f.write('{}\n'.format(noise_addition))
            f.write('{}\n'.format(remove_gaps))
            f.write('{}'.format(remove_arm))

    train_loader, valid_loader, len_spec = get_train_valid_loader(data_path,
                                                                  batch_size,
                                                                  save_folder,
                                                                  targets,
                                                                  spec_key,
                                                                  num_train,
                                                                  valid_size=0.1,
                                                                  shuffle=True,
                                                                  num_workers=10,
                                                                  pin_memory=True,
                                                                  remove_gaps=remove_gaps,
                                                                  remove_arm=remove_arm,
                                                                  noise_addition=noise_addition,
                                                                  val_data_path=val_data_path,
                                                                  min_wvl=min_wvl,
                                                                  max_wvl=max_wvl,
                                                                  starlink_mode=starlink_mode)

    trainSteps = len(train_loader.dataset) // batch_size
    valSteps = len(valid_loader.dataset) // batch_size

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Initializing the startorch model
    NN = StarNet(1, len(targets), (1, len_spec))

    # Load in previous best model weights if they were saved
    bestmodel_path = os.path.join(save_folder, 'model_best.pth')
    loss_val_min = 9999
    if os.path.exists(bestmodel_path):
        sys.stdout.write('Model aleady exists! Continuing from saved file: {}\n'.format(bestmodel_path))
        NN.load_state_dict(torch.load(bestmodel_path))
        loss_hist = np.loadtxt(os.path.join(save_folder, 'train_hist.txt'), dtype=float, delimiter=',')
        dim = len(np.shape(loss_hist))
        if dim==0:
            sys.stdout.write('No training history!\n')
        elif dim==1:
            loss_val_min = loss_hist[1]
        else:
            loss_val_hist = loss_hist[:, 1]
            loss_val_min = min(loss_val_hist)

    #weight_decay = 0
    # Setting up optimizer and learning rates
    #optimizer = optim.Adam([{'params': NN.parameters(), 'lr': lr}, 'weight_decay': weight_decay])
    optimizer = optim.Adam(NN.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     min_lr=0.000002,
                                                     eps=1e-08)

    NN.cuda()
    loss_fn = MSELoss()

    for epoch in range(max_epochs):
        sys.stdout.write('Epoch {}\n'.format(epoch))
        # Training epoch
        loss_train = train_epoch_generator(NN, training_generator=train_loader,
                                           optimizer=optimizer,
                                           device=device,
                                           train_steps=trainSteps,
                                           loss_fn=loss_fn)
        loss_val = val_epoch_generator(NN, valid_generator=valid_loader,
                                       device=device,
                                       val_steps=valSteps,
                                       loss_fn=loss_fn)
        scheduler.step(loss_val)

        sys.stdout.write('train_loss: {}, val_loss: {}\n'.format(loss_train,loss_val))
        # Saving results to txt file
        sys.stdout.write('Saving training losses to {}\n'.format(os.path.join(save_folder,'train_hist.txt')))
        with open((os.path.join(save_folder, 'train_hist.txt')), 'a+') as f:
            f.write('{}, '.format(loss_train))
            f.write('{}'.format(loss_val))
            f.write('\n')
        
        if loss_val < loss_val_min:
            loss_val_min = loss_val
            sys.stdout.write('Saving best model with validation loss of {}\n'.format(loss_val))
            torch.save(NN.state_dict(), bestmodel_path)
        
################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path of training file')
    parser.add_argument('--num_train', type=int, required=True,
                        help='Size of training set')
    parser.add_argument('--targets', nargs='+', required=True,
                        help='Keys of h5py file to train on')
    parser.add_argument('--spec_key', type=str, required=True,
                        help='h5py key for spectra')
    parser.add_argument('--layers', nargs='+', required=False,
                        help='sizes of layers of NN')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Folder to save trained model in (if None, folder name created based on date)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of spectra used in a single batch')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='Maximum number of epochs for training')
    parser.add_argument('-g', '--num_gpu', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0007,
                        help='Learning rate for NN')
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
    parser.add_argument('--starlink_mode', type=str, default='False',
                        help='Whether to randomly select a solar-contaminated spectrum')

    args = parser.parse_args()

    data_path = args.data_path
    num_train = args.num_train
    targets = args.targets
    spec_key = args.spec_key
    save_folder = args.save_folder
    max_epochs = args.epochs
    batch_size = args.batch_size
    num_gpu = args.num_gpu
    lr = args.learning_rate
    noise_addition = str2bool(args.noise_addition)
    remove_gaps = str2bool(args.remove_gaps)
    remove_arm = str2bool(args.remove_arm)
    weight_decay = args.weight_decay
    val_data_path = args.val_data_path
    min_wvl = args.min_wvl
    max_wvl = args.max_wvl
    starlink_mode = str2bool(args.starlink_mode)

    torch.multiprocessing.set_start_method('spawn')

    train_NN(lr, batch_size, num_train,
             data_path,
             targets, spec_key,
             save_folder,
             max_epochs,
             noise_addition,
             remove_gaps,
             remove_arm,
             weight_decay,
             val_data_path,
             min_wvl,
             max_wvl,
             starlink_mode)


