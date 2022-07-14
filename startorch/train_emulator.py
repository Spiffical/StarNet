import torch
import torch.optim as optim
import numpy as np
import os
import sys
import argparse
from emulator_model import DNN
from emulator_utils import get_train_valid_loader, train_epoch_generator, val_epoch_generator
from utils import l1


def train_nn(config, num_train, data_path, targets, spec_key, save_folder, wavegrid_path, max_epochs):
    
    sizes,lr,batch_size = config
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        np.save(os.path.join(save_folder, 'config.npy'), np.asarray(config))

    train_loader, valid_loader, len_spec = get_train_valid_loader(data_path,
                                                        batch_size,
                                                        save_folder,
                                                        targets,
                                                        spec_key,
                                                        num_train,
                                                        valid_size=0.1,
                                                        wavegrid_path=wavegrid_path,
                                                        shuffle=True,
                                                        num_workers=10,
                                                        pin_memory=True)

    trainSteps = len(train_loader.dataset) // batch_size
    valSteps = len(valid_loader.dataset) // batch_size

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Initializing the emulator NN
    NN = DNN(sizes, len(targets), len_spec).to(device)

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

    weight_decay = 0#1e-4
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

    for epoch in range(max_epochs):
        sys.stdout.write('Epoch {}\n'.format(epoch))
        # Training epoch
        loss_train = train_epoch_generator(NN, training_generator=train_loader,
                                           optimizer=optimizer,
                                           device=device,
                                           train_steps=trainSteps,
                                           loss_fn=l1)
        loss_val = val_epoch_generator(NN, valid_generator=valid_loader,
                                       device=device,
                                       val_steps=valSteps,
                                       loss_fn=l1)
        scheduler.step(loss_val)

        sys.stdout.write('train_loss: {}, val_loss: {}\n'.format(loss_train,loss_val))
        # Saving results to txt file
        sys.stdout.write('Saving training losses to {}\n'.format(os.path.join(save_folder,'train_hist.txt')))
        with open((os.path.join(save_folder,'train_hist.txt')), 'a+') as f:
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
    parser.add_argument('--layers', nargs='+', required=True,
                        help='sizes of layers of NN')
    parser.add_argument('--save_folder', type=str, default=None,
                        help='Folder to save trained model in (if None, folder name created based on date)')
    parser.add_argument('-b', '--batch_size', type=int, default=16,
                        help='Number of spectra used in a single batch')
    parser.add_argument('-e', '--epochs', type=int, default=300,
                        help='Maximum number of epochs for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=2.3e-3,
                        help='Learning rate for NN')
    parser.add_argument('--wavegrid_path', type=str, default='',
                        help='Path of wavegrid (needed if wave_grid is not in data file)')

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    train_nn([np.array(args.layers, dtype=int), args.lr, args.batch_size],
             args.num_train,
             args.data_path,
             args.targets,
             args.spec_key,
             args.save_folder,
             args.wavegrid_path,
             args.max_epochs)


