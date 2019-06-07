import csv
import os
import warnings
import numpy as np
from keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K


class CustomModelCheckpoint(Callback):
    """
    Modified version of Keras ModelCheckpoint()
    This function now reads any available training log for the validation loss 
    history and acquires the lowest loss. When fine-tuning or re-training a model, 
    this is important so that the best model weights are not overwritten when this 
    function is initialized.
    
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(CustomModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.save_dir = os.path.dirname(filepath)

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                
        # If there is already a training log, collect best val_loss so far
        training_log_path = os.path.join(self.save_dir, 'training.log')
        if os.path.exists(training_log_path):
            print('(ModelCheckpoint) Best validation loss being collected from: %s' % training_log_path)
            with open(training_log_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                val_losses = []
                for row in csv_reader:
                    val_losses.append(float(row['val_loss']))
                if not np.isnan(val_losses).all():  # Don't try to find min if all NaNs
                    self.best = np.nanmin(val_losses)
                
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        num_outputs = len(self.model.outputs)
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            #self.model.layers[-(num_outputs + 1)].save_weights(filepath, overwrite=True)
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            #self.model.layers[-(num_outputs + 1)].save(filepath, overwrite=True)
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    #self.model.layers[-(num_outputs + 1)].save_weights(filepath, overwrite=True)
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    #self.model.layers[-(num_outputs + 1)].save(filepath, overwrite=True)
                    self.model.save(filepath, overwrite=True)


class VirutalCSVLogger(Callback):
    """
    NAME: VirutalCSVLogger
    PURPOSE:
        A modification of keras' CSVLogger, but not actually write a file
    INPUT:
        No input for users
    OUTPUT:
        Output tensor
    HISTORY:
        2018-Feb-22 - Written - Henry Leung (University of Toronto)
        2018-Mar-12 - Update - Henry Leung (University of Toronto)
    """

    def __init__(self, filename='training_history.csv', separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        self.epoch = []
        self.history = {}
        #super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

    def savefile(self, folder_name=None):
        if folder_name is not None:
            full_path = os.path.normpath(folder_name)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
            self.filename = os.path.join(full_path, self.filename)

        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.keys = sorted(self.history.keys())

        self.writer = csv.DictWriter(self.csv_file, fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
        if self.append_header:
            self.writer.writeheader()

        for i in self.epoch:
            d = {'epoch': self.epoch[i]}
            d2 = dict([(k, self.history[k][i]) for k in self.keys])
            d.update(d2)
            self.writer.writerow(d)
        self.csv_file.close()
        
        
class CustomReduceLROnPlateau(Callback):
    """Modfied version of Keras ReduceLROnPlateau()
    This function now reads any available training log for the validation loss 
    history and acquires the lowest loss. When fine-tuning or re-training a model, 
    this is important so that the best model weights are not overwritten when this 
    function is initialized.
    # Example
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        min_delta: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 verbose=0, mode='auto', min_delta=1e-4, cooldown=0, min_lr=0,
                 log_path=None, **kwargs):
        super(CustomReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        if 'epsilon' in kwargs:
            min_delta = kwargs.pop('epsilon')
            warnings.warn('`epsilon` argument is deprecated and '
                          'will be removed, use `min_delta` instead.')
        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()
        
        # If there is already a training log, collect best val_loss so far
        training_log_path = os.path.join(log_path, 'training.log')
        if os.path.exists(training_log_path):
            print('(ReduceLR) Best validation loss being collected from: %s' % training_log_path)
            with open(training_log_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                val_losses = []
                for row in csv_reader:
                    val_losses.append(float(row['val_loss']))
                if not np.isnan(val_losses).all():  # Don't try to find min if all NaNs
                    self.best = np.nanmin(val_losses)
                
    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                print('Reduce LR wait: %s/%s' % (self.wait, self.patience))
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing '
                                  'learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                        
    def in_cooldown(self):
        return self.cooldown_counter > 0

