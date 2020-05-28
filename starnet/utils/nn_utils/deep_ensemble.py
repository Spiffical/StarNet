import os
import sys
sys.path.insert(0, "{}/StarNet".format(os.getenv('HOME')))
import numpy as np
import glob
import csv
import json

from keras.models import load_model, Model
from starnet.utils.nn_utils.custom_layers import GaussianLayer


def denormalize_mu(data, mu=None, sigma=None):
    if mu is None or sigma is None:
        print('Need mu and sigma supplied to denormalize. Returning normalized data..')
        return data
    else:
        return (data * sigma) + mu


def denormalize_sigma(data, sigma=None):
    if sigma is None:
        print('Need sigma supplied to denormalize. Returning normalized data..')
        return data
    else:
        return data * sigma


def get_model_training_info(model_folder):
    # Load model parameters for a model
    model_parameter_filepath = os.path.join(model_folder, 'model_parameter.json')
    if os.path.exists(model_parameter_filepath):
        with open(model_parameter_filepath, 'r') as f:

            datastore = json.load(f)

            # Load values used to normalize the labels
            mu_label = np.asarray(datastore['mu'])
            sigma_label = np.asarray(datastore['sigma'])

            # Load the targets and spec_key (keywords used to load data from the training dataset)
            targets = datastore['targetname']
            spec_name = datastore['spec_key']
    else:
        raise ValueError('No parameter file found for model {}'.format(model_folder))

    return mu_label, sigma_label, targets, spec_name


def mad_based_outlier(points, thresh=3.5):
    """
    Identify outliers based on the median absolute deviation (MAD)
    :param points: data to identify outliers in
    :param thresh: MAD threshold
    :return: boolean array identifying outliers
    """
    if len(np.shape(points)) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def load_ensemble(ensemble_folder, maximum_loss_accepted=2.0, top_n=5):
    ensemble_models = glob.glob(ensemble_folder + '/*')
    prediction_fns = []
    mu_labels = []
    sigma_labels = []
    min_val_losses = []

    # Collect minimum validation losses for each model, and remove those models which don't have
    # losses less than the maximum accepted loss
    for model_folder in ensemble_models:
        # keras.backend.clear_session()

        # Check to see if the validation loss reached a low enough minimum (if not, skip model)
        training_log_path = os.path.join(model_folder, 'training.log')
        if os.path.exists(training_log_path):
            with open(training_log_path, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                val_losses = []
                for row in csv_reader:
                    val_losses.append(float(row['val_loss']))
                if not np.isnan(val_losses).all():  # Only find min if not all NaNs
                    min_val_loss = np.nanmin(val_losses)
                    if min_val_loss > maximum_loss_accepted:
                        print('Skipping model {} with minimum validation loss of {:.2f}'.format(
                            os.path.basename(model_folder),
                            min_val_loss))
                        ensemble_models.remove(model_folder)
                        continue
                    else:
                        min_val_losses.append(min_val_loss)
                else:
                    continue

    # Collect the n models with lowest validation score, where n is given by the parameter top_n
    print('Collecting the best {} models'.format(top_n))
    top_n_models, top_n_losses = zip(*sorted(zip(ensemble_models, min_val_losses), key=lambda t: t[1])[:top_n])

    for model_folder in top_n_models:

        # Load model, grab output from an intermediate layer (which outputs both the mu and sigma)
        print('Loading model: {}'.format(os.path.basename(model_folder)))
        model_path = os.path.join(model_folder, 'weights.best.h5')
        # with tf.device('/cpu:0'):
        model = load_model(model_path, custom_objects={'GaussianLayer': GaussianLayer}, compile=False)
        layer_name = 'main_output'  # Where to extract the output from
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.get_layer(layer_name).output)
        prediction_fns.append(intermediate_layer_model)

        # Get normalization mu and sigma used to normalize the labels for each trained model
        try:
            mu_label, sigma_label, _, _ = get_model_training_info(model_folder)
        except ValueError:
            mu_label, sigma_label = None, None
        mu_labels.append(mu_label)
        sigma_labels.append(sigma_label)

    print('The ensemble contains {} models with the following validation losses: \n{}'.format(len(prediction_fns),
                                                                                              top_n_losses))
    return prediction_fns, mu_labels, sigma_labels


def ensemble_predict(X, ensemble_folder=None, ensemble=None, mu_labels=None,
                     sigma_labels=None, maximum_loss_accepted=2.0, top_n=5, print_results=False):
    """
    Can either provide a master folder name (parameter: ensemble_folder) containing folders of trained models,
    or provide a pre-loaded set of models (parameter: ensemble), loaded with the function load_ensemble()
    """
    if ensemble is None:
        if ensemble_folder is None:
            raise ValueError('Need to define ensemble_folder if not using pre-loaded ensemble')
        else:
            print('Loading ensemble of models from folder {}...'.format(ensemble_folder))
            ensemble, mu_labels, sigma_labels = load_ensemble(ensemble_folder, maximum_loss_accepted, top_n=top_n)
    else:
        print('Using pre-loaded ensemble...')

    # Reshape data for compatibility with CNN layers
    if np.ndim(X) == 2:
        X = np.reshape(X, (np.shape(X)[0], np.shape(X)[1], 1))

    # Predict with multiple networks
    print('Predicting on the dataset with ensemble of {} trained models...'.format(len(ensemble)))
    mu_sigma = [(ensemble[i].predict(X)[0],
                 ensemble[i].predict(X)[1])
                for i in range(len(ensemble))]
    out_mus = [i for i, j in mu_sigma]
    out_sigmas = [j for i, j in mu_sigma]

    out_mus_denorm = np.squeeze([denormalize_mu(out_mus[i], mu_labels[i], sigma_labels[i])
                                 for i in range(len(out_mus))])
    out_sigmas_denorm = np.squeeze([denormalize_sigma(out_sigmas[i], sigma_labels[i])
                                    for i in range(len(out_sigmas))])

    if print_results:
        print(out_mus_denorm)

    # Remove outliers if any are detected
    for j in range(np.shape(out_mus_denorm)[1]):
        mus = out_mus_denorm[:, j]
        sigmas = out_sigmas_denorm[:, j]
        outliers_mu = mad_based_outlier(mus)
        outliers_sigma = mad_based_outlier(sigmas)

        for i, outlier in enumerate(outliers_mu):
            if outlier:
                out_mus_denorm[i, j] = np.nan
                out_sigmas_denorm[i, j] = np.nan
        for i, outlier in enumerate(outliers_sigma):
            if outlier:
                out_mus_denorm[i, j] = np.nan
                out_sigmas_denorm[i, j] = np.nan

    # Get final test result
    out_mus_denorm = np.ma.array(out_mus_denorm, mask=np.isnan(out_mus_denorm))
    out_sigmas_denorm = np.ma.array(out_sigmas_denorm, mask=np.isnan(out_sigmas_denorm))
    out_mu_final = np.mean(out_mus_denorm, axis=0)
    out_sig_final = np.sqrt(np.mean(out_sigmas_denorm + np.square(out_mus_denorm), axis=0) - np.square(out_mu_final))
    print('Done! You now have the final predictions with uncertainties.')

    return out_mu_final, out_sig_final
