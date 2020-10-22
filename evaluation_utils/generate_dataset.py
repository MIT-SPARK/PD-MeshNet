import argparse
import os
import yaml

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.utils import create_dataset, compute_mean_and_std

training_parameters_filename = 'training_parameters.yml'


def generate_dataset(train_dataset_parameters, test_dataset_parameters,
                     data_loader_parameters):
    if (data_loader_parameters['validation_set_fraction'] is not None):
        assert (isinstance(data_loader_parameters['validation_set_fraction'],
                           float))
        assert (0.0 <= data_loader_parameters['validation_set_fraction'] <= 1.0)
    if ('compute_node_feature_stats' in train_dataset_parameters and
            not train_dataset_parameters['compute_node_feature_stats']):
        train_dataset, _ = create_dataset(**train_dataset_parameters)
        primal_mean = primal_std = dual_mean = dual_std = None
        if (data_loader_parameters['validation_set_fraction'] is not None):
            # Split the training set into a part that will be actually used for
            # training and another that will be used for validation.
            num_samples_train = int(
                len(train_dataset) *
                (1. - data_loader_parameters['validation_set_fraction']))
    else:
        whole_training_set_is_used_for_training = True
        if (data_loader_parameters['validation_set_fraction'] is not None):
            dataset_parameters_no_mean_std = train_dataset_parameters.copy()
            dataset_parameters_no_mean_std['compute_node_feature_stats'] = False
            train_dataset, _ = create_dataset(**dataset_parameters_no_mean_std)
            # Split the training set into a part that will be actually used for
            # training and another that will be used for validation.
            num_samples_train = int(
                len(train_dataset) *
                (1. - data_loader_parameters['validation_set_fraction']))
            if (num_samples_train < len(train_dataset)):
                whole_training_set_is_used_for_training = False

        if (not whole_training_set_is_used_for_training):
            print("\033[93mNote: it was requested that the training set be "
                  "be split in a training part ("
                  f"{num_samples_train}/{len(train_dataset)} samples) "
                  "and a validation part ("
                  f"{len(train_dataset) - num_samples_train}/"
                  f"{len(train_dataset)}).\033[00m")
            assert ('node_feature_stats_filename' in train_dataset_parameters
                    and train_dataset_parameters['node_feature_stats_filename']
                    is not None)
            stats_filename = train_dataset_parameters[
                'node_feature_stats_filename']

            # Compute mean/std on the part that will be used for training.
            # - Store the range of samples actually used for training and hence
            #   for computing mean/standard deviation.
            dataset_params_for_mean_and_std = train_dataset.input_parameters
            dataset_params_for_mean_and_std['sample_idx_start'] = 0
            dataset_params_for_mean_and_std[
                'sample_idx_end'] = num_samples_train - 1
            (primal_mean, primal_std, dual_mean,
             dual_std) = compute_mean_and_std(
                 dataset=train_dataset[:num_samples_train],
                 dataset_params=dataset_params_for_mean_and_std,
                 filename=stats_filename)
        else:
            train_dataset, (primal_mean, primal_std, dual_mean,
                            dual_std) = create_dataset(
                                **train_dataset_parameters)
    # It is supposed that a training split is used for training.
    assert (train_dataset.input_parameters['train'] is True)
    # Create the test dataset if necessary.
    test_dataset, _ = create_dataset(**test_dataset_parameters)
    assert (test_dataset.input_parameters['train'] is False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f',
        type=str,
        help=
        "Path to the folder containing the training- and dataset parameters.",
        required=True)
    args = parser.parse_args()

    training_job_folder = os.path.abspath(args.f)
    assert (os.path.exists(training_job_folder)
           ), f"Could not find the training job folder {training_job_folder}."
    log_folder = os.path.dirname(training_job_folder)
    training_job_name = os.path.basename(training_job_folder)

    # Load train-dataset, test-dataset and data-loader parameters.
    parameter_filename = os.path.join(training_job_folder,
                                      training_parameters_filename)
    try:
        with open(parameter_filename, 'rb') as f:
            previous_training_parameters = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        raise IOError("Unable to open previous-training-parameter file at "
                      f"'{parameter_filename}'. Exiting.")
    train_dataset_parameters = previous_training_parameters[
        'dataset_parameters']
    test_dataset_parameters = previous_training_parameters[
        'test_dataset_parameters']
    test_dataset_parameters['compute_node_feature_stats'] = False
    data_loader_parameters = previous_training_parameters[
        'data_loader_parameters']

    # Generate the dataset.
    generate_dataset(train_dataset_parameters=train_dataset_parameters,
                     test_dataset_parameters=test_dataset_parameters,
                     data_loader_parameters=data_loader_parameters)
