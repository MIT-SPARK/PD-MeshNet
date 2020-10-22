import argparse
import os
import yaml

from pd_mesh_net.utils import BaseTestJob

training_parameters_filename = 'training_parameters.yml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f',
        type=str,
        help="Path to the folder containing the pretrained model to evaluate "
        "and the training parameters.",
        required=True)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help="Batch size to use for evaluation (does not affect results).")
    parser.add_argument('--verbose',
                        help="If passed, will display verbose prints.",
                        action='store_true')
    parser.add_argument(
        '--save_clusters',
        help="If passed, will save clusterized/segmented meshes.",
        action='store_true')
    args = parser.parse_args()

    training_job_folder = os.path.abspath(args.f)
    assert (os.path.exists(training_job_folder)
           ), f"Could not find the training job folder {training_job_folder}."
    log_folder = os.path.dirname(training_job_folder)
    training_job_name = os.path.basename(training_job_folder)

    batch_size = args.batch_size

    # Load test-dataset parameters.
    parameter_filename = os.path.join(training_job_folder,
                                      training_parameters_filename)
    try:
        with open(parameter_filename, 'rb') as f:
            previous_training_parameters = yaml.load(f, Loader=yaml.FullLoader)
    except IOError:
        raise IOError("Unable to open previous-training-parameter file at "
                      f"'{parameter_filename}'. Exiting.")
    test_dataset_parameters = previous_training_parameters[
        'test_dataset_parameters']
    test_dataset_parameters['compute_node_feature_stats'] = False
    data_loader_parameters = {'batch_size': batch_size}
    task_type = previous_training_parameters['loss_parameters']['task_type']

    # Create test job.
    test_job = BaseTestJob(dataset_parameters=test_dataset_parameters,
                           data_loader_parameters=data_loader_parameters,
                           log_folder=log_folder,
                           training_job_name=training_job_name,
                           task_type=task_type,
                           save_clusterized_meshes=args.save_clusters)

    # Run test job.
    test_job.test()