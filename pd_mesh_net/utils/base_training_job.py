from datetime import datetime
import glob
import inspect
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
import yaml

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.utils import (create_dataset, create_loss, create_lr_scheduler,
                               create_model, create_optimizer,
                               compute_num_correct_predictions,
                               compute_mean_and_std,
                               get_epoch_most_recent_checkpoint,
                               get_epoch_and_batch_most_recent_checkpoint)

training_parameters_filename = 'training_parameters.yml'


class BaseTrainingJob():
    r"""Base class for training jobs. Implements methods to automatically save
    input training parameters and values of tensors and parameters at
    intermediate steps of training.

    Args:
        log_folder (str): Path of the folder where the training logs, parameters
            and checkpoints will be saved. The folder is created if it does not
            exists.
        final_training_epoch (int): Final epoch of training. If training does
            not resume from a previous training job, the network is trained for
            all the epochs with indices between 1 and `final_training_epoch`
            included. If training resumes from a previous training job, the last
            checkpoint of which was saved at epoch `last_previous_epoch`, the
            network is either: trained for all epochs with indices between
            `last_previous_epoch` + 1 and `final_training_epoch` - if
            `last_previous_epoch` < `final_training_epoch` -, or not trained -
            if `last_previous_epoch` >= `final_training_epoch`.
        network_parameters (dict, optional): Dictionary containing the
            parameters of the network on which the training job will be
            executed. Can be omitted if training is resumed from a previous
            checkpoint (cf. argument `training_job_name`).
            (default: :obj:`None`)
        dataset_parameters (dict, optional): Dictionary containing the
            parameters of the dataset from which the training data will be
            extracted. Can be omitted if training is resumed from a previous
            checkpoint (cf. argument `training_job_name`).
            (default: :obj:`None`)
        test_dataset_parameters (dict, optional): Dictionary containing the
            parameters of the dataset on which the trained model will be tested
            at the end of each training epoch if this argument is None. Can be
            omitted if training is resumed from a previous checkpoint (cf.
            argument `training_job_name`) or if you wish not to evaluate test
            accuracy at the end of each epoch. (default: :obj:`None`)
        data_loader_parameters (dict, optional): Dictionary containing the
            parameters of the data loader that will input samples from the
            dataset to the network. Can be omitted if training is resumed from a
            previous checkpoint (cf. argument `training_job_name`).
            (default: :obj:`None`)
        optimizer_parameters (dict, optional): Dictionary containing the
            parameters of the optimizer that will be used for training. Can be
            omitted if training is resumed from a previous checkpoint (cf.
            argument `training_job_name`). (default: :obj:`None`)
        lr_scheduler_parameters (dict, optional): Dictionary containing the
            parameters of the optional learning-rate scheduler that will be used
            to change the learning rate over the epochs. If omitted or `None`,
            no learning-rate scheduler will be instantiated, unless training is
            resumed from a previous training job in which a learning-rate
            scheduler was defined (cf. argument `training_job_name`).
            (default: :obj:`None`)
        loss_parameters (dict, optional): Dictionary containing the parameters
            of the loss that will be used in training. Can be omitted if
            training is resumed from a previous checkpoint (cf. argument
            `training_job_name`). (default: :obj:`None`)
        checkpoint_epoch_frequency (int, optional): If not None, frequency for
            saving the network-weight and optimizer-variable checkpoints. In
            particular, a checkpoint is saved whenever the current epoch index
            is a multiple of `checkpoint_epoch_frequency`. Exactly one of this
            and the argument `checkpoint_batch_frequency` must be non-None.
            (default::obj: `None`)
        checkpoint_batch_frequency (int, optional): If not None, frequency for
            saving the network-weight and optimizer-variable checkpoints. In
            particular, a checkpoint is saved is the batch index is a multiple
            of `checkpoint_batch_frequency`. Exactly one of this and the
            argument `checkpoint_epoch_frequency` must be non-None.
            (default::obj: `None`)
        minibatch_summary_writer_frequency (int, optional): Frequency for
            writing a TensorBoard summary. In particular, a data point is
            written in the summary whenever the index of the current minibatch
            is a multiple of `minibatch_summary_writer_frequency`. The only
            exception is for validation loss and test accuracy, for which the
            summary - in case a validation set and the test-dataset parameteres
            are provided, respectively - is written at the end of every epoch.
            (default: :obj:`10`)
        training_job_name (str, optional): Optional name of the training job. If
            not passed, the job will automatically be assigned as name the
            current time, in the format {YYYYMMDD_hhmmss}, with hh in 24-hour
            format. The logs associated to the training job are saved in a
            subfolder with this name within the log folder (cf. argument
            `log_folder`). If the subfolder is already existent, it will be
            assumed that the training job should be continued from a previous
            checkpoint. Specifically, the subfolder should contain the following
            files/subfolders:
            - A file with filename :obj:`training_parameters_filename`,
              containing the input parameters of the training job;
            - A subfolder 'checkpoints/', containing the checkpoints with the
              network weights and the optimizer state variables;
            (default: :obj:`None`)
            - A subfolder 'tensorboard/', containing the TensorBoard summaries
              associated to training job.
        use_gpu (bool, optional): If True, the parameters from the checkpoint of
            a previous training job are loaded in the GPU and the network
            tensors are moved to the GPU; otherwise, parameters are loaded on
            the CPU, and network tensors are moved to the CPU.
            (default: :obj:`True`)
        compute_area_weighted_accuracy (bool, optional): If True, if the task is
            of segmentation type (cf. argument `task_type`), and if a test set
            is provided for computation of test accuracy (cf. argument
            `test_dataset_parameters`), computes also a version of the accuracy
            where the contribution of each face is weighted by its area (cf.
            `pd_mesh_net.utils.losses.compute_num_correct_predictions`).
            Otherwise, the argument is ignored. (default: :obj:`True`)
        verbose (bool, optional): If True, displays status prints with a higher
            level of verbosity. (default: :obj:`False`)

    Attributes:
        training_parameters (dict): Dictionary having as keys dictionaries
            representing the parameters of the network, the dataset, the data
            loader, the optimizer, the learning-rate scheduler and the loss used
            in the training job. In particular, the keys in the dictionary are
            `'network_parameters'`, `'dataset_parameters'`,
            `test_dataset_parameters`, `'data_loader_parameters'`,
            `'optimizer_parameters'`, `'lr_scheduler_parameters'` and
            `'loss_parameters'`.
    """

    def __init__(self,
                 log_folder,
                 final_training_epoch,
                 network_parameters=None,
                 dataset_parameters=None,
                 test_dataset_parameters=None,
                 data_loader_parameters=None,
                 optimizer_parameters=None,
                 lr_scheduler_parameters=None,
                 loss_parameters=None,
                 checkpoint_epoch_frequency=None,
                 checkpoint_batch_frequency=None,
                 minibatch_summary_writer_frequency=10,
                 training_job_name=None,
                 use_gpu=True,
                 compute_area_weighted_accuracy=True,
                 verbose=False):
        self.__net = None
        self.__data_loader_train = None
        self.__data_loader_validation = None
        self.__data_loader_test = None
        self.__optimizer = None
        self.__lr_scheduler = None
        self.__loss = None
        self.__log_folder = log_folder
        self.__final_training_epoch = final_training_epoch
        self.__use_gpu = use_gpu
        self.__compute_area_weighted_accuracy = compute_area_weighted_accuracy
        self.__verbose = verbose

        # Initialize logging.
        if (self.__verbose):
            print("Initializing logging...")
        assert (
            (checkpoint_batch_frequency is None) !=
            (checkpoint_epoch_frequency is None)), (
                "Exactly one of the arguments 'checkpoint_batch_frequency' and "
                "'checkpoint_epoch_frequency' must be non-None.")
        if (checkpoint_batch_frequency is not None):
            assert (isinstance(checkpoint_batch_frequency, int) and
                    checkpoint_batch_frequency > 0)
            self.__save_dataset_indices_used = True
        elif (checkpoint_epoch_frequency is not None):
            assert (isinstance(checkpoint_epoch_frequency, int) and
                    checkpoint_epoch_frequency > 0)
            self.__save_dataset_indices_used = False
        self.__training_set_sample_indices_not_used = None
        self.__checkpoint_batch_frequency = checkpoint_batch_frequency
        self.__checkpoint_epoch_frequency = checkpoint_epoch_frequency
        assert (isinstance(minibatch_summary_writer_frequency, int) and
                minibatch_summary_writer_frequency > 0)
        (self.__minibatch_summary_writer_frequency
        ) = minibatch_summary_writer_frequency
        if (training_job_name is None):
            self.__training_job_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.__training_job_name = training_job_name
        # Create the log folder if non-existent.
        if (not os.path.exists(self.__log_folder)):
            try:
                os.mkdir(self.__log_folder)
            except OSError:
                raise OSError("Error while trying to create folder "
                              f"'{self.__log_folder}'. Exiting.")
        # Create the checkpoint subfolder if nonexistent.
        self.__checkpoint_subfolder = os.path.join(self.__log_folder,
                                                   self.__training_job_name,
                                                   'checkpoints')
        if (not os.path.exists(self.__checkpoint_subfolder)):
            try:
                os.makedirs(self.__checkpoint_subfolder)
            except OSError:
                raise OSError("Error while trying to create folder "
                              f"'{self.__checkpoint_subfolder}'. Exiting.")

        # If nonexistent, create the subfolder of the log folder associated with
        # the current training job. Otherwise, verify if a saved checkpoint can
        # be found, and in that case assume that training should continued from
        # it.
        complete_path_logs_current_training_job = os.path.join(
            self.__log_folder, self.__training_job_name)
        if (not os.path.exists(complete_path_logs_current_training_job)):
            try:
                os.mkdir(complete_path_logs_current_training_job)
            except OSError:
                raise OSError(
                    "Error while trying to create the log subfolder "
                    f"'{complete_path_logs_current_training_job}'. Exiting. ")
            self.__continue_training_from_previous_checkpoint = False
            self.__found_job_folder = False
        else:
            self.__found_job_folder = True

            # Load latest checkpoint.
            if (self.__checkpoint_epoch_frequency is not None):
                assert (self.__checkpoint_batch_frequency is None)
                self.__epoch_most_recent_checkpoint = (
                    get_epoch_most_recent_checkpoint(
                        checkpoint_subfolder=self.__checkpoint_subfolder))
                if (self.__epoch_most_recent_checkpoint is None):
                    print(
                        "No saved checkpoints were found for training job "
                        f"'{self.__training_job_name}'. Will assume that a new "
                        "training job should be started with the parameters "
                        "stored in the parameter file.")
                    self.__continue_training_from_previous_checkpoint = False
                else:
                    self.__most_recent_checkpoint_filename = os.path.join(
                        self.__checkpoint_subfolder, 'checkpoint_'
                        f'{self.__epoch_most_recent_checkpoint:04d}.pth')
                    self.__current_epoch = (
                        self.__epoch_most_recent_checkpoint + 1)
                    self.__current_batch = 0
                    self.__continue_training_from_previous_checkpoint = True
            else:
                assert (self.__checkpoint_batch_frequency is not None)
                (self.__epoch_most_recent_checkpoint,
                 self.__batch_most_recent_checkpoint
                ) = get_epoch_and_batch_most_recent_checkpoint(
                    checkpoint_subfolder=self.__checkpoint_subfolder)
                assert ((self.__epoch_most_recent_checkpoint is
                         None) == (self.__batch_most_recent_checkpoint is None))
                if (self.__epoch_most_recent_checkpoint is None):
                    print(
                        "No saved checkpoints were found for training job "
                        f"'{self.__training_job_name}'. Will assume that a new "
                        "training job should be started with the parameters "
                        "stored in the parameter file.")
                    self.__continue_training_from_previous_checkpoint = False
                else:
                    self.__most_recent_checkpoint_filename = os.path.join(
                        self.__checkpoint_subfolder,
                        f'checkpoint_{self.__epoch_most_recent_checkpoint:04d}'
                        f'_batch_{self.__batch_most_recent_checkpoint:05d}.pth')
                    self.__current_epoch = self.__epoch_most_recent_checkpoint
                    self.__current_batch = (
                        self.__batch_most_recent_checkpoint + 1)
                    self.__continue_training_from_previous_checkpoint = True

        self.__training_parameters = None

        # Initialize the TensorBoard summary writer.
        if (self.__verbose):
            print("Initializing TensorBoard summary writer...")
        self.__summary_writer = SummaryWriter(
            os.path.join(complete_path_logs_current_training_job,
                         'tensorboard/'))
        self.__x_axis_last_summary_datapoint = None

        # Initialize the main components of the training job: network, dataset,
        # data loader, optimizer, learning-rate scheduler, loss.
        self._initialize_components(
            network_parameters=network_parameters,
            dataset_parameters=dataset_parameters,
            test_dataset_parameters=test_dataset_parameters,
            data_loader_parameters=data_loader_parameters,
            optimizer_parameters=optimizer_parameters,
            lr_scheduler_parameters=lr_scheduler_parameters,
            loss_parameters=loss_parameters)

        # Initialize the training job.
        self.__initialize_training_job()

    @property
    def training_parameters(self):
        assert (
            self.__net is not None and self.__data_loader_train is not None and
            self.__optimizer is not None and self.__loss is not None
        ), ("Unable to retrieve the input parameters of the current training "
            "job: at least of one the network, the data loader, the optimizer "
            "and the loss has not been initialized.")
        if (self.__training_parameters is None):
            network_parameters = self.__network_input_parameters
            dataset_parameters = self.__dataset_input_parameters
            test_dataset_parameters = self.__test_dataset_input_parameters
            data_loader_parameters = self.__data_loader_input_parameters
            optimizer_parameters = self.__optimizer_input_parameters
            lr_scheduler_parameters = None
            if (self.__lr_scheduler is not None):
                lr_scheduler_parameters = self.__lr_scheduler_input_parameters
            loss_parameters = self.__loss_input_parameters
            self.__training_parameters = {
                'network_parameters': network_parameters,
                'dataset_parameters': dataset_parameters,
                'test_dataset_parameters': test_dataset_parameters,
                'data_loader_parameters': data_loader_parameters,
                'optimizer_parameters': optimizer_parameters,
                'lr_scheduler_parameters': lr_scheduler_parameters,
                'loss_parameters': loss_parameters
            }
        return self.__training_parameters

    def _initialize_components(self,
                               network_parameters=None,
                               dataset_parameters=None,
                               test_dataset_parameters=None,
                               data_loader_parameters=None,
                               optimizer_parameters=None,
                               lr_scheduler_parameters=None,
                               loss_parameters=None):
        r"""Instantiates and initializes: network, dataset, data loader,
        optimizer, learning-rate scheduler, loss.

        Args:
            network_parameters, dataset_parameters, test_dataset_parameters,
            data_loader_parameters, optimizer_parameters,
            lr_scheduler_parameters, loss_parameters (dict): Cf. respective
            arguments in constructor.
        
        Returns:
            None.
        """
        # Initialize model, performing weight initialization only if training is
        # not resumed from a previous checkpoint.
        if (network_parameters is not None):
            assert ('should_initialize_weights' not in network_parameters), (
                "Network parameters should not contain the parameter "
                "'should_initialize_weights', as weights will be automatically "
                "initialized or not, depending on whether training is resumed "
                "from a previous job or not.")
            if (self.__verbose):
                print("Initializing network...")
            self.__net = create_model(should_initialize_weights=(
                not self.__continue_training_from_previous_checkpoint),
                                      **network_parameters)
            # Store model input parameters.
            self.__network_input_parameters = self.__net.input_parameters
            assert ('model_name' in network_parameters)
            assert ('model_name' not in self.__network_input_parameters)
            self.__network_input_parameters['model_name'] = network_parameters[
                'model_name']
            # Check whether the average ratios between the number of nodes/edges
            # in the primal graphs after and before pooling, for each pooling
            # layer, should be logged.
            should_log_node_ratios = (
                'log_ratios_new_old_primal_nodes' in
                self.__network_input_parameters and self.
                __network_input_parameters['log_ratios_new_old_primal_nodes'])
            should_log_edge_ratios = (
                'log_ratios_new_old_primal_edges' in
                self.__network_input_parameters and self.
                __network_input_parameters['log_ratios_new_old_primal_edges'])
            self.__log_ratios_new_old_primal_nodes = False
            self.__log_ratios_new_old_primal_edges = False
            if (should_log_node_ratios or should_log_edge_ratios):
                at_least_one_pooling_layer = False
                if (self.__network_input_parameters['num_primal_edges_to_keep']
                        is not None):
                    at_least_one_pooling_layer |= (len([
                        coeff for coeff in self.
                        __network_input_parameters['num_primal_edges_to_keep']
                        if coeff != None
                    ]) > 0)
                elif (self.__network_input_parameters[
                        'fractions_primal_edges_to_keep'] is not None):
                    at_least_one_pooling_layer |= (len([
                        coeff for coeff in self.__network_input_parameters[
                            'fractions_primal_edges_to_keep'] if coeff != None
                    ]) > 0)
                elif (self.__network_input_parameters[
                        'primal_attention_coeffs_thresholds'] is not None):
                    at_least_one_pooling_layer |= (len([
                        coeff for coeff in self.__network_input_parameters[
                            'primal_attention_coeffs_thresholds']
                        if coeff != None
                    ]) > 0)
                if (at_least_one_pooling_layer):
                    if (should_log_node_ratios):
                        self.__log_ratios_new_old_primal_nodes = True
                    if (should_log_edge_ratios):
                        self.__log_ratios_new_old_primal_edges = True

            # Move network to GPU if necessary.
            if (self.__use_gpu):
                self.__net.to("cuda")
            else:
                self.__net.to("cpu")
        # Initialize dataset.
        if (dataset_parameters is not None):
            assert (data_loader_parameters is not None), (
                "Dataset can be only initialized if also a data loader is "
                "concurrently initialized.")
            if (data_loader_parameters['validation_set_fraction'] is not None):
                assert (isinstance(
                    data_loader_parameters['validation_set_fraction'], float))
                assert (0.0 <= data_loader_parameters['validation_set_fraction']
                        <= 1.0)
            if (self.__save_dataset_indices_used):
                dataset_parameters['return_sample_indices'] = True
            if (self.__verbose):
                print("Initializing dataset...")
            if ('compute_node_feature_stats' in dataset_parameters and
                    not dataset_parameters['compute_node_feature_stats']):
                dataset, _ = create_dataset(**dataset_parameters)
                primal_mean = primal_std = dual_mean = dual_std = None
                if (data_loader_parameters['validation_set_fraction'] is
                        not None):
                    # Split the training set into a part that will be actually
                    # used for training and another that will be used for
                    # validation.
                    self.__num_samples_train = int(
                        len(dataset) *
                        (1. -
                         data_loader_parameters['validation_set_fraction']))
            else:
                whole_training_set_is_used_for_training = True
                if (data_loader_parameters['validation_set_fraction'] is
                        not None):
                    dataset_parameters_no_mean_std = dataset_parameters.copy()
                    dataset_parameters_no_mean_std[
                        'compute_node_feature_stats'] = False
                    dataset, _ = create_dataset(
                        **dataset_parameters_no_mean_std)
                    # Split the training set into a part that will be actually
                    # used for training and another that will be used for
                    # validation.
                    self.__num_samples_train = int(
                        len(dataset) *
                        (1. -
                         data_loader_parameters['validation_set_fraction']))
                    if (self.__num_samples_train < len(dataset)):
                        whole_training_set_is_used_for_training = False

                if (not whole_training_set_is_used_for_training):
                    print("\033[93mNote: it was requested that the training "
                          "set be split in a training part ("
                          f"{self.__num_samples_train}/{len(dataset)} samples) "
                          "and a validation part ("
                          f"{len(dataset) - self.__num_samples_train}/"
                          f"{len(dataset)}).\033[00m")
                    if ('node_feature_stats_filename' in dataset_parameters and
                            dataset_parameters['node_feature_stats_filename'] is
                            not None):
                        stats_filename = dataset_parameters[
                            'node_feature_stats_filename']
                    else:
                        # Automatically define a path where to save the
                        # statistics-file about the part of the training set
                        # actually used for training. Note: if this file exists,
                        # the data loader will try to load the statistics from
                        # there, and will raise an error if the parameters of
                        # the current dataset do not match those of the dataset
                        # with which the file was produced.
                        stats_filename = os.path.join(
                            dataset_parameters['root'],
                            f"{dataset_parameters['dataset_name']}_"
                            f"{self.__num_samples_train}_training_samples.pkl")
                        dataset_parameters[
                            'node_feature_stats_filename'] = stats_filename

                    if (self.__verbose):
                        if (os.path.exists(stats_filename)):
                            print("\033[93mWill restore mean/standard "
                                  "deviation of the node features of the part "
                                  "of the training set actually used for "
                                  f"training from the file '{stats_filename}'."
                                  "\033[00m")
                        else:
                            print("\033[93mWill save mean/standard deviation "
                                  "of the node features of the part of the "
                                  "training set actually used for training "
                                  f"in the file '{stats_filename}'.\033[00m")

                    # Compute mean/std on the part that will be used for
                    # training.
                    # - Store the range of samples actually used for training
                    #   and hence for computing mean/standard deviation.
                    dataset_params_for_mean_and_std = dataset.input_parameters
                    dataset_params_for_mean_and_std['sample_idx_start'] = 0
                    dataset_params_for_mean_and_std[
                        'sample_idx_end'] = self.__num_samples_train - 1
                    (primal_mean, primal_std, dual_mean,
                     dual_std) = compute_mean_and_std(
                         dataset=dataset[:self.__num_samples_train],
                         dataset_params=dataset_params_for_mean_and_std,
                         filename=stats_filename)
                else:
                    dataset, (primal_mean, primal_std, dual_mean,
                              dual_std) = create_dataset(**dataset_parameters)
            # It is supposed that a training split is used for training.
            if (dataset.input_parameters['train'] is False):
                print("\033[93mWill use the test split of the input dataset "
                      "for training. Please double-check that this is the "
                      "desired behaviour.\033[00m")
            # Create the test dataset if necessary.
            if (test_dataset_parameters is not None):
                test_dataset, _ = create_dataset(**test_dataset_parameters)
                if (test_dataset.input_parameters['train'] is True):
                    print("\033[93mUsing a training split to compute test "
                          "accuracy. Please double-check in the test-dataset "
                          "parameters that this is the desired behaviour."
                          "\033[00m")
            # Store dataset(s) input parameters.
            self.__dataset_input_parameters = dataset.input_parameters
            input_parameters_list = [self.__dataset_input_parameters]
            if (test_dataset_parameters is not None):
                self.__test_dataset_input_parameters = (
                    test_dataset.input_parameters)
                input_parameters_list.append(
                    self.__test_dataset_input_parameters)
            else:
                self.__test_dataset_input_parameters = None

            all_dataset_creator_input_parameters = {
                k: v for k, v in inspect.signature(
                    create_dataset).parameters.items()
            }
            # Do not save the parameters 'sample_idx_start' and
            # 'sample_idx_end', because there cannot be passed later to the
            # factory function create_dataset.
            for train_0_test_1, dataset_input_parameters in enumerate(
                    input_parameters_list):
                if ('sample_idx_start' in dataset_input_parameters):
                    assert ('sample_idx_end' in dataset_input_parameters)
                    dataset_input_parameters.pop('sample_idx_start')
                    dataset_input_parameters.pop('sample_idx_end')
                for k, v in all_dataset_creator_input_parameters.items():
                    # Ignore input parameters of the dataset already included in
                    # the dataset property 'input_parameters'; just include the
                    # rest of the input arguments of the function
                    # 'create_dataset'.
                    if (k != 'dataset_params'):
                        assert (k not in dataset_input_parameters)
                        if (train_0_test_1 == 0 and k in dataset_parameters):
                            if (k == 'node_feature_stats_filename'):
                                if (dataset_parameters[k] is not None):
                                    dataset_parameters[k] = os.path.abspath(
                                        dataset_parameters[k])
                            dataset_input_parameters[k] = dataset_parameters[k]
                        elif (train_0_test_1 == 1 and
                              k in test_dataset_parameters):
                            if (k == 'node_feature_stats_filename'):
                                if (test_dataset_parameters[k] is not None):
                                    test_dataset_parameters[
                                        k] = os.path.abspath(
                                            test_dataset_parameters[k])
                            dataset_input_parameters[
                                k] = test_dataset_parameters[k]
                        else:
                            # Use default value from 'create_dataset' function.
                            assert (v.default is not inspect.Parameter.empty)
                            dataset_input_parameters[k] = v.default

            # If necessary, exclude from the dataset the samples already used in
            # the non-terminated epoch.
            if (self.__training_set_sample_indices_not_used is not None):
                dataset = dataset[self.__training_set_sample_indices_not_used]

        # Initialize data loader(s).
        if (data_loader_parameters is not None):
            assert (dataset_parameters is not None), (
                "Data loader can be only initialized if also a dataset is "
                "concurrently initialized.")
            assert (len(
                set(['primal_mean', 'primal_std', 'dual_mean', 'dual_std']) &
                set(data_loader_parameters)
            ) == 0), (
                "Data-loader parameters should not contain any of the "
                "following parameters, as they will be automatically computed "
                "from the dataset if set to do so: 'primal_mean', "
                "'primal_std', 'dual_mean', 'dual_std'.")
            if (self.__verbose):
                print("Initializing data loader...")
            validation_set_fraction = data_loader_parameters.pop(
                'validation_set_fraction')
            if (self.__save_dataset_indices_used):
                data_loader_parameters['return_sample_indices'] = True
            # The validation set needs to contain at least one sample.
            if (validation_set_fraction is not None and
                    self.__num_samples_train < len(dataset)):
                self.__data_loader_train = DualPrimalDataLoader(
                    dataset=dataset[:self.__num_samples_train],
                    primal_mean=primal_mean,
                    primal_std=primal_std,
                    dual_mean=dual_mean,
                    dual_std=dual_std,
                    **data_loader_parameters)
                self.__data_loader_validation = DualPrimalDataLoader(
                    dataset=dataset[self.__num_samples_train:],
                    primal_mean=primal_mean,
                    primal_std=primal_std,
                    dual_mean=dual_mean,
                    dual_std=dual_std,
                    **data_loader_parameters)
            else:
                self.__data_loader_train = DualPrimalDataLoader(
                    dataset=dataset,
                    primal_mean=primal_mean,
                    primal_std=primal_std,
                    dual_mean=dual_mean,
                    dual_std=dual_std,
                    **data_loader_parameters)
            # Optionally create data loader for test dataset.
            if (test_dataset_parameters is not None):
                test_data_loader_parameters = data_loader_parameters.copy()
                test_data_loader_parameters['return_sample_indices'] = False
                self.__data_loader_test = DualPrimalDataLoader(
                    dataset=test_dataset,
                    primal_mean=primal_mean,
                    primal_std=primal_std,
                    dual_mean=dual_mean,
                    dual_std=dual_std,
                    **test_data_loader_parameters)
            # Store data-loader input parameters.
            (self.__data_loader_input_parameters
            ) = self.__data_loader_train.input_parameters
            assert ('validation_set_fraction' not in
                    self.__data_loader_input_parameters)
            self.__data_loader_input_parameters[
                'validation_set_fraction'] = validation_set_fraction
            for keyword in ['primal', 'dual']:
                self.__data_loader_input_parameters.pop(f'{keyword}_mean')
                self.__data_loader_input_parameters.pop(f'{keyword}_std')
        # Initialize optimizer.
        if (optimizer_parameters is not None):
            assert (self.__net is not None), (
                "Optimizer can only be instantiated if a network is already "
                "initialized.")
            assert ('network_parameters' not in optimizer_parameters), (
                "Optimizer parameters should not contain the parameter "
                "'network_parameters', as this will be obtained directly from "
                "the instantiated network.")
            if (self.__verbose):
                print("Initializing optimizer...")
            self.__optimizer = create_optimizer(
                network_parameters=self.__net.parameters(),
                **optimizer_parameters)
            # Store optimizer input parameters.
            self.__optimizer_input_parameters = optimizer_parameters.copy()
            all_optimizer_input_parameters = {
                k: v for k, v in inspect.signature(
                    self.__optimizer.__class__).parameters.items()
            }
            for k, v in all_optimizer_input_parameters.items():
                # Ignore the parameter 'params' (parameters on which the
                # optimizer should optimize).
                if (k != 'params'):
                    # Use default values for optimizer parameters not explicitly
                    # passed as input arguments of the training job.
                    if (not k in optimizer_parameters):
                        assert (v.default is not inspect.Parameter.empty)
                        self.__optimizer_input_parameters[k] = v.default
        # Initialize learning-rate scheduler.
        if (lr_scheduler_parameters is not None):
            assert (self.__optimizer is not None), (
                "Learning-rate scheduler can only be instantiated if an "
                "optimizer is already initialized.")
            assert ('optimizer' not in lr_scheduler_parameters), (
                "Learning-rate scheduler parameters should not contain the "
                "parameter 'optimizer', as this will be obtained directly from "
                "the instantiated optimizer of the training job.")
            if (self.__verbose):
                print("Initializing learning-rate scheduler...")
            self.__lr_scheduler = create_lr_scheduler(
                optimizer=self.__optimizer, **lr_scheduler_parameters)
            # Store learning-rate scheduler input parameters.
            self.__lr_scheduler_input_parameters = lr_scheduler_parameters.copy(
            )
            all_lr_scheduler_input_parameters = {
                k: v for k, v in inspect.signature(
                    self.__lr_scheduler.__class__).parameters.items()
            }
            for k, v in all_lr_scheduler_input_parameters.items():
                # Ignore the parameter 'optimizer' (optimizer of which the
                # learning-rate scheduler should modify the learning rate).
                if (k != 'optimizer'):
                    if (not k in lr_scheduler_parameters):
                        # Use default values for learning-rate scheduler
                        # parameters not explicitly passed as input arguments of
                        # the training job.
                        # Ignore the input argument 'lr_lambda' of the StepLR
                        # scheduler, as this is defined in create_lr_scheduler.
                        if (k != 'lr_lambda'):
                            self.__lr_scheduler_input_parameters[k] = v.default
        # Initialize loss.
        if (loss_parameters is not None):
            if (self.__verbose):
                print("Initializing loss...")
            self.__loss = create_loss(**loss_parameters)
            # Store loss input parameters.
            self.__loss_input_parameters = loss_parameters.copy()
            all_loss_input_parameters = {
                k: v for k, v in inspect.signature(
                    self.__loss.__class__).parameters.items()
            }
            for k, v in all_loss_input_parameters.items():
                if (not k in loss_parameters):
                    # Use default values for loss parameters not explicitly
                    # passed as input arguments of the training job.
                    assert (v.default is not inspect.Parameter.empty)
                    self.__loss_input_parameters[k] = v.default
            self.__task_type = self.__loss_input_parameters['task_type']

    def __initialize_training_job(self):
        r"""If training should resume from a previous checkpoint, verifies that
        the network is compatible with the input parameters of the previous
        training job and restores the network weights and the optimizer
        parameters from the latest available checkpoint.
        If training should be started from scratch, saves the input parameters
        of the training job.

        Args:
            None.

        Returns:
            None.
        """
        if (self.__verbose):
            print("Initializing training job...")
        if (self.__found_job_folder):
            training_job_complete_path = os.path.join(self.__log_folder,
                                                      self.__training_job_name)
            training_parameters_full_filename = os.path.join(
                training_job_complete_path, training_parameters_filename)
            # Load training parameters of previous training job.
            (previous_training_parameters) = self.__load_training_parameters()
            # Initialize, based on the parameters from the previous training
            # job, the components that were not initialized with input
            # parameters in the current job.
            components_to_initialize = set()
            if (self.__net is None):
                components_to_initialize.add('network')
            if (self.__data_loader_train is None):
                components_to_initialize.add('data_loader')
                # Data loader requires the dataset to be concurrently
                # initialized.
                components_to_initialize.add('dataset')
            if (self.__data_loader_test is None):
                components_to_initialize.add('test_dataset')
            if (self.__optimizer is None):
                components_to_initialize.add('optimizer')
            if (self.__lr_scheduler is None):
                components_to_initialize.add('lr_scheduler')
            if (self.__loss is None):
                components_to_initialize.add('loss')

            for component_to_initialize in components_to_initialize:
                if (component_to_initialize is not 'lr_scheduler'):
                    # Other than the learning-rate scheduler, the other
                    # components must be loaded from the previous training job
                    # or initialized with the input parameters of the current
                    # training job.
                    assert (f'{component_to_initialize}_parameters' in
                            previous_training_parameters)
            self._initialize_components(
                **{
                    f'{c}_parameters':
                    previous_training_parameters[f'{c}_parameters']
                    for c in components_to_initialize
                })

            # Check whether the average ratios between the number of nodes/edges
            # in the primal graphs after and before pooling, for each pooling
            # layer, should be logged.
            if ('log_ratios_new_old_primal_nodes' in
                    previous_training_parameters['network_parameters'] and
                    previous_training_parameters['network_parameters']
                ['log_ratios_new_old_primal_nodes'] is True):
                self.__log_ratios_new_old_primal_nodes = True
            else:
                self.__log_ratios_new_old_primal_nodes = False
            if ('log_ratios_new_old_primal_edges' in
                    previous_training_parameters['network_parameters'] and
                    previous_training_parameters['network_parameters']
                ['log_ratios_new_old_primal_edges'] is True):
                self.__log_ratios_new_old_primal_edges = True
            else:
                self.__log_ratios_new_old_primal_edges = False

            # Load training parameters of the current training job.
            training_parameters = self.training_parameters
            # Verify that the network is compatible with the input parameters of
            # the previous training job.
            network_parameters = training_parameters['network_parameters']
            previous_network_parameters = previous_training_parameters[
                'network_parameters']
            if (network_parameters != previous_network_parameters):
                raise ValueError(
                    "Trying to resume training from job at "
                    f"'{training_job_complete_path}', but the current network "
                    "has incompatible input parameters. Exiting.")
            # Verify that dataset, data loader, optimizer, learning-rate
            # scheduler and loss have the same parameters as those used in the
            # previous training job.
            for component in [
                    'dataset', 'data_loader', 'optimizer', 'lr_scheduler',
                    'loss'
            ]:
                # If the component was restored from the previous training job,
                # there is no need to check.
                if (not component in components_to_initialize):
                    current_parameters = training_parameters[
                        f'{component}_parameters']
                    previous_parameters = previous_training_parameters[
                        f'{component}_parameters']
                    are_parameters_equal = True
                    if (current_parameters.keys() !=
                            previous_parameters.keys()):
                        are_parameters_equal = False
                    else:
                        for parameter_key in current_parameters:
                            if (type(current_parameters[parameter_key]) != type(
                                    previous_parameters[parameter_key])):
                                are_parameters_equal = False
                                break
                            else:
                                if (isinstance(
                                        current_parameters[parameter_key],
                                        np.ndarray)):
                                    if (not np.array_equal(
                                            current_parameters[parameter_key],
                                            previous_parameters[parameter_key])
                                       ):
                                        are_parameters_equal = False
                                        break
                                else:
                                    if (current_parameters[parameter_key] !=
                                            previous_parameters[parameter_key]):
                                        are_parameters_equal = False
                                        break
                    if (not are_parameters_equal):
                        raise ValueError(
                            "Trying to resume training from job at "
                            f"'{training_job_complete_path}', but the current "
                            f"{component} has parameters incompatible with "
                            "those used in the previous training job. If you "
                            "are sure that you would like to resume training "
                            f"with new {component} parameters, please manually "
                            "update the file "
                            f"'{training_parameters_full_filename}'. Exiting.")

            # Restore the network weights and the optimizer state variables if
            # available.
            if (self.__continue_training_from_previous_checkpoint):
                assert (self.__epoch_most_recent_checkpoint is not None)
                try:
                    device = "cuda" if self.__use_gpu else "cpu"
                    checkpoint = torch.load(
                        self.__most_recent_checkpoint_filename,
                        map_location=device)
                    assert (checkpoint['epoch'] ==
                            self.__epoch_most_recent_checkpoint)
                    # Restore the training samples not used in case of
                    # checkpoints saved before the completion of an epoch.
                    if (self.__checkpoint_batch_frequency is not None):
                        self.__training_set_sample_indices_not_used = (
                            checkpoint['sample_indices_to_use'])
                except OSError:
                    raise OSError(
                        "Unable to open checkpoint "
                        f"'{self.__most_recent_checkpoint_filename}'. Exiting.")
                self.__net.load_state_dict(checkpoint['net'])
                self.__optimizer.load_state_dict(checkpoint['optimizer'])
                if (self.__lr_scheduler is not None):
                    self.__lr_scheduler.load_state_dict(
                        checkpoint['lr_scheduler'])
                else:
                    if ('lr_scheduler' in checkpoint):
                        raise ValueError(
                            "Found 'lr_scheduler' info in checkpoint "
                            f"'{most_recent_checkpoint_filename}', but no "
                            "learning-rate scheduler was given as input to the "
                            "training job. Exiting.")
                self.__x_axis_last_summary_datapoint = checkpoint[
                    'x_axis_last_summary_datapoint']

                print(
                    "\033[92mContinuing training from previous job (parameters "
                    f"stored in file '{training_parameters_full_filename}').\n"
                    f"Starting from epoch {self.__current_epoch}.\033[00m")
            else:
                # Assume that a new training job with the parameters provided
                # should be started.
                # - Initialize the current training epoch.
                self.__current_epoch = 1
                # - Initialize the current batch.
                self.__current_batch = 0
                # - Initialize the x-axis of the summary writer.
                self.__x_axis_last_summary_datapoint = 0
        else:
            # Save the training parameters of the current training job.
            self.__save_training_parameters()
            # Initialize the current training epoch.
            self.__current_epoch = 1
            # Initialize the current batch.
            self.__current_batch = 0
            # Initialize the x-axis of the summary writer.
            self.__x_axis_last_summary_datapoint = 0

        # Check that the type of graphs (i.e., the dual-graph configuration) in
        # the dataset is the same as the one for which the network was
        # instantiated.
        single_dual_nodes_dataset = self.training_parameters[
            'dataset_parameters']['single_dual_nodes']
        single_dual_nodes_network = self.training_parameters[
            'network_parameters']['single_dual_nodes']
        undirected_dual_nodes_dataset = self.training_parameters[
            'dataset_parameters']['undirected_dual_edges']
        undirected_dual_nodes_network = self.training_parameters[
            'network_parameters']['undirected_dual_edges']
        assert (
            single_dual_nodes_dataset == single_dual_nodes_network and
            undirected_dual_nodes_dataset == undirected_dual_nodes_network), (
                "The dual-graph configuration in the dataset must be the same "
                "as the one for which the network was instantiated.")

    def __reinitialize_dataset_and_dataloader(self):
        r"""Re-initializes dataset and data loader with the parameters stored in
        the parameter file. Useful if a training job was resumed from a
        checkpoint generated from an incomplete epoch, and therefore dataset and
        data loader have been initialized to operate with a subset of the
        dataset.
        Args:
            None.
        Returns:
            None.
        """
        # Ensures that all samples from the dataset are used.
        self.__training_set_sample_indices_not_used = None

        previous_training_parameters = self.__load_training_parameters()

        components_to_initialize = ['dataset', 'data_loader']

        for component_to_initialize in components_to_initialize:
            assert (f'{component_to_initialize}_parameters' in
                    previous_training_parameters)
        self.__initialize_components(
            **{
                f'{c}_parameters':
                previous_training_parameters[f'{c}_parameters']
                for c in components_to_initialize
            })

    def __get_layers_input_parameters(self, base_layer):
        r"""Returns the input parameters of all layers that are children of the
        given input layer. If a torch.nn.Sequential child layer is found, the
        method is recursively called on all of its children.

        [Deprecated].

        Args:
            base_layer (torch.nn.Module): Base parent layer of the children
                layers of which the input parameters should be found.
        
        Returns:
            layers_input_parameters (dict): Dictionary having as keys the
                attribute name of child layer of the base parent layer, and as
                values the input parameters of the associated child layer.
        """
        layers_input_parameters = dict()

        for layer in base_layer.named_children():
            layer_name = layer[0]
            is_layer_sequential = isinstance(layer[1], torch.nn.Sequential)
            assert (layer_name not in layers_input_parameters)

            if (not is_layer_sequential):
                layers_input_parameters[layer_name] = getattr(
                    base_layer, layer_name).input_parameters
            else:
                # Recursively find the input parameters of each layer in the
                # sequential layer.
                sublayers_input_parameters = self.__get_layers_input_parameters(
                    getattr(base_layer, layer_name))
                # Add the sublayer dictionary to the layer dictionary by adding
                # the layer name as a prefix to the sublayer names.
                sublayers_input_parameters = {
                    f'{layer_name}.{k}': v
                    for k, v in sublayers_input_parameters.items()
                }
                layers_input_parameters.update(sublayers_input_parameters)

        return layers_input_parameters

    def __load_training_parameters(self):
        r"""Loads the training parameters from a previous training job, by
        reading the associated file in the log folder.

        Args:
            None.

        Returns:
            previous_training_parameters (dict): Dictionary having as keys
                dictionaries representing the parameters of the network, the
                dataset, the data loader, the optimizer, the learning-rate
                scheduler and the loss used in the training job. Cf. docs of
                property `training_parameters`.
        """
        # Read training-parameter file.
        parameter_filename = os.path.join(self.__log_folder,
                                          self.__training_job_name,
                                          training_parameters_filename)
        if (self.__verbose):
            print(f"Loading training parameters from '{parameter_filename}'...")
        try:
            with open(parameter_filename, 'r') as f:
                previous_training_parameters = yaml.load(f,
                                                         Loader=yaml.FullLoader)
                return previous_training_parameters
        except IOError:
            raise IOError("Unable to open previous-training-parameter file at "
                          f"'{parameter_filename}'. Exiting.")

    def __save_training_parameters(self):
        r"""Saves the training parameters of the current training job.

        Args:
            None.

        Returns:
            None.
        """
        # Save training-parameter file.
        parameter_filename = os.path.join(self.__log_folder,
                                          self.__training_job_name,
                                          training_parameters_filename)
        try:
            with open(parameter_filename, 'w') as f:
                yaml.dump(self.training_parameters, f)
        except IOError:
            raise IOError("Error while writing training-parameter file "
                          f"'{parameter_filename}'. Exiting.")

    def __save_checkpoint(self, batch_index=None):
        r"""Saves a checkpoint file in the subfolder `checkpoints/` of the log
        folder, containing the network weights, the optimizer state variables,
        and the epoch number.

        Args:
             batch_index (int/None, optional): If not None, index of the last
                batch completed before saving the checkpoint. The index is saved
                in the checkpoint and in the checkpoint name.
                (default: :obj:`None`)

        Returns:
            None.
        """
        if (batch_index is None):
            checkpoint_filename = os.path.join(
                self.__checkpoint_subfolder,
                f'checkpoint_{self.__current_epoch:04d}.pth')
        else:
            checkpoint_filename = os.path.join(
                self.__checkpoint_subfolder,
                f'checkpoint_{self.__current_epoch:04d}_batch_'
                f'{batch_index:05d}.pth')
        checkpoint_dict = {
            'epoch':
                self.__current_epoch,
            'net':
                self.__net.state_dict(),
            'optimizer':
                self.__optimizer.state_dict(),
            'x_axis_last_summary_datapoint':
                self.__x_axis_last_summary_datapoint,
            'batch_index':
                batch_index,
            'sample_indices_to_use':
                self.__training_set_sample_indices_not_used
        }
        if (self.__lr_scheduler is not None):
            checkpoint_dict['lr_scheduler'] = self.__lr_scheduler.state_dict()
        torch.save(checkpoint_dict, checkpoint_filename)

    def train(self):
        r"""Starts the training job, during which checkpoints are saved for
        networks weight and optimizer state variables.

        Args:
            None.

        Returns:
            None.
        """
        if (self.__current_epoch > self.__final_training_epoch):
            warnings.warn(
                "No training step will be performed, as the current epoch is "
                f"{self.__current_epoch}, while the final epoch is set to be "
                f"{self.__final_training_epoch}.")
            return

        device = "cuda" if self.__use_gpu else "cpu"
        print(f"Running training job {self.__training_job_name}")
        training_set_size = len(self.__data_loader_train.dataset)
        if (len(self.__data_loader_train) !=
                training_set_size // self.__data_loader_train.batch_size):
            print("\033[93mNote: the last batch in each epoch will not be of "
                  f"size {self.__data_loader_train.batch_size}, as "
                  f"{training_set_size} is not a multiple of "
                  f"{self.__data_loader_train.batch_size}.\033[00m")

        while (self.__current_epoch <= self.__final_training_epoch):
            num_iterations = len(self.__data_loader_train)
            print(f"Current epoch: {self.__current_epoch}")
            if (self.__current_batch != 0):
                self.__restarted_training_from_nonzero_batch = True
                num_iterations += self.__current_batch
                print(f"Starting from batch: {self.__current_batch}")
            else:
                self.__restarted_training_from_nonzero_batch = False
            num_batches_since_last_summary = 1
            # Training.
            # - Set up the network and the optimizer for training.
            self.__net.train()
            running_loss = 0.0
            if (self.__log_ratios_new_old_primal_nodes):
                running_average_node_ratios_in_pooling_layer = {
                    pooling_layer_idx: torch.tensor(0.0)
                    for pooling_layer_idx in self.__net._indices_pooling_layers
                }
            if (self.__log_ratios_new_old_primal_edges):
                running_average_edge_ratios_in_pooling_layer = {
                    pooling_layer_idx: torch.tensor(0.0)
                    for pooling_layer_idx in self.__net._indices_pooling_layers
                }
            if (self.__training_set_sample_indices_not_used is None or
                    len(self.__training_set_sample_indices_not_used) == 0):
                self.__training_set_sample_indices_not_used = [
                    idx for idx in range(len(self.__data_loader_train.dataset))
                ]
            # The `sample_indices` below are set by the dataset, which however
            # infers them from the indices in the current dataset. This means
            # that if a dataset is 'sliced', the sample indices do not match the
            # indices in the complete dataset, and it is therefore necessary to
            # keep a mapping from the sample indices in the 'sliced' dataset to
            # those in the original dataset.
            self.__training_set_sample_indices_not_used_train_start = (
                self.__training_set_sample_indices_not_used.copy())
            for data in self.__data_loader_train:
                print(f"\tIteration no. {self.__current_batch}. Last one is "
                      f"{num_iterations-1}.")
                if (self.__save_dataset_indices_used):
                    (primal_graph_batch, dual_graph_batch,
                     primal_edge_to_dual_node_idx_batch, sample_indices) = data
                else:
                    (primal_graph_batch, dual_graph_batch,
                     primal_edge_to_dual_node_idx_batch) = data
                primal_graph_batch = primal_graph_batch.to(device)
                dual_graph_batch = dual_graph_batch.to(device)
                # Note: primal_graph_edge_to_dual_graph_idx_batch is a dict, and
                # therefore cannot be moved to GPU.

                self.__optimizer.zero_grad()
                outputs, log_info = self.__net(
                    primal_graph_batch=primal_graph_batch,
                    dual_graph_batch=dual_graph_batch,
                    primal_edge_to_dual_node_idx_batch=
                    primal_edge_to_dual_node_idx_batch)
                if (self.__log_ratios_new_old_primal_nodes):
                    ratios_new_old_primal_nodes_per_pooling_layer = (
                        log_info.ratios_new_old_primal_nodes)
                if (self.__log_ratios_new_old_primal_edges):
                    ratios_new_old_primal_edges_per_pooling_layer = (
                        log_info.ratios_new_old_primal_edges)

                targets = primal_graph_batch.y
                step_loss = self.__loss(outputs, targets)
                step_loss.backward()
                self.__optimizer.step()
                # Update training loss.
                running_loss += step_loss.item()

                if (self.__log_ratios_new_old_primal_nodes):
                    # Optionally update, for each pooling layer, the average
                    # ratio between the number of nodes in the primal graphs
                    # after and before pooling.
                    for (
                            pooling_layer_idx, ratios_in_pooling_layer
                    ) in ratios_new_old_primal_nodes_per_pooling_layer.items():
                        # - Compute average for each pooling layer.
                        average_ratio_in_pooling_layer = (
                            ratios_in_pooling_layer.mean())
                        running_average_node_ratios_in_pooling_layer[
                            pooling_layer_idx] += average_ratio_in_pooling_layer
                if (self.__log_ratios_new_old_primal_edges):
                    # Optionally update, for each pooling layer, the average
                    # ratio between the number of edges in the primal graphs
                    # after and before pooling.
                    for (
                            pooling_layer_idx, ratios_in_pooling_layer
                    ) in ratios_new_old_primal_edges_per_pooling_layer.items():
                        # - Compute average for each pooling layer.
                        average_ratio_in_pooling_layer = (
                            ratios_in_pooling_layer.mean())
                        running_average_edge_ratios_in_pooling_layer[
                            pooling_layer_idx] += average_ratio_in_pooling_layer

                if (self.__save_dataset_indices_used):
                    # Update the indices in the dataset that were already used.
                    for idx in sample_indices:
                        self.__training_set_sample_indices_not_used.remove(
                            self.
                            __training_set_sample_indices_not_used_train_start[
                                idx])

                # Save checkpoint if required.
                if (self.__checkpoint_batch_frequency is not None):
                    if (self.__current_batch %
                            self.__checkpoint_batch_frequency == 0 and
                            self.__current_batch != 0):
                        # Save checkpoint.
                        self.__save_checkpoint(batch_index=self.__current_batch)

                # Save TensorBoard summary if necessary.
                if (num_batches_since_last_summary %
                        self.__minibatch_summary_writer_frequency == 0):
                    # Update position on the x-axis.
                    (self.__x_axis_last_summary_datapoint
                    ) += self.__minibatch_summary_writer_frequency
                    # Save summary for training loss and learning rate.
                    self.__summary_writer.add_scalar(
                        'training_loss', running_loss /
                        self.__minibatch_summary_writer_frequency,
                        self.__x_axis_last_summary_datapoint)

                    if (self.__lr_scheduler is not None):
                        curr_learning_rate = self.__lr_scheduler.get_last_lr(
                        )[0]
                    else:
                        assert (len(self.__optimizer.param_groups) == 1)
                        curr_learning_rate = self.__optimizer.param_groups[0][
                            'lr']
                    self.__summary_writer.add_scalar(
                        'learning_rate', curr_learning_rate,
                        self.__x_axis_last_summary_datapoint)

                    if (self.__log_ratios_new_old_primal_nodes):
                        # Save summary for average ratio between the number of
                        # nodes in the primal graphs after and before pooling.
                        for (
                                pooling_layer_idx, running_average_ratio
                        ) in running_average_node_ratios_in_pooling_layer.items(
                        ):
                            self.__summary_writer.add_scalar(
                                'mean_ratio_new_old_primal_nodes_'
                                f'pool{pooling_layer_idx}',
                                running_average_ratio /
                                self.__minibatch_summary_writer_frequency,
                                self.__x_axis_last_summary_datapoint)
                            running_average_node_ratios_in_pooling_layer[
                                pooling_layer_idx] = 0.0
                    if (self.__log_ratios_new_old_primal_edges):
                        # Save summary for average ratio between the number of
                        # edges in the primal graphs after and before pooling.
                        for (
                                pooling_layer_idx, running_average_ratio
                        ) in running_average_edge_ratios_in_pooling_layer.items(
                        ):
                            self.__summary_writer.add_scalar(
                                'mean_ratio_new_old_primal_edges_'
                                f'pool{pooling_layer_idx}',
                                running_average_ratio /
                                self.__minibatch_summary_writer_frequency,
                                self.__x_axis_last_summary_datapoint)
                            running_average_edge_ratios_in_pooling_layer[
                                pooling_layer_idx] = 0.0

                    num_batches_since_last_summary = 1
                    running_loss = 0.0
                else:
                    num_batches_since_last_summary += 1

                # Increment batch index.
                self.__current_batch += 1

            if (self.__data_loader_validation is not None):
                # Optional validation.
                # - Set up the network and the optimizer for evaluation.
                self.__net.eval()
                total_validation_loss = 0.0

                for validation_iteration_idx, data in enumerate(
                        self.__data_loader_validation):
                    with torch.no_grad():
                        (primal_graph_batch, dual_graph_batch,
                         primal_edge_to_dual_node_idx_batch) = data
                        primal_graph_batch = primal_graph_batch.to(device)
                        dual_graph_batch = dual_graph_batch.to(device)
                        outputs, _ = self.__net(
                            primal_graph_batch=primal_graph_batch,
                            dual_graph_batch=dual_graph_batch,
                            primal_edge_to_dual_node_idx_batch=
                            primal_edge_to_dual_node_idx_batch)

                        targets = primal_graph_batch.y
                        validation_step_loss = self.__loss(outputs, targets)
                        # Update validation loss.
                        total_validation_loss += validation_step_loss.item()

                total_validation_loss /= (validation_iteration_idx + 1)
                print(f"Validation loss: {total_validation_loss:.4f}")
                # Save TensorBoard summary.
                x_axis_validation_loss = (self.__x_axis_last_summary_datapoint +
                                          num_batches_since_last_summary - 1)
                # Save summary for validation loss.
                self.__summary_writer.add_scalar('validation_loss',
                                                 total_validation_loss,
                                                 x_axis_validation_loss)
            if (self.__data_loader_test is not None):
                # Optional test accuracy.
                # - Set up the network and the optimizer for evaluation.
                self.__net.eval()
                total_num_correct_predictions = 0
                total_num_predictions = 0
                # Optionally count also the number of predictions according to
                # area-weighted accuracy (in case of segmentation tasks).
                if (self.__compute_area_weighted_accuracy and
                        self.__task_type == 'segmentation'):
                    total_num_correct_predictions_areaweighted = 0
                for data in self.__data_loader_test:
                    (primal_graph_batch, dual_graph_batch,
                     primal_edge_to_dual_node_idx_batch) = data
                    primal_graph_batch = primal_graph_batch.to(device)
                    dual_graph_batch = dual_graph_batch.to(device)
                    # Note: primal_graph_edge_to_dual_graph_idx_batch is a dict,
                    # and therefore cannot be moved to GPU.
                    outputs, log_info = self.__net(
                        primal_graph_batch=primal_graph_batch,
                        dual_graph_batch=dual_graph_batch,
                        primal_edge_to_dual_node_idx_batch=
                        primal_edge_to_dual_node_idx_batch)

                    targets = primal_graph_batch.y
                    # Compute number of correct predictions.
                    num_correct_predictions = compute_num_correct_predictions(
                        task_type=self.__task_type,
                        outputs=outputs,
                        targets=targets)
                    # The number of predictions corresponds to the number of
                    # samples in the batch in case of mesh classification (in
                    # which a single label is assigned to each shape) and to the
                    # number of total mesh faces in the batch in case of mesh
                    # segmentation (in which a label is assigned to each face).
                    num_predictions_in_batch = targets.shape[0]
                    total_num_correct_predictions += num_correct_predictions
                    total_num_predictions += num_predictions_in_batch
                    # If required, compute area-weighted accuracy.
                    if (self.__compute_area_weighted_accuracy and
                            self.__task_type == 'segmentation'):
                        face_areas = primal_graph_batch.face_areas
                        assert (face_areas is not None), (
                            "It is required that the primal graphs have a "
                            "non-None `face_areas` attribute when computing "
                            "area-weighted accuracy.")
                        face_to_batch_sample = primal_graph_batch.batch

                        (num_correct_predictions_areaweighted
                        ) = compute_num_correct_predictions(
                            task_type=self.__task_type,
                            outputs=outputs,
                            targets=targets,
                            face_areas=face_areas,
                            face_to_batch_sample=face_to_batch_sample)
                        (total_num_correct_predictions_areaweighted
                        ) += num_correct_predictions_areaweighted

                if (self.__task_type == 'classification'):
                    assert (total_num_predictions == len(
                        self.__data_loader_test.dataset))
                overall_accuracy = (total_num_correct_predictions /
                                    total_num_predictions)
                print(f"Test accuracy is {100. * overall_accuracy:.2f}%.")

                if (self.__compute_area_weighted_accuracy and
                        self.__task_type == 'segmentation'):
                    overall_areaweighted_accuracy = (
                        total_num_correct_predictions_areaweighted /
                        total_num_predictions)
                    print("Area-weighted test accuracy is "
                          f"{100. * overall_areaweighted_accuracy:.2f}%.")

                # Save TensorBoard summary.
                x_axis_test_accuracy = (self.__x_axis_last_summary_datapoint +
                                        num_batches_since_last_summary - 1)
                # Save summary for test accuracy/accuracies.
                self.__summary_writer.add_scalar('test_accuracy',
                                                 overall_accuracy,
                                                 x_axis_test_accuracy)
                if (self.__compute_area_weighted_accuracy and
                        self.__task_type == 'segmentation'):
                    self.__summary_writer.add_scalar(
                        'test_accuracy_area-weighted',
                        overall_areaweighted_accuracy, x_axis_test_accuracy)
            # Since at the beginning of each epoch
            # num_batches_since_last_summary is reset to 1 although it a few
            # batches from the previous epoch might not have been included in a
            # summary, update the x-axis of the summary to take into account
            # these batches.
            self.__x_axis_last_summary_datapoint += (
                num_batches_since_last_summary - 1)

            print("Training loss at the last batch in the epoch was "
                  f"{step_loss:.4f}")
            if (self.__checkpoint_epoch_frequency is not None):
                if ((self.__current_epoch %
                     self.__checkpoint_epoch_frequency == 0) or
                    (self.__current_epoch == self.__final_training_epoch)):
                    # Save checkpoint.
                    self.__save_checkpoint()
            # Update learning rate if necessary.
            if (self.__lr_scheduler is not None):
                # Set the learning-rate for the next epoch.
                self.__lr_scheduler.step(epoch=self.__current_epoch + 1)
            # Increment epoch index.
            self.__current_epoch += 1
            # Reset batch index.
            self.__current_batch = 0
            # If the epoch just terminated was completed on more than one run,
            # re-initialize dataset and data loader so that for the next epoch
            # they cover all the data in the dataset.
            if (self.__restarted_training_from_nonzero_batch):
                self.__reinitialize_dataset_and_dataloader()