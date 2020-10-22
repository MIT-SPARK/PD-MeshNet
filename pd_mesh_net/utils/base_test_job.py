import glob
import os
import pickle as pkl
import torch
import yaml

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.utils import (create_dataset, create_loss, create_model,
                               compute_num_correct_predictions,
                               get_epoch_most_recent_checkpoint,
                               get_epoch_and_batch_most_recent_checkpoint,
                               get_generic_checkpoint,
                               write_meshes_with_clusters)

training_parameters_filename = 'training_parameters.yml'


class BaseTestJob():
    r"""Base class for test jobs. Runs evaluation of a pretrained network on a
    dataset.

    Args:
        dataset_parameters (dict): Dictionary containing the parameters of the
            dataset from which the test data will be extracted.
        data_loader_parameters (dict): Dictionary containing the parameters of
            the data loader that will input samples from the dataset to the
            network.
        log_folder (str): Path of the folder from where the training parameters
            and checkpoints will be restored.
        training_job_name (str): Name of the training job from which a
            checkpoint and the parameters will be restored to perform the test
            job. The parameters associated to the training job must be in a
            subfolder with this name within the log folder (cf. argument
            `log_folder`). Specifically, the subfolder should contain the
            following files/subfolders:
            - A file with filename :obj:`training_parameters_filename`,
              containing the input parameters of the training job;
            - A subfolder 'checkpoints/', containing the checkpoints with the
              network weights.
        task_type (str): Type of task, used to perform the evaluation. Valid
            values are: 'classification', 'segmentation'.
        standardize_features_using_training_set (bool, optional): If True,
            input-feature standardization is performed using the mean and
            standard deviation of the primal-graph-/dual-graph- node features of
            the training set used in the previous job. If False, standardization
            will be performed or not depending on the input dataset parameters
            `compute_node_feature_stats` and `node_feature_stats_filename` (cf.
            `pd_mesh_net.utils.datasets.create_dataset`). (default: :obj:`True`)
        epoch_checkpoint_to_use (int, optional): If not None, the test job will
            be performed by restoring the checkpoint corresponding to the epoch
            `epoch_checkpoint_to_use`. Otherwise, the most recent available
            checkpoint will be used (default: :obj:`None`)
        batch_checkpoint_to_use (int, optional): If not None, also the argument
            `epoch_checkpoint_to_use` must be non-None. In this case, the test
            job will be performed by restoring the checkpoint corresponding to
            the epoch `epoch_checkpoint_to_use` and batch
            `batch_checkpoint_to_use`. Checkpoints of this form are generated if
            saving the model status before completion of an epoch (cf.
            `deep_mesh.utils.BaseTrainingJob`). If None, it will be assumed that
            the checkpoints are in the epoch-only format, i.e., they were saved
            only at the end of epochs. In this case, the checkpoint to load will
            be determined only by the argument `epoch_checkpoint_to_use`.
            (default: :obj:`None`)
        use_gpu (bool, optional): If True, the parameters from the checkpoint of
            the training job are loaded in the GPU and the network tensors are
            moved to the GPU; otherwise, parameters are loaded on the CPU, and
            network tensors are moved to the CPU. (default: :obj:`True`)
        save_clusterized_meshes (bool, optional): If True - and if the model on
            which the evaluation is performed contains a pooling layer - saves
            the meshes after pooling (i.e., with the face clusters) to file,
            with faces from the same cluster being colored with the same color.
            If the task is of segmentation type (cf. argument `task_type`), also
            another version of the mesh files is saved, in which each face
            cluster is colored according to the class that the network predicts
            for that cluster.
            (default: :obj:`False`)
        compute_area_weighted_accuracy (bool, optional): If True, and if the
            task is of segmentation type (cf. argument `task_type`), computes
            also a version of the accuracy where the contribution of each face
            is weighted by its area (cf.
            `pd_mesh_net.utils.losses.compute_num_correct_predictions`).
            Otherwise, the argument is ignored. (default: :obj:`True`)
        verbose (bool, optional): If True, displays status prints with a higher
            level of verbosity. (default: :obj:`False`)

    Attributes:
        None.
    """

    def __init__(self,
                 dataset_parameters,
                 data_loader_parameters,
                 log_folder,
                 training_job_name,
                 task_type,
                 standardize_features_using_training_set=True,
                 epoch_checkpoint_to_use=None,
                 batch_checkpoint_to_use=None,
                 use_gpu=True,
                 save_clusterized_meshes=False,
                 compute_area_weighted_accuracy=True,
                 verbose=True):
        self.__net = None
        self.__data_loader = None
        self.__loss = None
        self.__log_folder = log_folder
        self.__training_job_name = training_job_name
        self.__save_clusterized_meshes = save_clusterized_meshes
        assert (task_type in [
            "classification", "segmentation"
        ]), ("Invalid task type. Valid values are: 'classification', "
             "'segmentation'.")
        self.__compute_area_weighted_accuracy = (compute_area_weighted_accuracy
                                                 and
                                                 task_type == 'segmentation')
        self.__task_type = task_type
        (self.__standardize_features_using_training_set
        ) = standardize_features_using_training_set
        if (epoch_checkpoint_to_use is not None):
            assert (isinstance(epoch_checkpoint_to_use, int) and
                    epoch_checkpoint_to_use > 0)
        if (batch_checkpoint_to_use is not None):
            assert (epoch_checkpoint_to_use is not None), (
                "In case you wish to load a checkpoint saved before epoch "
                "termination, you must specify both the batch index and the "
                "epoch index of the checkpoint to load.")
            assert (isinstance(batch_checkpoint_to_use, int) and
                    batch_checkpoint_to_use > 0)
        self.__epoch_checkpoint_to_use = epoch_checkpoint_to_use
        self.__batch_checkpoint_to_use = batch_checkpoint_to_use
        self.__use_gpu = use_gpu
        self.__verbose = verbose
        # Check that the subfolder of the log folder associated with the
        # training job exists.
        complete_path_logs_training_job = os.path.join(self.__log_folder,
                                                       self.__training_job_name)
        if (not os.path.exists(complete_path_logs_training_job)):
            raise OSError(
                "Unable to find the log subfolder associated with training job "
                f"'{self.__training_job_name}' in the log folder "
                f"'{self.__log_folder}'. Exiting. ")

        self.__checkpoint_subfolder = os.path.join(self.__log_folder,
                                                   self.__training_job_name,
                                                   'checkpoints')

        self.__test_parameters = None

        self.__input_dataset_parameters = dataset_parameters
        self.__input_data_loader_parameters = data_loader_parameters

        # Initialize the test job.
        self.__initialize_test_job()

    def _initialize_components(self, network_parameters, dataset_parameters,
                               data_loader_parameters, loss_parameters):
        r"""Instantiates and initializes: network, dataset, data loader and
        loss.

        Args:
            network_parameters, dataset_parameters, data_loader_parameters,
            loss_parameters (dict): Input parameters used to construct and
                initialize the network, the dataset, the data loader and the
                loss.
        
        Returns:
            None.
        """
        # Initialize model.
        assert ('should_initialize_weights' not in network_parameters), (
            "Network parameters should not contain the parameter "
            "'should_initialize_weights', as weights will be automatically "
            "initialized or not, depending on whether training is resumed "
            "from a previous job or not.")
        if (self.__verbose):
            print("Initializing network...")
        if (self.__save_clusterized_meshes):
            network_contains_at_least_one_pooling_layer = False
            if ('num_primal_edges_to_keep' in network_parameters and
                    network_parameters['num_primal_edges_to_keep'] is not None):
                num_pooling_layers = len([
                    threshold for threshold in
                    network_parameters['num_primal_edges_to_keep']
                    if threshold is not None
                ])
                network_contains_at_least_one_pooling_layer |= (
                    num_pooling_layers >= 1)
            elif ('fractions_primal_edges_to_keep' in network_parameters and
                  network_parameters['fractions_primal_edges_to_keep'] is
                  not None):
                num_pooling_layers = len([
                    threshold for threshold in
                    network_parameters['fractions_primal_edges_to_keep']
                    if threshold is not None
                ])
                network_contains_at_least_one_pooling_layer |= (
                    num_pooling_layers >= 1)
            elif ('primal_attention_coeffs_thresholds' in network_parameters and
                  network_parameters['primal_attention_coeffs_thresholds'] is
                  not None):
                num_pooling_layers = len([
                    threshold for threshold in
                    network_parameters['primal_attention_coeffs_thresholds']
                    if threshold is not None
                ])
                network_contains_at_least_one_pooling_layer |= (
                    num_pooling_layers >= 1)
            assert (network_contains_at_least_one_pooling_layer), (
                "Please use at least one pooling layer in the test model to "
                "save the clusterized meshes.")
            # Add to the input parameters of the network the flag that specifies
            # that the node-to-cluster correspondences should be returned.
            network_parameters['return_node_to_cluster'] = True

        self.__net = create_model(should_initialize_weights=False,
                                  **network_parameters)
        if ('log_ratios_new_old_primal_nodes' in network_parameters and
                network_parameters['log_ratios_new_old_primal_nodes'] is True):
            self.__are_ratios_new_old_primal_nodes_logged = True
        else:
            self.__are_ratios_new_old_primal_nodes_logged = False
        # Move network to GPU if necessary.
        if (self.__use_gpu):
            self.__net.to("cuda")
        else:
            self.__net.to("cpu")
        # Initialize dataset.
        if (self.__verbose):
            print("Initializing dataset...")
        if (dataset_parameters['train'] == True):
            print("\033[93mNote: running evaluation on a 'train' split! If you "
                  "instead want to use the 'test' split of the dataset, please "
                  "set the dataset parameter 'train' as False.\033[0m")
            self.__split = 'train'
        else:
            self.__split = 'test'
        if (self.__standardize_features_using_training_set):
            assert (
                'compute_node_feature_stats' not in dataset_parameters or
                not dataset_parameters['compute_node_feature_stats']
            ), ("Setting argument 'standardize_features_using_training_set' of "
                "the test job to True is incompatible with dataset parameter "
                "'compute_node_feature_stats' = True.")
            # Perform input-feature normalization using the statistics from
            # the training set.
            print("\033[92mWill perform input-feature standardization using "
                  "the provided mean and standard deviation of the "
                  "primal-graph-/dual-graph- node features of the training "
                  f"set (file '{self.__training_params_filename}').\033[0m")
            primal_mean = dataset_parameters.pop('primal_mean_train')
            primal_std = dataset_parameters.pop('primal_std_train')
            dual_mean = dataset_parameters.pop('dual_mean_train')
            dual_std = dataset_parameters.pop('dual_std_train')
            dataset_parameters['compute_node_feature_stats'] = False
            dataset, _ = create_dataset(**dataset_parameters)
        else:
            if ('compute_node_feature_stats' in dataset_parameters and
                    not dataset_parameters['compute_node_feature_stats']):
                # No feature standardization.
                dataset, _ = create_dataset(**dataset_parameters)
                primal_mean = primal_std = dual_mean = dual_std = None
                print("\033[93mNote: no input-feature standardization will be "
                      "performed! If you wish to use standardization instead, "
                      "please set the argument "
                      "'standardize_features_using_training_set' of the test "
                      "job to True or set the dataset-parameter "
                      "`compute_node_feature_stats` to True.\033[0m")
            else:
                print("\033[93mNote: input-feature standardization will be "
                      "performed using the mean and standard deviation of the "
                      "primal-graph-/dual-graph- node features of the test "
                      "set! If you wish to use those of the training set "
                      "instead, please set the argument "
                      "'standardize_features_using_training_set' of the test "
                      "job to True.\033[0m")
                dataset, (primal_mean, primal_std, dual_mean,
                          dual_std) = create_dataset(**dataset_parameters)
        # Initialize data loader.
        assert (len(
            set(['primal_mean', 'primal_std', 'dual_mean', 'dual_std']) &
            set(data_loader_parameters)) == 0), (
                "Data-loader parameters should not contain any of the "
                "following parameters, as they will be automatically computed "
                "from the dataset or restored from the previous training job, "
                "if set to do so: 'primal_mean', "
                "'primal_std', 'dual_mean', 'dual_std'.")
        if (self.__verbose):
            print("Initializing data loader...")
        # Add to the input parameters of the data-loader the flag that specifies
        # that the indices of the sample in the dataset should be returned when
        # iterating on it.
        data_loader_parameters['return_sample_indices'] = True

        self.__data_loader = DualPrimalDataLoader(dataset=dataset,
                                                  primal_mean=primal_mean,
                                                  primal_std=primal_std,
                                                  dual_mean=dual_mean,
                                                  dual_std=dual_std,
                                                  **data_loader_parameters)
        # Initialize loss.
        if (loss_parameters is not None):
            if (self.__verbose):
                print("Initializing loss...")
            self.__loss = create_loss(**loss_parameters)

    def __initialize_test_job(self):
        r"""Initializes the components necessary for the test job, and restores
        the parameters from the previous training job.

        Args:
            None.

        Returns:
            None.
        """
        if (self.__verbose):
            print("Initializing test job...")
        training_job_complete_path = os.path.join(self.__log_folder,
                                                  self.__training_job_name)
        training_parameters_full_filename = os.path.join(
            training_job_complete_path, training_parameters_filename)
        # Load training parameters of previous training job.
        previous_training_parameters = self.__load_training_parameters()
        # Verify that the previous job contains the information about the
        # network and the loss, that need to be restored.
        for component_to_initialize in ['network', 'loss']:
            # The components must be loaded from the previous training job or
            # initialized with the input parameters of the current test job.
            assert (f'{component_to_initialize}_parameters' in
                    previous_training_parameters)
        dataset_parameters = self.__input_dataset_parameters
        data_loader_parameters = self.__input_data_loader_parameters
        network_parameters = previous_training_parameters['network_parameters']
        loss_parameters = previous_training_parameters['loss_parameters']
        # Check that the type of graphs (i.e., the dual-graph configuration) in
        # the dataset is the same as the one for which the trained network was
        # instantiated.
        single_dual_nodes_dataset = dataset_parameters['single_dual_nodes']
        single_dual_nodes_network = network_parameters['single_dual_nodes']
        undirected_dual_nodes_dataset = dataset_parameters[
            'undirected_dual_edges']
        undirected_dual_nodes_network = network_parameters[
            'undirected_dual_edges']
        assert (
            single_dual_nodes_dataset == single_dual_nodes_network and
            undirected_dual_nodes_dataset == undirected_dual_nodes_network), (
                "The dual-graph configuration in the dataset must be the same "
                "as the one for which the network was instantiated.")

        # If the clusterized meshes should be saved or if the per-mesh
        # statistics should be displayed (which is done when the batch size is
        # 1), set the dataset and data loader to return the indices of the
        # samples in the dataset.
        if (self.__save_clusterized_meshes or
                data_loader_parameters['batch_size'] == 1):
            dataset_parameters['return_sample_indices'] = True
            data_loader_parameters['return_sample_indices'] = True
        else:
            dataset_parameters['return_sample_indices'] = False
            data_loader_parameters['return_sample_indices'] = False

        if (self.__standardize_features_using_training_set):
            # Add the mean/std statistics of the training set as input
            # parameters of the dataset.
            assert (
                'primal_mean_train' not in dataset_parameters and
                'primal_std_train' not in dataset_parameters and
                'dual_mean_train' not in dataset_parameters and
                'dual_std_train' not in dataset_parameters
            ), ("The input dataset parameters should not contain any of the "
                "keywords 'primal_mean_train', 'primal_std_train', "
                "'dual_mean_train', 'dual_std_train', as these statistics will "
                "be retrieved from the previous training job if required.")
            assert (
                previous_training_parameters['dataset_parameters']
                ['compute_node_feature_stats'] and
                previous_training_parameters['dataset_parameters']
                ['node_feature_stats_filename'] is not None), (
                    "Unable to retrieve mean/std statistics of the previous "
                    "training job, as these were not computed or not saved to "
                    "disk. Please select a different training job, or set the "
                    "argument 'standardize_features_using_training_set' of the "
                    "test job to False.")
            # Load the statistics from the file where they were saved.
            self.__training_params_filename = previous_training_parameters[
                'dataset_parameters']['node_feature_stats_filename']
            try:

                with open(self.__training_params_filename, 'rb') as f:
                    training_set_stats = pkl.load(f)
            except IOError:
                raise IOError("Unable to open training-set statistics file "
                              f"'{self.__training_params_filename}'.")
            for graph_type in ['primal', 'dual']:
                for statistic in ['mean', 'std']:
                    full_stat_name = f'{graph_type}_{statistic}'
                    try:
                        (dataset_parameters[f'{full_stat_name}_train']
                        ) = training_set_stats[full_stat_name]
                    except KeyError:
                        raise KeyError(
                            "Training-set parameters at "
                            f"'{self.__training_params_filename}' do not "
                            f"contain the required statistic {full_stat_name}.")

        # Initialize dataset, data loader, network and loss.
        self._initialize_components(
            dataset_parameters=dataset_parameters,
            data_loader_parameters=data_loader_parameters,
            network_parameters=network_parameters,
            loss_parameters=loss_parameters)

        # Restore the network weights.
        if (self.__epoch_checkpoint_to_use is None):
            # If no checkpoint-epoch was manually specified, choose the most
            # recent available one.
            epoch_most_recent_checkpoint = get_epoch_most_recent_checkpoint(
                checkpoint_subfolder=self.__checkpoint_subfolder)
            batch_most_recent_checkpoint = None
            if (epoch_most_recent_checkpoint is None):
                # Check if the checkpoints are in the epoch-and-batch format.
                (epoch_most_recent_checkpoint, batch_most_recent_checkpoint
                ) = get_epoch_and_batch_most_recent_checkpoint(
                    checkpoint_subfolder=self.__checkpoint_subfolder)

            self.__epoch_checkpoint_to_use = epoch_most_recent_checkpoint
            self.__batch_checkpoint_to_use = batch_most_recent_checkpoint

        if (self.__epoch_checkpoint_to_use is not None):
            # A checkpoint with a recognized format was found.
            if (self.__batch_checkpoint_to_use is None):
                # Epoch-only format.
                checkpoint_filename = os.path.join(
                    self.__checkpoint_subfolder,
                    f'checkpoint_{self.__epoch_checkpoint_to_use:04d}.pth')
            else:
                # Epoch-and-batch format.
                checkpoint_filename = os.path.join(
                    self.__checkpoint_subfolder,
                    f'checkpoint_{self.__epoch_checkpoint_to_use:04d}'
                    f'_batch_{self.__batch_checkpoint_to_use:05d}.pth')
        else:
            # No checkpoints with a recognized format were found. Check for the
            # existence of a single checkpoint with generic valid filename in
            # the folder.
            checkpoint_filename = get_generic_checkpoint(
                checkpoint_subfolder=self.__checkpoint_subfolder)

            assert (checkpoint_filename is
                    not None), ("No saved checkpoints were found for training "
                                f"job '{self.__training_job_name}'. Maybe no "
                                "epoch was completed? Please specify a "
                                "different name for the training job.")
        try:
            device = "cuda" if self.__use_gpu else "cpu"
            checkpoint = torch.load(checkpoint_filename, map_location=device)
            if (self.__epoch_checkpoint_to_use is not None):
                assert (checkpoint['epoch'] == self.__epoch_checkpoint_to_use)
        except OSError:
            raise OSError(f"Unable to open checkpoint '{checkpoint_filename}'. "
                          "Exiting.")
        self.__net.load_state_dict(checkpoint['net'])

        batch_msg = ""
        epoch_msg = ""
        if (self.__batch_checkpoint_to_use is not None):
            batch_msg = f" and batch {self.__batch_checkpoint_to_use}"
        if (self.__epoch_checkpoint_to_use is not None):
            epoch_msg = f" of epoch {self.__epoch_checkpoint_to_use}"
        print(f"\033[92mPerfoming test using checkpoint{epoch_msg}"
              f"{batch_msg} from previous job (parameters stored in file "
              f"'{training_parameters_full_filename}').\033[00m")

    def __load_training_parameters(self):
        r"""Loads the training parameters from the previous training job, by
        reading the associated file in the log folder.

        Args:
            None.

        Returns:
            previous_training_parameters (dict): Dictionary having as keys
                dictionaries representing the parameters of the network, the
                dataset, the data loader, the optimizer, the learning-rate
                scheduler and the loss used in the training job.
        """
        # Read layer-input-parameter file.
        parameter_filename = os.path.join(self.__log_folder,
                                          self.__training_job_name,
                                          training_parameters_filename)
        try:
            with open(parameter_filename, 'rb') as f:
                previous_training_parameters = yaml.load(f,
                                                         Loader=yaml.FullLoader)
                return previous_training_parameters
        except IOError:
            raise IOError("Unable to open previous-training-parameter file at "
                          f"'{parameter_filename}'. Exiting.")

    def test(self, return_accuracy=False):
        r"""Performs the test job, feeding the test data to the pretrained
        network and evaluating the accuracy.

        Args:
            return_accuracy (bool): If True, the function returns the overall
                accuracy and area-weighted accuracy (if applicable to the task,
                and if the class argument :obj:`compute_area_weighted_accuracy`
                is True; otherwise `None` is returned). If False, the function
                returns nothing (default: :obj:`False`).

        Returns:
            None.
        """
        # Set up the network for testing.
        self.__net.eval()

        device = "cuda" if self.__use_gpu else "cpu"
        # Set up the folder to which the clusterized test meshes should be saved
        # if required.
        if (self.__save_clusterized_meshes):
            output_folder = os.path.join(self.__log_folder,
                                         self.__training_job_name,
                                         'clusterized_test_meshes')
            if (not os.path.exists(output_folder)):
                os.makedirs(output_folder)
            # In case of segmentation tasks, also set up a folder where to store
            # the meshes with clusters colored based on their predicted class
            # label.
            if (self.__task_type == 'segmentation'):
                output_folder_segments = os.path.join(self.__log_folder,
                                                      self.__training_job_name,
                                                      'segmented_test_meshes')
                if (not os.path.exists(output_folder_segments)):
                    os.makedirs(output_folder_segments)

        # Test.
        with torch.no_grad():
            num_iterations = len(self.__data_loader)
            total_num_correct_predictions = 0
            total_num_predictions = 0
            # Optionally count also the number of predictions according to
            # area-weighted accuracy (in case of segmentation tasks).
            if (self.__compute_area_weighted_accuracy):
                total_num_correct_predictions_areaweighted = 0
            for iteration_idx, data in enumerate(self.__data_loader):
                print(f"\tIteration no. {iteration_idx+1}/{num_iterations}")
                (primal_graph_batch, dual_graph_batch,
                 primal_edge_to_dual_node_idx_batch, sample_indices) = data
                primal_graph_batch = primal_graph_batch.to(device)
                dual_graph_batch = dual_graph_batch.to(device)
                # Note: primal_graph_edge_to_dual_graph_idx_batch is a dict, and
                # therefore cannot be moved to GPU.

                outputs, log_info = self.__net(
                    primal_graph_batch=primal_graph_batch,
                    dual_graph_batch=dual_graph_batch,
                    primal_edge_to_dual_node_idx_batch=
                    primal_edge_to_dual_node_idx_batch)

                # Optionally save the clusterized meshes to file.
                if (self.__save_clusterized_meshes):
                    if (isinstance(
                            log_info.node_to_cluster[list(
                                log_info.node_to_cluster.keys())[0]], list)):
                        node_to_cluster = log_info.node_to_cluster
                    else:
                        node_to_cluster = {
                            k: v.tolist()
                            for k, v in log_info.node_to_cluster.items()
                        }
                    write_meshes_with_clusters(
                        dataset=self.__data_loader.dataset,
                        sample_indices=sample_indices,
                        node_to_local_sample_idx=primal_graph_batch.batch.
                        tolist(),
                        node_to_clusters=node_to_cluster,
                        output_folder=output_folder)
                    if (self.__task_type == 'segmentation'):
                        # If the task is a segmentation task, also save the
                        # clusterized meshes with each cluster colored according
                        # to the predicted class of that cluster.
                        write_meshes_with_clusters(
                            dataset=self.__data_loader.dataset,
                            sample_indices=sample_indices,
                            node_to_local_sample_idx=primal_graph_batch.batch.
                            tolist(),
                            node_to_clusters=node_to_cluster,
                            output_folder=output_folder_segments,
                            perclass_scores=outputs)

                targets = primal_graph_batch.y
                # Compute number of correct predictions.
                num_correct_predictions = compute_num_correct_predictions(
                    task_type=self.__task_type,
                    outputs=outputs,
                    targets=targets)
                # The number of predictions corresponds to the number of samples
                # in the batch in case of mesh classification (in which a single
                # label is assigned to each shape) and to the number of total
                # mesh faces in the batch in case of mesh segmentation (in which
                # a label is assigned to each face).
                num_predictions_in_batch = targets.shape[0]
                total_num_correct_predictions += num_correct_predictions
                total_num_predictions += num_predictions_in_batch
                # If required, compute area-weighted accuracy.
                if (self.__compute_area_weighted_accuracy):
                    face_areas = primal_graph_batch.face_areas
                    assert (face_areas is not None), (
                        "It is required that the primal graphs have a non-None "
                        "`face_areas` attribute when computing area-weighted "
                        "accuracy.")
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

                if (len(sample_indices) == 1):
                    # If using batch size 1, print the accuracy for each input
                    # sample.
                    if (self.__split == 'test'):
                        mesh_filename = (
                            self.__data_loader.dataset.
                            processed_file_names_test[
                                3 * sample_indices[0]].split('_primal')[0])
                    else:
                        mesh_filename = (
                            self.__data_loader.dataset.
                            processed_file_names_train[
                                3 * sample_indices[0]].split('_primal')[0])
                    if (self.__task_type == 'classification'):
                        assert (num_correct_predictions <= 1 and
                                num_predictions_in_batch == 1)
                        additional_message = ""
                        if (num_correct_predictions == 0):
                            additional_message = " not"
                        print(
                            f"Mesh \"{mesh_filename}\" was{additional_message} "
                            "correctly classified.")
                    elif (self.__task_type == 'segmentation'):
                        additional_message_area_weighted = ""
                        if (self.__compute_area_weighted_accuracy):
                            additional_message_area_weighted = (
                                f" ({num_correct_predictions_areaweighted:.2f}/"
                                f"{num_predictions_in_batch} with "
                                "area-weighted accuracy)")
                        print(
                            f"Mesh \"{mesh_filename}\": "
                            f"{num_correct_predictions}/"
                            f"{num_predictions_in_batch} correctly classified "
                            f"faces{additional_message_area_weighted}.")

            if (self.__task_type == 'classification'):
                assert (total_num_predictions == len(
                    self.__data_loader.dataset))
            overall_accuracy = (total_num_correct_predictions /
                                total_num_predictions)
            print(f"Overall accuracy is {100. * overall_accuracy:.2f}%.")

            if (self.__compute_area_weighted_accuracy):
                overall_areaweighted_accuracy = (
                    total_num_correct_predictions_areaweighted /
                    total_num_predictions)
                print("Overall area-weighted accuracy is "
                      f"{100. * overall_areaweighted_accuracy:.2f}%.")
            else:
                overall_areaweighted_accuracy = None

            if (return_accuracy):
                return overall_accuracy, overall_areaweighted_accuracy
