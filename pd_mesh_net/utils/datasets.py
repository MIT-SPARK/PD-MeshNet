import numpy as np
import os
import pickle as pkl
import pymesh
import torch
import warnings

from pd_mesh_net.datasets import (Shrec2016DualPrimal, CubesDualPrimal,
                                  CosegDualPrimal, HumanSegDualPrimal)
from pd_mesh_net.utils.colors import SegmentationColors


def create_dataset(dataset_name,
                   compute_node_feature_stats=True,
                   node_feature_stats_filename=None,
                   **dataset_params):
    r"""Creates an instance of the input dataset with the input parameters.

    Args:
        dataset_name (str): Name that identifies the dataset. Valid values are:
            `shrec_16` (SHREC2016 dataset), `cubes` (Cube engraving dataset from
            MeshCNN), `coseg` (COSEG dataset), `human_seg` (Human Body
            Segmentation dataset).
        compute_node_feature_stats (bool, optional): If True, the mean and
            standard deviation of the node features in the dataset are computed
            (cf. function :obj:`compute_mean_and_std`). (default: :obj:`True`)
        node_feature_stats_filename (str, optional): If not None, filename of
            the file containing the mean and standard deviation of the node
            features in the dataset (cf. function :obj:`compute_mean_and_std`).
            The argument is considered only if the argument
            `compute_node_feature_stats` is True. (default: :obj:`None`)
        ...
        Optional parameters of the datasets.

    Returns:
        dataset (torch_geometric.data.Dataset): The instance of the dataset with
            the input parameters.
        primal_graph_mean, primal_graph_std, dual_graph_mean, dual_graph_std
            (tuple of numpy array or None): If the argument
            :obj:`compute_node_feature_stats` is True, statistics about the node
            features in the dataset (cf. docs above); otherwise, None.
    """
    if (dataset_name == 'shrec_16'):
        dataset = Shrec2016DualPrimal(**dataset_params)
    elif (dataset_name == 'cubes'):
        dataset = CubesDualPrimal(**dataset_params)
    elif (dataset_name == 'coseg'):
        dataset = CosegDualPrimal(**dataset_params)
    elif (dataset_name == 'human_seg'):
        dataset = HumanSegDualPrimal(**dataset_params)
    else:
        raise KeyError(
            f"No known dataset can be generated with the name '{dataset_name}'."
        )

    node_statistics = None

    if (compute_node_feature_stats):
        dataset_params = dataset.input_parameters
        (primal_graph_mean, primal_graph_std, dual_graph_mean,
         dual_graph_std) = compute_mean_and_std(
             dataset=dataset,
             dataset_params=dataset_params,
             filename=node_feature_stats_filename)
        node_statistics = (primal_graph_mean, primal_graph_std, dual_graph_mean,
                           dual_graph_std)
    return dataset, node_statistics


def compute_mean_and_std(dataset=None, dataset_params=None, filename=None):
    r"""Computes mean and standard deviation of the primal-graph- and
    dual-graph- node features. The results can be cached to disk and later
    loaded to avoid recomputation. Performs similar operations as `MeshCNN
    <https://github.com/ranahanocka/MeshCNN/>`_.


    Args:
        dataset (torch_geometric.data.Dataset, optional): Dataset on which to
            compute the mean and standard deviation of the primal-graph- and
            dual-graph node features. Its __getitem__ method should return four
            values, representing respectively the primal graph, the dual graph,
            the primal-edge-to-dual-node-index dictionary and the index of the
            sample in the dataset. It can optionally be not passed, in case the
            filename of a cached file and the dataset parameters are provided
            (cf. arguments `filename` and `dataset_params`).
            (default: :obj:`None`)
        dataset_params (dict, optional): Dictionary containing the parameters
            associated to input dataset, that will be used, in case the argument
            `filename` is not :obj:`None`, to save the parameters to disk
            together with the mean and standard deviation - if the file is
            nonexistent - or to compare them with the parameters of the file -
            if it already exists -, to ensure that the statistics retrieved from
            disk are compatible with the dataset being used. In the latter case,
            all the parameters in `dataset_params` should be contained in the
            file and match with the parameters therein, and all the parameters
            in the file should be contained in `dataset_params` and match with
            these parameters. Furthermore, the size of the dataset should match
            the size of the dataset from which the file was created.
            (default: :obj:`None`)
        filename (str, optional): If not None, the functions searchs for a file
            with this filename. If the latter exists, mean and standard
            deviation are retrieved from the file, if compatible with the
            current dataset (cf. argument `dataset_params`). Otherwise, the mean
            and standard deviation computed - together with the dataset
            parameters and the dataset size - are saved to disk with this
            filename.
            (default: :obj:`None`)
    
    Returns:
        primal_graph_mean, primal_graph_std (numpy array of shape
            `[num_primal_features,]`, where `num_primal_features` is the
            dimensionality of the primal-graph node features): Mean and standard
            deviation of the primal-graph node features.
        dual_graph_mean, dual_graph_std (numpy array of shape
            `[num_dual_features,]`, where `num_dual_features` is the
            dimensionality of the dual-graph node features): Mean and standard
            deviation of the dual-graph node features.
    """
    if (dataset_params is not None):
        # Check that the mean and standard deviation are not being passed as
        # parameters of the input dataset.
        for param_keyword in ['mean', 'std']:
            for graph_keyword in ['primal', 'dual']:
                keyword = f"{graph_keyword}_{param_keyword}"
                if (keyword in dataset_params):
                    raise KeyError(
                        f"The parameters of the input dataset already contain "
                        f"an entry '{keyword}'. Exiting.")
    file_exists = False
    if (filename is not None):
        # Load the data from disk, if the file exists.
        if (os.path.exists(filename)):
            file_exists = True
    if (file_exists):
        assert (dataset_params is not None)
        assert (isinstance(dataset_params, dict))
        try:
            with open(filename, "rb") as f:
                data_from_disk = pkl.load(f)
        except IOError:
            raise IOError(f"Error loading cache mean-std file '{filename}'. "
                          "Exiting.")
        # Check that the file contains the mean and standard deviation.
        for keyword in ['primal', 'dual']:
            if (f'{keyword}_mean' not in data_from_disk):
                raise KeyError(
                    f"Cached file '{filename}' does not contain the mean of "
                    f"the {keyword}-graph node features. Exiting.")
            if (f'{keyword}_std' not in data_from_disk):
                raise KeyError(
                    f"Cached file '{filename}' does not contain the standard "
                    f"deviation of the {keyword}-graph node features. Exiting.")
        # Check that the size of the dataset is compatible.
        try:
            size_dataset_of_file = data_from_disk['dataset_size']
        except KeyError:
            raise KeyError(
                f"Cached file '{filename}' does not contain the dataset size. "
                f"Exiting.")
        current_dataset_size = len(dataset)
        if (size_dataset_of_file != current_dataset_size):
            warnings.warn("Please note that the current dataset has size "
                          f"{current_dataset_size}, whereas the cached file ("
                          f"'{filename}') was generated from a dataset of size "
                          f"{size_dataset_of_file}.")

        # Check that the parameters match.
        for param_name, param_value in dataset_params.items():
            if (param_name not in data_from_disk):
                raise KeyError(
                    f"Could not find dataset parameter {param_name} in the "
                    f"cached file '{filename}'. Please provide a different "
                    "filename.")
            else:
                if (data_from_disk[param_name] != param_value):
                    raise ValueError(
                        f"Cached file '{filename}' is incompatible with "
                        f"current dataset. Expected parameter {param_name} to "
                        f"be {param_value}, found "
                        f"{data_from_disk[param_name]}. Please provide a "
                        "different filename.")
        for cached_param_name in dataset_params.keys():
            if (cached_param_name in [
                    'primal_mean', 'primal_std', 'dual_mean', 'dual_std'
            ]):
                continue
            if (cached_param_name not in dataset_params):
                raise KeyError(
                    f"Cached file '{filename}' is incompatible with "
                    "current dataset, as it contains parameter "
                    f"{cached_param_name}, which is missing in the input "
                    "dataset. Please provide a different filename.")
        # Return the cached data.
        primal_graph_mean = data_from_disk['primal_mean']
        primal_graph_std = data_from_disk['primal_std']
        dual_graph_mean = data_from_disk['dual_mean']
        dual_graph_std = data_from_disk['dual_std']
    else:
        # Compute the mean and standard deviation of the node features from
        # scratch.
        primal_graph_xs = torch.empty([0, dataset[0][0].x.shape[1]])
        dual_graph_xs = torch.empty([0, dataset[0][1].x.shape[1]])
        for sample_idx, (primal_graph, dual_graph, _, _) in enumerate(dataset):
            primal_graph_xs = torch.cat([primal_graph_xs, primal_graph.x])
            dual_graph_xs = torch.cat([dual_graph_xs, dual_graph.x])
        assert (len(dataset) == sample_idx + 1)
        primal_graph_mean = primal_graph_xs.mean(axis=0).numpy()
        primal_graph_std = primal_graph_xs.std(axis=0).numpy()
        dual_graph_mean = dual_graph_xs.mean(axis=0).numpy()
        dual_graph_std = dual_graph_xs.std(axis=0).numpy()
        assert (np.all(
            primal_graph_std > 10 * np.finfo(primal_graph_std.dtype).eps))
        assert (np.all(
            dual_graph_std > 10 * np.finfo(dual_graph_std.dtype).eps))

        if (filename is not None):
            # Save the values to file, together with the dataset parameters and
            # the dataset size, if required.
            if (dataset_params is None):
                dataset_params = {}
            output_values = {
                **dataset_params, 'primal_mean': primal_graph_mean,
                'primal_std': primal_graph_std,
                'dual_mean': dual_graph_mean,
                'dual_std': dual_graph_std,
                'dataset_size': sample_idx + 1
            }
            try:
                with open(filename, 'wb') as f:
                    pkl.dump(output_values, f)
            except IOError:
                raise IOError(
                    "Unable to save mean-std data to file at location "
                    f"{filename}.")

    return (primal_graph_mean, primal_graph_std, dual_graph_mean,
            dual_graph_std)


def write_meshes_with_clusters(dataset,
                               sample_indices,
                               node_to_local_sample_idx,
                               node_to_clusters,
                               output_folder,
                               perclass_scores=None):
    r"""Given the face clusters formed after pooling meshes with one or multiple
    DualPrimalEdgePooling layer, the indices of the samples in the dataset, as
    well as the dataset, writes an output file for each mesh, in which faces
    from the same face cluster are assigned the same color. Optionally, for
    segmentation tasks, unnormalized per-class scores can be given as input for
    each face cluster. If this is the case, each class is associated to a
    different color, and each cluster is colored with the color of the class
    that has the largest score in that cluster.

    Args:
        dataset (torch_geometric.data.Dataset): Dataset from which the samples
            with the input face clusters were extracted.
        sample_indices (list of int): The i-th element contains the index in the
            input dataset of the i-th element in the batch being considered.
        node_to_sample_idx (list of int): The i-th element contains the index in
            the batch of the mesh to which the i-th node of the primal graph
            inputted to the network corresponds (i.e., the i-th node of the
            input primal graph, before any pooling, corresponds to a face in
            this mesh). The value `sample_indices[node_to_sample_idx[i]]`
            corresponds to the index of the sample in the dataset from which the
            i-th node is taken.
        node_to_clusters (dict of list): The i-th element of
            `node_to_clusters[j]` contains the index of the 'face cluster' to
            which the i-th node in the primal graph inputted to the j-th pooling
            layer belongs in the primal graph outputted by the same pooling
            layer.
        output_folder (str): Path of the folder where the output colored meshes
            should be saved.
        perclass_scores (torch.Tensor of shape
            `[num_faces, num_classes]` - where `num_faces` is the number of
            faces in the batch and `num_classes` is the number of possible
            classes -, optional): If not None, the element
            `perclass_scores[i, j]` contains the unnormalized score associated
            to assigning the `i`-th face to the `j`-th class.
            (default: :obj:`None`)
    
    Returns:
        None.
    """

    def __save_previous_mesh():
        assert (num_faces_colored == mesh.num_faces)
        # - Set the color attributes of the mesh.
        mesh.set_attribute("face_blue", blue_values)
        mesh.set_attribute("face_green", green_values)
        mesh.set_attribute("face_red", red_values)
        mesh.set_attribute("face_alpha", alpha_values)
        # - Save the mesh.
        if (not os.path.exists(output_folder)):
            os.makedirs(output_folder)
        output_filename = os.path.join(
            output_folder, f"{filename_root.split('/')[-1]}_clusterized.ply")
        pymesh.save_mesh(output_filename,
                         mesh,
                         "face_blue",
                         "face_green",
                         "face_red",
                         "face_alpha",
                         ascii=True)

    # For each pooling layer, map the output nodes to the input nodes, starting
    # from the last one (i.e., the one closest to the end of the network) and
    # going backwards, until the output nodes of the last layer have been mapped
    # to the input nodes of the first layer.
    last_to_first_pooling_layer_indices = sorted(node_to_clusters.keys())[::-1]
    node_to_clusters_next_layer = np.array(
        node_to_clusters[last_to_first_pooling_layer_indices[0]])
    for pooling_layer_idx in last_to_first_pooling_layer_indices[1:]:
        node_to_clusters_next_layer = node_to_clusters_next_layer[
            node_to_clusters[pooling_layer_idx]]

    assert (len(node_to_local_sample_idx) == len(node_to_clusters_next_layer))
    # Check if the split of the input dataset is train or test.
    is_dataset_train = dataset.input_parameters['train']

    # If per-class scores are provided, find the class label with highest score
    # for each face.
    face_to_class = None
    if (perclass_scores is not None):
        face_to_class = perclass_scores.argmax(axis=1).tolist()

    # Iterate over all the input samples.
    previous_local_sample_idx = None
    face_idx_in_batch = 0
    for local_sample_idx, cluster_idx in zip(node_to_local_sample_idx,
                                             node_to_clusters_next_layer):
        if (local_sample_idx != previous_local_sample_idx):
            # New sample.
            node_idx = 0
            # - Save the previous mesh if any.
            if (previous_local_sample_idx is not None):
                __save_previous_mesh()
            # - Retrieve the associated mesh in the dataset.
            sample_idx_in_dataset = sample_indices[local_sample_idx]
            if (is_dataset_train):
                mesh_filename = dataset.processed_file_names_train[
                    3 * sample_idx_in_dataset]
            else:
                mesh_filename = dataset.processed_file_names_test[
                    3 * sample_idx_in_dataset]
            # - Extract the root of the filename from the name of the
            #   primal-graph file.
            filename_root = mesh_filename.split('_primal.pt')
            assert (len(filename_root) == 2)
            filename_root = filename_root[0]
            mesh_filename = os.path.join(dataset.processed_dir,
                                         f"{filename_root}.obj")
            # Load the mesh.
            mesh = pymesh.load_mesh(mesh_filename)
            # Set up the structures to store the face colors.
            blue_values = np.zeros([mesh.num_faces])
            green_values = np.zeros([mesh.num_faces])
            red_values = np.zeros([mesh.num_faces])
            alpha_values = np.zeros([mesh.num_faces], dtype=np.float64)
            mesh.add_attribute("face_blue")
            mesh.add_attribute("face_green")
            mesh.add_attribute("face_red")
            mesh.add_attribute("face_alpha")
            if (face_to_class is None):
                # Reset the cluster colors.
                cluster_colors = {}
                colors_used = set()
            num_faces_colored = 0

        if (face_to_class is not None):
            # If per-class scores are provided, color the face with the color
            # associated to the class with highest score for that face.
            cluster_color = SegmentationColors.colors[
                face_to_class[face_idx_in_batch]]
        else:
            # If no per-class scores are provided, retrieve the color of the
            # cluster corresponding to the current node or assign one to it if
            # it does not have one yet, avoiding assigning the same color to two
            # different clusters.
            if (cluster_idx not in cluster_colors):
                assert (
                    len(colors_used) < 256**
                    3), f"All available {256**3} colors have already been used."
                was_new_color_already_used = True
                while (was_new_color_already_used):
                    new_color = np.random.randint(0, 255, size=3)
                    was_new_color_already_used = tuple(
                        new_color.tolist()) in colors_used
                cluster_colors[cluster_idx] = new_color
                colors_used.add(tuple(new_color.tolist()))

            cluster_color = cluster_colors[cluster_idx]
        # Color the face.
        blue_values[node_idx] = cluster_color[0]
        green_values[node_idx] = cluster_color[1]
        red_values[node_idx] = cluster_color[2]

        num_faces_colored += 1
        node_idx += 1
        face_idx_in_batch += 1
        previous_local_sample_idx = local_sample_idx

    # Save the last mesh.
    __save_previous_mesh()
