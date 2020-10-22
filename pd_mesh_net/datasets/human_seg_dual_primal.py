import numpy as np
import os
import os.path as osp
import functools
import glob
import multiprocessing
import pickle as pkl
import pymesh
import torch
import torch_geometric
from torch_geometric.data import extract_zip
from torch_geometric.data.dataset import files_exist
from torch_geometric.io import read_off, read_txt_array

from pd_mesh_net.data import augmentation, post_augmentation
from pd_mesh_net.datasets import BaseDualPrimalDataset
from pd_mesh_net.utils import GraphCreator, preprocess_mesh


class HumanSegDualPrimal(BaseDualPrimalDataset):
    r"""The Human Body Segmentation dataset from the `"Convolutional Neural
    Networks on Surfaces via Seamless Toric Covers"
    <https://dl.acm.org/doi/10.1145/3072959.3073616>`_ paper, in the
    downsampled version provided by `MeshCNN
    <https://github.com/ranahanocka/MeshCNN/>`_ and with edge labels converted
    to face labels using the script in
    `pd_mesh_net.utils.meshcnn_data_converter.py`.
    The data is processed to form dual-primal graphs.

    Args:
        root (string): Root directory where the dataset should be saved.
        single_dual_nodes (bool): If True, the dual graphs will be created with
            single nodes; otherwise, they will be created with double nodes. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`.
        undirected_dual_edges (bool): If True, every directed edge in the dual
            graphs will have an opposite directed edge; otherwise, directed
            edges in the dual graphs will not have an opposite directed edge.
            Cf. :obj:`pd_mesh_net.utils.GraphCreator`.
        primal_features_from_dual_features (bool, optional): Whether or not the
            features of the nodes in the primal graphs should be computed from
            the features of the nodes in the dual graphs. Cf.
            :obj:`pd_mesh_net.utils.GraphCreator`. (default: :obj:`False`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        prevent_nonmanifold_edges (bool, optional): If True, the faces of the
            meshes in the dataset are parsed one at a time and only considered
            if adding them to the set of previous faces does not cause any edges
            to become non-manifold. (default: :obj:`True`)
        num_augmentations (int, optional): Number of data augmentations to
            perform for each sample in the dataset. (default: :obj:`1`)
        vertices_scale_mean, vertices_scale_var (float, optional): If both are
            not None, the vertices from each input mesh are scaled by
            multiplying each of them by a scaling factor drawn from a normal
            distribution with mean `vertices_scale_mean` and variance
            `vertices_scale_var`. (default: :obj:`None`)
        edges_flip_fraction (float, optional): If not None, a fraction equal to
            `edges_flip_fraction` of edges from each input mesh are flipped if
            the dihedral angle between the two faces associated to it is within
            a certain range of values (cf. function :obj:`flip_edges` in
            `pd_mesh_net.data`). (default: :obj:`None`)
        slide_vertices_fraction (float, optional): If not None, a fraction equal
            to `slide_vertices_fraction` of the vertices from each input mesh
            are slid. (cf. function :obj:`slide_vertices` in
            `pd_mesh_net.data`). (default: :obj:`None`)
        return_sample_indices (bool, optional): If True, the get method will
            also return the indices of the samples selected in the dataset.
            (default: :obj:`False`)
        num_workers (int/None, optional): Number of worker processes to use to
            parallelize the mesh conversion job. If this value is larger than
            the number of available CPU cores, the value is set to the number of
            CPU cores. If None is given as argument, the value is set to 75% of
            the number of available CPU cores, rounded to the largest smaller
            integer if necessary. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    
    Attributes:
        None.
    """
    url = 'https://www.dropbox.com/s/byk8oisbm75g5yb/human_seg.zip'

    def __init__(self,
                 root,
                 single_dual_nodes,
                 undirected_dual_edges,
                 primal_features_from_dual_features=False,
                 train=True,
                 prevent_nonmanifold_edges=True,
                 num_augmentations=1,
                 vertices_scale_mean=None,
                 vertices_scale_var=None,
                 edges_flip_fraction=None,
                 slide_vertices_fraction=None,
                 return_sample_indices=False,
                 num_workers=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.__num_augmentations = num_augmentations
        self.__vertices_scale_mean = vertices_scale_mean
        self.__vertices_scale_var = vertices_scale_var
        self.__edges_flip_fraction = edges_flip_fraction
        self.__slide_vertices_fraction = slide_vertices_fraction
        self.__single_dual_nodes = single_dual_nodes
        self.__undirected_dual_edges = undirected_dual_edges
        (self.__primal_features_from_dual_features
        ) = primal_features_from_dual_features
        self.__prevent_nonmanifold_edges = prevent_nonmanifold_edges
        self.__split = 'train' if train else 'test'
        self.__return_sample_indices = return_sample_indices
        self.processed_file_names_train = []
        self.processed_file_names_test = []
        # Store input parameters.
        self.__input_parameters = {
            k: v for k, v in locals().items() if (k[0] != '_' and k != 'self')
        }
        assert (num_workers is None or isinstance(num_workers, int))
        if (num_workers is None):
            self.__num_workers = int(0.75 * os.cpu_count())
        else:
            assert (num_workers >= 1)
            self.__num_workers = min(num_workers, os.cpu_count())

        # Do not insert the parameters 'return_sample_indices' and 'num_workers'
        # in the input parameters, as these are only used for data access/
        # parallelization and do not vary the features of the dataset.
        self.__input_parameters.pop('return_sample_indices')
        self.__input_parameters.pop('num_workers')
        self.__input_parameters['root'] = osp.abspath(root)
        super(HumanSegDualPrimal, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        # Check that if the processed data will not be recomputed (but loaded
        # from disk), the parameters of the processed data stored on disk match
        # the input parameters of the current dataset.
        if (files_exist(self.processed_paths)):
            # Load parameter file of the previously-saved preprocessed data.
            dataset_parameters_filename = osp.join(
                self.processed_dir, f'processed_data_params_{self.__split}.pkl')
            try:
                with open(dataset_parameters_filename, 'rb') as f:
                    previous_dataset_params = pkl.load(f)
            except IOError:
                raise IOError("Unable to open preprocessed-data parameter file "
                              f"'{dataset_parameters_filename}'. Exiting.")
            assert (previous_dataset_params.keys(
            ) == self.__input_parameters.keys()), (
                "The current dataset and the processed one at "
                f"'{self.processed_dir} should have the same list of possible "
                "input parameters, but they do not.")
            if (previous_dataset_params != self.__input_parameters):
                # The two datasets are still compatible if the only difference
                # is in the number of augmentation, if the augmentation
                # parameters match: in this case, as long as the current dataset
                # has a number of augmentations at most equal to that of the
                # previous dataset, it is possible to keep using the previous
                # one, taking only as many augmented versions as specified in
                # the current dataset.
                different_params = set(
                    k for k in previous_dataset_params.keys()
                    if previous_dataset_params[k] != self.__input_parameters[k])
                are_parameters_compatible = False
                if (len(different_params) == 1):
                    if ('num_augmentations' in different_params):
                        are_parameters_compatible = (
                            self.__input_parameters['num_augmentations'] <=
                            previous_dataset_params['num_augmentations'])
                if (not are_parameters_compatible):
                    raise KeyError(
                        "Trying to use preprocessed data at "
                        f"'{self.processed_dir}', but the parameters with "
                        "which these data were generated do not match the "
                        "input parameters of the current dataset. The "
                        f"parameters that differ are {different_params}. "
                        "Either delete the preprocessed data, specify a "
                        "different root folder, or change the input parameters "
                        "of the current dataset.")

    @property
    def input_parameters(self):
        return self.__input_parameters

    @property
    def raw_file_names(self):
        # We assume that if the 'raw' folder contains one subfolder named
        # 'human_seg', having itself subfolders 'train', 'test' and 'seg', there
        # is no need to download the dataset again.
        return [
            osp.join(f"human_seg", "train"),
            osp.join(f"human_seg", "test"),
            osp.join(f"human_seg", "seg")
        ]

    def __len__(self):
        if (self.__split == 'train'):
            len_processed_file_names = len(self.processed_file_names_train)
        else:
            len_processed_file_names = len(self.processed_file_names_test)
        assert (len_processed_file_names % 3 == 0)
        return len_processed_file_names // 3

    @property
    def processed_file_names(self):
        # List of files that the 'processed' folder should contain to avoid
        # reprocessing the data.
        mesh_names = []
        filenames = []

        should_obtain_processed_files_list = False
        if (self.__split == 'train' and
                len(self.processed_file_names_train) == 0):
            assert (len(self.processed_file_names_test) == 0)
            should_obtain_processed_files_list = True
        elif (self.__split == 'test' and
              len(self.processed_file_names_test) == 0):
            assert (len(self.processed_file_names_train) == 0)
            should_obtain_processed_files_list = True

        if (should_obtain_processed_files_list):
            assert (len(self.processed_file_names_test) == 0)
            paths = sorted(
                glob.glob(
                    osp.join(self.raw_dir, f'human_seg',
                             f'{self.__split}/*.ply')))

            new_filenames = []
            for path in paths:
                mesh_name = path.rpartition('/')[2].split('.')[0]
                mesh_names.append(mesh_name)
                for augmentation_idx in range(self.__num_augmentations):
                    # Primal-graph data.
                    new_filenames.append(
                        osp.join(
                            self.__split,
                            f"{mesh_name}_aug_{augmentation_idx}_primal.pt"))
                    # Dual-graph data.
                    new_filenames.append(
                        osp.join(self.__split,
                                 f"{mesh_name}_aug_{augmentation_idx}_dual.pt"))
                    # Primal-edge-to-dual-node-index data.
                    new_filenames.append(
                        osp.join(
                            self.__split,
                            f"{mesh_name}_aug_{augmentation_idx}_petdni.pkl"))

            if (self.__split == 'train'):
                self.processed_file_names_train = new_filenames
            else:
                self.processed_file_names_test = new_filenames

        if (self.__split == 'train'):
            filenames = self.processed_file_names_train
        else:
            filenames = self.processed_file_names_test

        return filenames

    def download(self):
        # Download and extract the archive.
        filename = self.url.rpartition('/')[2]
        path = osp.join(self.raw_dir, filename)
        if osp.exists(path):  # pragma: no cover
            print(f"Using existing file {filename}")
        else:
            # - We need to use the system 'wget' because urllib does not work
            #   with Dropbox.
            print(f'Downloading {self.url}')
            os.system(f"wget {self.url} -O {path} -q")
        # - Extract the downloaded archive.
        extract_zip(path, self.raw_dir)
        # Delete the archive.
        os.unlink(path)

    def _process_single_mesh(self, path, process_subfolder):
        r"""Given an input path associated to a single mesh, processes the
        latter by converting it to primal-dual graphs.
        Args:
            path (str): Path to the input mesh.
            process_subfolder (str): Folder where the output processed mesh
                should be saved.
        Returns:
            None.
        """
        # Mesh name without extension.
        mesh_name = path.rpartition('/')[2].split('.')[0]
        # Load mesh.
        mesh = pymesh.load_mesh(path)
        # Load per-face class label.
        face_label_file = osp.join(self.raw_dir, f'human_seg', 'seg',
                                   f"{mesh_name}.eseg")
        with open(face_label_file, 'r') as f:
            y = np.loadtxt(f, dtype='long')
        # Preprocess mesh.
        mesh = preprocess_mesh(
            input_mesh=mesh,
            prevent_nonmanifold_edges=self.__prevent_nonmanifold_edges)
        # Perform data augmentation and post-augmentation.
        for augmentation_idx in range(self.__num_augmentations):
            augmented_mesh = augmentation(
                mesh=mesh,
                vertices_scale_mean=self.__vertices_scale_mean,
                vertices_scale_var=self.__vertices_scale_var,
                edges_flip_fraction=self.__edges_flip_fraction)
            postaugmented_mesh = post_augmentation(
                mesh=augmented_mesh,
                slide_vertices_fraction=self.__slide_vertices_fraction)
            # Convert the mesh to dual-primal graphs.
            graph_creator = GraphCreator(
                mesh=postaugmented_mesh,
                single_dual_nodes=self.__single_dual_nodes,
                undirected_dual_edges=self.__undirected_dual_edges,
                primal_features_from_dual_features=self.
                __primal_features_from_dual_features,
                prevent_nonmanifold_edges=self.__prevent_nonmanifold_edges)
            primal_graph, dual_graph = graph_creator.create_graphs()
            (primal_edge_to_dual_node_idx
            ) = graph_creator.primal_edge_to_dual_node_idx
            # Add the ground-truth per-face class label to the primal
            # graph.
            assert (len(y) == primal_graph.num_nodes)
            primal_graph.y = torch.from_numpy(y).long()
            # Save the graphs and the dictionary.
            torch.save(
                primal_graph,
                osp.join(process_subfolder,
                         f"{mesh_name}_aug_{augmentation_idx}_primal.pt"))
            torch.save(
                dual_graph,
                osp.join(process_subfolder,
                         f"{mesh_name}_aug_{augmentation_idx}_dual.pt"))
            petdni_filename = osp.join(
                process_subfolder,
                f"{mesh_name}_aug_{augmentation_idx}_petdni.pkl")
            pymesh.save_mesh(
                osp.join(process_subfolder,
                         f"{mesh_name}_aug_{augmentation_idx}.obj"),
                postaugmented_mesh)
            try:
                with open(petdni_filename, 'wb') as f:
                    pkl.dump(primal_edge_to_dual_node_idx, f)
            except IOError:
                raise IOError("Error while writing file "
                              f"'{petdni_filename}'. Exiting.")

    def process(self):
        # Verify that no previous data is being inadvertently erased.
        processed_data_folder = osp.join(self.processed_dir, self.__split)
        if (osp.exists(processed_data_folder)):
            if (os.listdir(processed_data_folder)):
                raise IOError(
                    "The folder containing the processed data, "
                    f"'{processed_data_folder}', is not empty. Most likely the "
                    "root folder you have set for the current dataset "
                    "coincides with that of a previously-generated dataset, "
                    "and the current dataset has parameters not fully "
                    "compatible with those used to generate the data already "
                    "in the folder. Please double-check the dataset parameters "
                    "or delete the content of the folder/specify a different "
                    "root folder of the dataset.")

        process_subfolder = processed_data_folder
        if (not osp.exists(process_subfolder)):
            os.makedirs(process_subfolder)

        paths = glob.glob(
            osp.join(self.raw_dir, 'human_seg', self.__split, '*.ply'))

        # Parallelize the mesh processing.
        pool = multiprocessing.Pool(processes=self.__num_workers)
        pool.map(
            functools.partial(self._process_single_mesh,
                              process_subfolder=process_subfolder), paths)
        pool.close()
        pool.join()

        if (self.pre_filter is not None):
            raise NotImplementedError

        if (self.pre_transform is not None):
            raise NotImplementedError

        # Save the input parameters of the dataset, so that when using it
        # without repreprocessing the data, one can make sure that the input
        # parameters match those with which the preprocessed data saved to disk
        # was generated.
        dataset_parameters_filename = osp.join(
            self.processed_dir, f'processed_data_params_{self.__split}.pkl')
        try:
            with open(dataset_parameters_filename, 'wb') as f:
                pkl.dump(self.input_parameters, f)

        except IOError:
            raise IOError("Error while writing file dataset parameter file "
                          f"'{dataset_parameters_filename}'. Exiting.")

    def get(self, idx):
        if (isinstance(idx, int)):
            if (self.__split == 'train'):
                primal_graph_filename = self.processed_file_names_train[3 * idx]
                dual_graph_filename = self.processed_file_names_train[3 * idx +
                                                                      1]
                petdni_filename = self.processed_file_names_train[3 * idx + 2]
            else:
                primal_graph_filename = self.processed_file_names_test[3 * idx]
                dual_graph_filename = self.processed_file_names_test[3 * idx +
                                                                     1]
                petdni_filename = self.processed_file_names_test[3 * idx + 2]

            primal_graph_filename = osp.join(self.processed_dir,
                                             primal_graph_filename)
            dual_graph_filename = osp.join(self.processed_dir,
                                           dual_graph_filename)
            petdni_filename = osp.join(self.processed_dir, petdni_filename)

            primal_graph_data = torch.load(primal_graph_filename)
            dual_graph_data = torch.load(dual_graph_filename)
            try:
                with open(petdni_filename, 'rb') as f:
                    petdni_data = pkl.load(f)
            except IOError:
                raise IOError(
                    f"Unable to open file '{petdni_filename}'. Exiting.")
            if (self.__return_sample_indices):
                return primal_graph_data, dual_graph_data, petdni_data, idx
            else:
                return primal_graph_data, dual_graph_data, petdni_data, None
        elif isinstance(idx, slice):
            # Obtains the indices in the dataset from the input slice.
            indices = [*range(*idx.indices(len(self)))]
            return self.__indexing__(indices)

    def __indexing__(self, indices):
        copy = self.__class__.__new__(self.__class__)
        copy.__dict__ = self.__dict__.copy()

        copy.processed_file_names_train = []
        copy.processed_file_names_test = []

        if (self.__split == 'train'):
            for idx in indices:
                copy.processed_file_names_train.append(
                    self.processed_file_names_train[3 * idx])
                copy.processed_file_names_train.append(
                    self.processed_file_names_train[3 * idx + 1])
                copy.processed_file_names_train.append(
                    self.processed_file_names_train[3 * idx + 2])
        else:
            for idx in indices:
                copy.processed_file_names_test.append(
                    self.processed_file_names_test[3 * idx])
                copy.processed_file_names_test.append(
                    self.processed_file_names_test[3 * idx + 1])
                copy.processed_file_names_test.append(
                    self.processed_file_names_test[3 * idx + 2])

        return copy

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))
