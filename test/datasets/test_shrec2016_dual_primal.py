import numpy as np
import os.path as osp
import shutil
import sys
import torch
import unittest

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.datasets import Shrec2016DualPrimal

current_dir = osp.dirname(__file__)


class TestShrec2016DualPrimal(unittest.TestCase):

    def test_download_process_and_get(self):
        dataset = Shrec2016DualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016/')),
                                      categories=['shark'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=4)
        # Get one element of the dataset.
        print(dataset[0])

    def test_right_graph_connectivity(self):
        # The shape in SHREC are closed, manifold meshes. Therefore the
        # associated 'primal graph' (simplex mesh) and 'dual graph' (medial
        # graph) should be respectively 3-regular and 4-regular (when directness
        # is not considered).
        root_shrec = osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016/'))
        processed_shrec_fold = osp.join(root_shrec, 'processed/')
        if (osp.exists(processed_shrec_fold)):
            sys.stdout.write(
                "\nWarning: running the following test will cause the folder "
                f"'{processed_shrec_fold}' to be deleted! ")
            valid_choice = False
            while (not valid_choice):
                sys.stdout.write("Do you want to continue? [y/n] ")
                user_input = input().lower()
                if (user_input == 'y'):
                    print("Removing folder...")
                    shutil.rmtree(processed_shrec_fold)
                    valid_choice = True
                elif (user_input == 'n'):
                    print("Skipping test.")
                    valid_choice = True
                    return
                else:
                    sys.stdout.write(
                        "Please respond with 'y'/'Y' or 'n'/'N'.\n")

        print("Running test...")
        dataset = Shrec2016DualPrimal(root=root_shrec,
                                      categories=['shark'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=4)

        for primal_graph, dual_graph, _, _ in dataset:
            neighbors_incoming_edges_primal = dict()
            neighbors_outgoing_edges_primal = dict()
            for a, b in primal_graph.edge_index.t():
                if (not a.item() in neighbors_incoming_edges_primal):
                    neighbors_incoming_edges_primal[a.item()] = [b.item()]
                else:
                    neighbors_incoming_edges_primal[a.item()].append(b.item())
                if (not b.item() in neighbors_outgoing_edges_primal):
                    neighbors_outgoing_edges_primal[b.item()] = [a.item()]
                else:
                    neighbors_outgoing_edges_primal[b.item()].append(a.item())

            self.assertEqual(
                len([
                    n for n in neighbors_incoming_edges_primal.keys()
                    if len(neighbors_incoming_edges_primal[n]) != 3
                ]), 0)
            self.assertEqual(
                len([
                    n for n in neighbors_outgoing_edges_primal.keys()
                    if len(neighbors_outgoing_edges_primal[n]) != 3
                ]), 0)

            neighbors_incoming_edges_dual = dict()
            neighbors_outgoing_edges_dual = dict()
            for a, b in dual_graph.edge_index.t():
                if (not a.item() in neighbors_incoming_edges_dual):
                    neighbors_incoming_edges_dual[a.item()] = [b.item()]
                else:
                    neighbors_incoming_edges_dual[a.item()].append(b.item())
                if (not b.item() in neighbors_outgoing_edges_dual):
                    neighbors_outgoing_edges_dual[b.item()] = [a.item()]
                else:
                    neighbors_outgoing_edges_dual[b.item()].append(a.item())

            self.assertEqual(
                len([
                    n for n in neighbors_incoming_edges_dual.keys()
                    if len(neighbors_incoming_edges_dual[n]) != 4
                ]), 0)
            self.assertEqual(
                len([
                    n for n in neighbors_outgoing_edges_dual.keys()
                    if len(neighbors_outgoing_edges_dual[n]) != 4
                ]), 0)

    def test_shrec_multiple_classes(self):
        # The shape in SHREC are closed, manifold meshes. Therefore the
        # associated 'primal graph' (simplex mesh) and 'dual graph' (medial
        # graph) should be respectively 3-regular and 4-regular (when directness
        # is not considered).
        root_shrec = osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016_shark_gorilla/'))
        dataset = Shrec2016DualPrimal(root=root_shrec,
                                      train=True,
                                      categories=['shark', 'gorilla'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=2)
        batch_size = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        dataset_size = len(dataset)
        # There are 16 'gorilla' shapes and 16 'shark' shapes in the 'train'
        # split of SHREC2016. Counting 2 versions per shape due to data
        # augmentation, one has (16 + 16) * 2 = 64 shapes in total.
        self.assertEqual(dataset_size, 64)
        # Check the ground-truth class index of the samples in the batch.
        shark_class_index = Shrec2016DualPrimal.valid_categories.index('shark')
        gorilla_class_index = Shrec2016DualPrimal.valid_categories.index(
            'gorilla')
        for (primal_graph_batch, _, _) in data_loader:
            self.assertEqual(primal_graph_batch.y.size(), (batch_size,))
            for primal_graph_idx in range(batch_size):
                sample_class_index = primal_graph_batch.y[
                    primal_graph_idx].item()
                self.assertTrue(sample_class_index in
                                [shark_class_index, gorilla_class_index])

    def test_slicing(self):
        dataset = Shrec2016DualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016/')),
                                      categories=['shark'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=4)

        reduced_dataset = dataset[:10]

        for idx in range(10):
            for graph_idx in range(2):
                # Verify that all attributes of the samples from the original
                # dataset and the "sliced" version match.
                self.assertEqual(
                    dataset[idx][graph_idx].contains_isolated_nodes(),
                    reduced_dataset[idx][graph_idx].contains_isolated_nodes())
                self.assertEqual(
                    dataset[idx][graph_idx].contains_self_loops(),
                    reduced_dataset[idx][graph_idx].contains_self_loops())
                self.assertEqual(dataset[idx][graph_idx].is_coalesced(),
                                 reduced_dataset[idx][graph_idx].is_coalesced())
                self.assertEqual(dataset[idx][graph_idx].is_directed(),
                                 reduced_dataset[idx][graph_idx].is_directed())
                self.assertEqual(
                    dataset[idx][graph_idx].is_undirected(),
                    reduced_dataset[idx][graph_idx].is_undirected())
                for scalar_attr in [
                        'keys', 'norm', 'num_edge_features', 'num_edges',
                        'num_faces', 'num_node_features', 'num_nodes'
                ]:
                    if (getattr(dataset[idx][graph_idx], scalar_attr) is None):
                        self.assertTrue(
                            getattr(reduced_dataset[idx][graph_idx],
                                    scalar_attr) is None)
                    else:
                        self.assertEqual(
                            getattr(dataset[idx][graph_idx], scalar_attr),
                            getattr(reduced_dataset[idx][graph_idx],
                                    scalar_attr))

                for tensor_attr in [
                        'edge_attr', 'edge_index', 'face', 'pos', 'x', 'y'
                ]:
                    if (getattr(dataset[idx][graph_idx], tensor_attr) is None):
                        self.assertTrue(
                            getattr(reduced_dataset[idx][graph_idx],
                                    tensor_attr) is None)
                    else:
                        self.assertTrue(
                            np.all(
                                getattr(dataset[idx][graph_idx],
                                        tensor_attr).numpy() == getattr(
                                            reduced_dataset[idx][graph_idx],
                                            tensor_attr).numpy()))
            # Check primal-edge-to-dual-node index dictionary.
            self.assertEqual(dataset[idx][2], reduced_dataset[idx][2])

        reduced_dataset = dataset[2:60:5]

        for reduced_dataset_idx, dataset_idx in enumerate(range(2, 60, 5)):
            for graph_idx in range(2):
                # Verify that all attributes of the samples from the original
                # dataset and the "sliced" version match.
                self.assertEqual(
                    dataset[dataset_idx][graph_idx].contains_isolated_nodes(),
                    reduced_dataset[reduced_dataset_idx]
                    [graph_idx].contains_isolated_nodes())
                self.assertEqual(
                    dataset[dataset_idx][graph_idx].contains_self_loops(),
                    reduced_dataset[reduced_dataset_idx]
                    [graph_idx].contains_self_loops())
                self.assertEqual(
                    dataset[dataset_idx][graph_idx].is_coalesced(),
                    reduced_dataset[reduced_dataset_idx]
                    [graph_idx].is_coalesced())
                self.assertEqual(
                    dataset[dataset_idx][graph_idx].is_directed(),
                    reduced_dataset[reduced_dataset_idx]
                    [graph_idx].is_directed())
                self.assertEqual(
                    dataset[dataset_idx][graph_idx].is_undirected(),
                    reduced_dataset[reduced_dataset_idx]
                    [graph_idx].is_undirected())
                for scalar_attr in [
                        'keys', 'norm', 'num_edge_features', 'num_edges',
                        'num_faces', 'num_node_features', 'num_nodes'
                ]:
                    if (getattr(dataset[dataset_idx][graph_idx], scalar_attr) is
                            None):
                        self.assertTrue(
                            getattr(
                                reduced_dataset[reduced_dataset_idx][graph_idx],
                                scalar_attr) is None)
                    else:
                        self.assertEqual(
                            getattr(dataset[dataset_idx][graph_idx],
                                    scalar_attr),
                            getattr(
                                reduced_dataset[reduced_dataset_idx][graph_idx],
                                scalar_attr))

                for tensor_attr in [
                        'edge_attr', 'edge_index', 'face', 'pos', 'x', 'y'
                ]:
                    if (getattr(dataset[dataset_idx][graph_idx], tensor_attr) is
                            None):
                        self.assertTrue(
                            getattr(
                                reduced_dataset[reduced_dataset_idx][graph_idx],
                                tensor_attr) is None)
                    else:
                        self.assertTrue(
                            np.all(
                                getattr(dataset[dataset_idx][graph_idx],
                                        tensor_attr).numpy() == getattr(
                                            reduced_dataset[reduced_dataset_idx]
                                            [graph_idx], tensor_attr).numpy()))
            # Check primal-edge-to-dual-node index dictionary.
            self.assertEqual(dataset[dataset_idx][2],
                             reduced_dataset[reduced_dataset_idx][2])

    def test_validation_split(self):
        dataset = Shrec2016DualPrimal(root=osp.join(
            current_dir, '../../datasets_no_augmentation/'),
                                      categories=[],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      num_augmentations=1)
        validation_set_fraction = 0.1
        num_samples_train = int(len(dataset) * (1. - validation_set_fraction))

        dataset_train = dataset[:num_samples_train]
        dataset_validation = dataset[num_samples_train:]

        print("\n")
        for category_idx in range(30):
            print(f"* Category {category_idx}:")
            elements_in_training_set = [
                idx for idx, i in enumerate(dataset_train)
                if i[0].y.item() == category_idx
            ]
            elements_in_validation_set = [
                idx for idx, i in enumerate(dataset_validation)
                if i[0].y.item() == category_idx
            ]
            self.assertGreater(len(elements_in_training_set), 0)
            self.assertGreater(len(elements_in_validation_set), 0)
            print("\t- Num elements in training set: "
                  f"{len(elements_in_training_set)}")
            print("\t- Num elements in validation set: "
                  f"{len(elements_in_validation_set)}")

    def test_sample_indices_no_shuffle(self):
        dataset = Shrec2016DualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016/')),
                                      categories=['shark'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=4,
                                      return_sample_indices=True)

        self.assertEqual(len(dataset), 64)
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=8,
                                           shuffle=False,
                                           return_sample_indices=True)
        for batch_idx, (_, _, _, sample_indices) in enumerate(data_loader):
            self.assertEqual(sample_indices,
                             [*range(batch_idx * 8, batch_idx * 8 + 8)])

    def test_sample_indices_shuffle(self):
        dataset = Shrec2016DualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/shrec2016/')),
                                      categories=['shark'],
                                      single_dual_nodes=False,
                                      undirected_dual_edges=True,
                                      vertices_scale_mean=1.,
                                      vertices_scale_var=0.1,
                                      edges_flip_fraction=0.5,
                                      slide_vertices_fraction=0.2,
                                      num_augmentations=4,
                                      return_sample_indices=True)

        self.assertEqual(len(dataset), 64)
        sample_indices_found = set()
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           return_sample_indices=True)
        for _, _, _, sample_indices in data_loader:
            for sample_idx in sample_indices:
                sample_indices_found.add(sample_idx)

        self.assertEqual(len(sample_indices_found), len(dataset))
        self.assertEqual(list(sample_indices_found), [*range(0, 64)])
