import os.path as osp
import unittest

from pd_mesh_net.datasets import Shrec2016DualPrimal
from pd_mesh_net.utils import (compute_mean_and_std, create_dual_primal_batch,
                               create_dataset)

current_dir = osp.dirname(__file__)


class TestShrec2016DualPrimal(unittest.TestCase):

    __primal_mean = None
    __primal_std = None
    __dual_mean = None
    __dual_std = None

    def _test_compute_mean_and_std(self):
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
        dataset_params = dataset.input_parameters
        primal_mean, primal_std, dual_mean, dual_std = compute_mean_and_std(
            dataset=dataset,
            dataset_params=dataset_params,
            filename=osp.join(current_dir, '../output_data/dataset_params.pkl'))

        # Store statistics, so that they can be compared with those computed for
        # other tests using the same dataset input parameters.
        self.__class__.__primal_mean = primal_mean
        self.__class__.__primal_std = primal_std
        self.__class__.__dual_mean = dual_mean
        self.__class__.__dual_std = dual_std

    def test_feature_standardization(self):
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
        dataset_params = dataset.input_parameters
        primal_mean, primal_std, dual_mean, dual_std = compute_mean_and_std(
            dataset=dataset,
            dataset_params=dataset_params,
            filename=osp.join(current_dir, '../output_data/dataset_params.pkl'))
        # Create batch of the size of the entire dataset, with standardized
        # features. The mean of the primal-graph-/dual-graph- node features
        # should now be 0, while the standard deviation should be 1.
        primal_graph_list = []
        dual_graph_list = []
        primal_edge_to_dual_node_idx_list = []
        for (primal_graph, dual_graph, primal_edge_to_dual_node_idx,
             _) in dataset:
            primal_graph_list.append(primal_graph)
            dual_graph_list.append(dual_graph)
            primal_edge_to_dual_node_idx_list.append(
                primal_edge_to_dual_node_idx)
        (primal_graph_batch, dual_graph_batch,
         _) = create_dual_primal_batch(primal_graph_list,
                                       dual_graph_list,
                                       primal_edge_to_dual_node_idx_list,
                                       primal_mean=primal_mean,
                                       primal_std=primal_std,
                                       dual_mean=dual_mean,
                                       dual_std=dual_std)

        self.assertAlmostEqual(primal_graph_batch.x.mean(axis=0).item(), 0., 5)
        self.assertAlmostEqual(primal_graph_batch.x.std(axis=0).item(), 1., 5)

        for dual_features_idx in range(4):
            self.assertAlmostEqual(
                dual_graph_batch.x.mean(axis=0)[dual_features_idx].item(), 0.,
                4)
            self.assertAlmostEqual(
                dual_graph_batch.x.std(axis=0)[dual_features_idx].item(), 1., 4)

    def _test_create_dataset(self):
        dataset_input_params = {
            'root':
                osp.abspath(osp.join(current_dir, '../common_data/shrec2016/')),
            'categories': ['shark'],
            'single_dual_nodes':
                False,
            'undirected_dual_edges':
                True,
            'vertices_scale_mean':
                1.,
            'vertices_scale_var':
                0.1,
            'edges_flip_fraction':
                0.5,
            'slide_vertices_fraction':
                0.2,
            'num_augmentations':
                4
        }
        dataset, (primal_mean, primal_std, dual_mean,
                  dual_std) = create_dataset(
                      dataset_name='shrec_16',
                      compute_node_feature_stats=True,
                      node_feature_stats_filename=osp.join(
                          current_dir, '../output_data/dataset_params.pkl'),
                      **dataset_input_params)

        self.assertTrue(isinstance(dataset, Shrec2016DualPrimal))
        # As a further check, verify that mean and standard deviation of the
        # node features match with the values computed in the previous tests.
        self.assertAlmostEqual(primal_mean.item(),
                               self.__class__.__primal_mean.item())
        self.assertAlmostEqual(primal_std.item(),
                               self.__class__.__primal_std.item())
        self.assertAlmostEqual(dual_mean[0], self.__class__.__dual_mean[0])
        self.assertAlmostEqual(dual_mean[1], self.__class__.__dual_mean[1])
        self.assertAlmostEqual(dual_mean[2], self.__class__.__dual_mean[2])
        self.assertAlmostEqual(dual_mean[3], self.__class__.__dual_mean[3])
        self.assertAlmostEqual(dual_std[0], self.__class__.__dual_std[0])
        self.assertAlmostEqual(dual_std[1], self.__class__.__dual_std[1])
        self.assertAlmostEqual(dual_std[2], self.__class__.__dual_std[2])
        self.assertAlmostEqual(dual_std[3], self.__class__.__dual_std[3])

        # Check that a name not associated to a dataset raises an exception.
        self.assertRaises(KeyError,
                          create_dataset,
                          dataset_name='shrec_12',
                          **dataset_input_params)

    def test_sequential_tests(self):
        # Force the test '_test_compute_mean_and_std' to be run before test
        # '_test_create_dataset', as the latter checks the mean-std parameters
        # against those computed in the former.
        self._test_compute_mean_and_std()
        self._test_create_dataset()