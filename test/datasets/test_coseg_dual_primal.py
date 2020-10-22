import numpy as np
import os.path as osp
import sys
import torch
import unittest

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.datasets import CosegDualPrimal
current_dir = osp.dirname(__file__)


class TestCosegDualPrimal(unittest.TestCase):

    def test_batch_formation(self):
        dataset = CosegDualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/coseg_config_A/')),
                                  categories=['aliens'],
                                  single_dual_nodes=True,
                                  undirected_dual_edges=True,
                                  return_sample_indices=True)
        segmentation_data_root = osp.join(
            current_dir, '../common_data/coseg_config_A/raw/coseg_aliens/seg')
        batch_size = 13  # 169 = 13 ** 2

        self.assertEqual(len(dataset), 169)
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           return_sample_indices=True)
        for (primal_graph_batch, dual_graph_batch, _,
             sample_indices) in data_loader:
            self.assertEqual(primal_graph_batch.num_graphs, batch_size)
            self.assertEqual(dual_graph_batch.num_graphs, batch_size)
            # Check that the class labels associated to each of the primal nodes
            # in the batch matches the one in the segmentation data file
            # associated to that sample.
            for sample_index_in_batch, sample_index in enumerate(
                    sample_indices):
                # - Load the ground-truth labels.
                base_filename = dataset.processed_file_names_train[
                    3 * sample_index].rpartition('/')[-1].split('_')[0]
                gt_label_file = osp.join(segmentation_data_root,
                                         f'{base_filename}.eseg')
                with open(gt_label_file, 'r') as f:
                    gt_labels = np.loadtxt(f, dtype='float64')
                # - Find the class labels of the nodes in the batch that belong
                #   to the current sample.
                indices_nodes_in_sample = (
                    primal_graph_batch.batch == sample_index_in_batch
                ).nonzero().view(-1)
                class_labels_nodes_in_sample = primal_graph_batch.y[
                    indices_nodes_in_sample].numpy()
                self.class_labels_nodes_in_sample = class_labels_nodes_in_sample
                self.gt_labels = gt_labels
                # - Verify that the class labels match the ground-truth ones.
                self.assertTrue(
                    np.array_equal(class_labels_nodes_in_sample, gt_labels))
