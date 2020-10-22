import numpy as np
import os.path as osp
import sys
import torch
import unittest

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.datasets import HumanSegDualPrimal
current_dir = osp.dirname(__file__)


class TestHumanSegDualPrimal(unittest.TestCase):

    def test_batch_formation(self):
        dataset = HumanSegDualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/human_seg/')),
                                     single_dual_nodes=True,
                                     undirected_dual_edges=True,
                                     return_sample_indices=True)
        segmentation_data_root = osp.join(
            current_dir, '../common_data/human_seg/raw/human_seg/seg')
        batch_size = 3  # 381 = 3 * 127

        self.assertEqual(len(dataset), 381)
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
                    3 * sample_index].rpartition('/')[-1].split('_aug')[0]
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
