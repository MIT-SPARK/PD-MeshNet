import os.path as osp
import unittest

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.datasets import Shrec2016DualPrimal

current_dir = osp.dirname(__file__)


class TestDualPrimalDataLoader(unittest.TestCase):

    def test_batch_formation(self):
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
        batch_size = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        dataset_len = len(dataset)
        self.assertEqual(dataset_len, 64)

        for i, (primal_graph, dual_graph, _) in enumerate(data_loader):
            # Each mesh has 750 edges, hence the number of dual nodes/primal
            # edges, considering the directness of the graphs, must be
            # 750 * 2 * batch_size.
            self.assertEqual(primal_graph.num_edges, 750 * 2 * batch_size)
            self.assertEqual(dual_graph.x.shape[0], 750 * 2 * batch_size)
            self.assertEqual(dual_graph.batch.shape[0], 750 * 2 * batch_size)

        num_batches = i + 1
        self.assertEqual(num_batches, 16)
