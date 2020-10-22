import torch
import unittest

from pd_mesh_net.utils import TensorClusters


class TestTensorClusters(unittest.TestCase):

    def test_clusters(self):
        # Example 'edge-index' tensor.
        # The first two edges ({1, 5} and {6, 7}) do not share any endpoint with
        # the other edges and hence each make a cluster by themselves. All the
        # other edges, instead, share an edge with another edge, and their
        # endpoints are therefore all merged in a single cluster.
        edges_to_pool = torch.LongTensor([[1, 6, 8, 9, 9, 10, 11, 13],
                                          [5, 7, 9, 10, 13, 11, 12, 14]])

        node_clusterizer = TensorClusters(store_tensor_indices_per_cluster=True)

        for edge in edges_to_pool.t():
            node_clusterizer.add(edge)

        node_clusters, edge_idx_clusters = node_clusterizer.clusters

        self.assertEqual(len(node_clusters), 3)
        self.assertEqual(len(edge_idx_clusters), 3)

        # Transform the tensors to lists.
        for cluster_idx in range(3):
            node_cluster = node_clusters[cluster_idx].tolist()
            edge_idx_cluster = edge_idx_clusters[cluster_idx].tolist()

            if (1 in node_cluster):
                self.assertEqual(len(node_cluster), 2)
                self.assertTrue(5 in node_cluster)
                self.assertEqual(len(edge_idx_cluster), 1)
                self.assertTrue(0 in edge_idx_cluster)
            elif (6 in node_cluster):
                self.assertEqual(len(node_cluster), 2)
                self.assertTrue(7 in node_cluster)
                self.assertEqual(len(edge_idx_cluster), 1)
                self.assertTrue(1 in edge_idx_cluster)
            else:
                # There are 7 nodes in this cluster: 8, 9, 10, 11, 12, 13, 14.
                self.assertEqual(len(node_cluster), 7)
                for node_idx in range(8, 15):
                    self.assertTrue(node_idx in node_cluster)
                # 6 edges were merged in the cluster.
                self.assertEqual(len(edge_idx_cluster), 6)
                for edge_idx in range(2, 8):
                    self.assertTrue(edge_idx in edge_idx_cluster)
