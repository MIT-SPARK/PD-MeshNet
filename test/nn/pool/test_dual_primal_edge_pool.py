import numpy as np
import os.path as osp
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
import unittest

from pd_mesh_net.nn import DualPrimalEdgePooling
from pd_mesh_net.utils import create_graphs, create_dual_primal_batch

current_dir = osp.dirname(__file__)


class TestDualEdgePooling(unittest.TestCase):

    def test_large_simple_mesh_config_A_no_output_self_loops(self):
        # In all cases, we aim at pooling the following pairs of primal edges,
        # out of the 21 in the mesh:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.

        # All the three experiments are repeated by considering once pooling
        # based on decreasing attention coefficients and in the other pooling
        # based on increasing attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                # Test with number of primal edges to keep.
                self.__test_large_simple_mesh_config_A_no_output_self_loops(
                    num_primal_edges_to_keep=21 - 8,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with fraction of primal edges to keep. Pooling the top-8
                # out of the 21 primal-edge pairs corresponds to keeping a
                # fraction of the primal edges around (21 - 8) / 21 = 0.6190...
                # Since the pooling layer internally finds the number of primal
                # edges to pool as
                # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one needs
                # to have:
                #     8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                # <=> 0.5714... < fraction_primal_edges_to_keep <= 0.6190...;
                # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                self.__test_large_simple_mesh_config_A_no_output_self_loops(
                    fraction_primal_edges_to_keep=0.619,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with minimal attention coefficient.
                self.__test_large_simple_mesh_config_A_no_output_self_loops(
                    primal_att_coeff_threshold=0.5,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_A_no_output_self_loops_nonconsecutive(
        self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf. `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                self.__test_config_A_no_output_self_loops_nonconsecutive(
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_A_with_output_self_loops_nonconsecutive(
        self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                self.__test_config_A_with_output_self_loops_nonconsecutive(
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_A_with_output_self_loops(self):
        # In all cases, we aim at pooling the following pairs of primal edges,
        # out of the 21 in the mesh:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.

        # All the three experiments are repeated by considering once pooling
        # based on decreasing attention coefficients and in the other pooling
        # based on increasing attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                # Test with number of primal edges to keep.
                self.__test_large_simple_mesh_config_A_with_output_self_loops(
                    num_primal_edges_to_keep=21 - 8,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with fraction of primal edges to keep. Pooling the top-8
                # out of the 21 primal-edge pairs corresponds to keeping a
                # fraction of the primal edges around (21 - 8) / 21 = 0.6190...
                # Since the pooling layer internally finds the number of primal
                # edges to pool as
                # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one needs
                # to have:
                #     8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                # <=> 0.5714... < fraction_primal_edges_to_keep <= 0.6190...;
                # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                self.__test_large_simple_mesh_config_A_with_output_self_loops(
                    fraction_primal_edges_to_keep=0.619,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with minimal attention coefficient.
                self.__test_large_simple_mesh_config_A_with_output_self_loops(
                    primal_att_coeff_threshold=0.5,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_B_with_output_self_loops(self):
        # In all cases, we aim at pooling the following pairs of primal edges,
        # out of the 21 in the mesh:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.

        # All the three experiments are repeated by considering once pooling
        # based on decreasing attention coefficients and in the other pooling
        # based on increasing attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                # Test with number of primal edges to keep.
                self.__test_large_simple_mesh_config_B_with_output_self_loops(
                    num_primal_edges_to_keep=21 - 8,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with fraction of primal edges to keep. Pooling the top-8
                # out of the 21 primal-edge pairs corresponds to keeping a
                # fraction of the primal edges around (21 - 8) / 21 = 0.6190...
                # Since the pooling layer internally finds the number of primal
                # edges to pool as
                # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one needs
                # to have:
                #      8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                # <=> 0.5714... < fraction_primal_edges_to_keep <= 0.6190...;
                # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                self.__test_large_simple_mesh_config_B_with_output_self_loops(
                    fraction_primal_edges_to_keep=0.619,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with minimal attention coefficient.
                self.__test_large_simple_mesh_config_B_with_output_self_loops(
                    primal_att_coeff_threshold=0.5,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_B_with_output_self_loops_nonconsecutive(
        self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                self.__test_config_B_with_output_self_loops_nonconsecutive(
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_C_with_output_self_loops(self):
        # In all cases, we aim at pooling the following pairs of primal edges,
        # out of the 21 in the mesh:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.
        # All the three experiments are repeated by considering once pooling
        # based on decreasing attention coefficients and in the other pooling
        # based on increasing attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                # Test with number of primal edges to keep.
                self.__test_large_simple_mesh_config_C_with_output_self_loops(
                    num_primal_edges_to_keep=21 - 8,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with fraction of primal edges to keep. Pooling the top-8
                # out of the 21 primal-edge pairs corresponds to keeping a
                # fraction of the primal edges around (21 - 8) / 21 = 0.6190...
                # Since the pooling layer internally finds the number of primal
                # edges to pool as
                # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one needs
                # to have:
                #     8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                # <=> 0.5714... < fraction_primal_edges_to_keep <= 0.6190...;
                # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                self.__test_large_simple_mesh_config_C_with_output_self_loops(
                    fraction_primal_edges_to_keep=0.619,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)
                # Test with minimal attention coefficient.
                self.__test_large_simple_mesh_config_C_with_output_self_loops(
                    primal_att_coeff_threshold=0.5,
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def test_large_simple_mesh_config_C_with_output_self_loops_nonconsecutive(
        self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test also with more than one attention head.
            for num_heads in range(1, 4):
                self.__test_config_C_with_output_self_loops_nonconsecutive(
                    use_decreasing_attention_coefficient=
                    use_decreasing_attention_coefficient,
                    num_heads=num_heads)

    def __test_large_simple_mesh_config_A_no_output_self_loops(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges // 2)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so as to pool the following
        # primal edges:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.
        # (cf. file `../../common_data/simple_mesh_large_pool_1.png`)
        if (primal_att_coeff_threshold is not None):
            attention_threshold = primal_att_coeff_threshold
        else:
            attention_threshold = 0.5
        primal_attention_coeffs = torch.rand(
            [num_primal_edges, num_heads],
            dtype=torch.float) * attention_threshold

        if (use_decreasing_attention_coefficient):
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in [[0, 10], [6, 7], [7, 11], [10, 11],
                                            [1, 5], [2, 3], [3, 8], [4, 13]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{2, 1} = 0.7 > 0.5, but
                    # \alpha_{1, 2} = 0.2, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.45 < 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.2
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7
        else:
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) not in [[0, 10], [6, 7], [7, 11],
                                                [10, 11], [1, 5], [2, 3],
                                                [3, 8], [4, 13], [1, 2]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{1, 2} = 0.4 < 0.5, but
                    # \alpha_{2, 1} = 0.7, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.55 > 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.4
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=False,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 6)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0, 6, 7, 10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [2, 3, 8]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node == 9):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [[0, 6, 7, 10, 11], [1, 5],
                                                    [4, 13], [2, 3, 8], 9, 12]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->5 / 5->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->4 / 4->3.
        self.assertEqual(num_new_primal_edges, 16)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 5], [1, 2], [1, 3], [2, 3],
                                [2, 5], [3, 4]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertEqual(new_petdni_batch[tuple(new_primal_edge)],
                             new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 0--1, 0--5, 2--3 and 3--4.
        idx_new_dual_node = new_petdni_batch[(0, 1)]
        idx_old_dual_node_1 = petdni_batch[(0, 1)]
        idx_old_dual_node_2 = petdni_batch[(5, 6)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(0, 5)]
        idx_old_dual_node_1 = petdni_batch[(6, 12)]
        idx_old_dual_node_2 = petdni_batch[(11, 12)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(2, 3)]
        idx_old_dual_node_1 = petdni_batch[(3, 4)]
        idx_old_dual_node_2 = petdni_batch[(8, 13)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(3, 4)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(8, 9)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 4), (1, 2), (1, 3), (2, 5)]
        old_dual_nodes = [(9, 10), (4, 5), (1, 2), (12, 13)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (2, 3), (1, 2), (
                0, 1), (0, 5), (3, 4), (2, 3), (0, 4), (0, 5), (2, 5)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set):
        #   - {0, 1} -> {0, 4};
        #   - {0, 1} -> {0, 5};
        #   - {0, 1} -> {1, 2};
        #   - {0, 1} -> {1, 3};
        #   - {0, 4} -> {0, 1};
        #   - {0, 4} -> {0, 5};
        #   - {0, 4} -> {3, 4};
        #   - {0, 5} -> {0, 1};
        #   - {0, 5} -> {0, 4};
        #   - {0, 5} -> {2, 5};
        #   - {1, 2} -> {0, 1};
        #   - {1, 2} -> {1, 3};
        #   - {1, 2} -> {2, 3};
        #   - {1, 2} -> {2, 5};
        #   - {1, 3} -> {0, 1};
        #   - {1, 3} -> {1, 2};
        #   - {1, 3} -> {2, 3};
        #   - {1, 3} -> {3, 4};
        #   - {2, 3} -> {1, 2};
        #   - {2, 3} -> {2, 5};
        #   - {2, 3} -> {1, 3};
        #   - {2, 3} -> {3, 4};
        #   - {2, 5} -> {1, 2};
        #   - {2, 5} -> {2, 3};
        #   - {2, 5} -> {0, 5};
        #   - {3, 4} -> {1, 3};
        #   - {3, 4} -> {2, 3};
        #   - {3, 4} -> {0, 4}.
        self.assertEqual(num_new_dual_edges, 28)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()
        dual_node_1 = (0, 1)
        other_dual_nodes = [(0, 4), (0, 5), (1, 2), (1, 3)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (0, 4)
        other_dual_nodes = [(0, 1), (0, 5), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (0, 5)
        other_dual_nodes = [(0, 1), (0, 4), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (1, 2)
        other_dual_nodes = [(0, 1), (1, 3), (2, 3), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (1, 3)
        other_dual_nodes = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (2, 3)
        other_dual_nodes = [(1, 2), (2, 5), (1, 3), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (2, 5)
        other_dual_nodes = [(1, 2), (2, 3), (0, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        dual_node_1 = (3, 4)
        other_dual_nodes = [(1, 3), (2, 3), (0, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)

    def __test_large_simple_mesh_config_A_with_output_self_loops(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges // 2)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so as to pool the following
        # primal edges:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.
        # (cf. file `../../common_data/simple_mesh_large_pool_1.png`)
        if (primal_att_coeff_threshold is not None):
            attention_threshold = primal_att_coeff_threshold
        else:
            attention_threshold = 0.5
        primal_attention_coeffs = torch.rand(
            [num_primal_edges, num_heads],
            dtype=torch.float) * attention_threshold

        if (use_decreasing_attention_coefficient):
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in [[0, 10], [6, 7], [7, 11], [10, 11],
                                            [1, 5], [2, 3], [3, 8], [4, 13]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{2, 1} = 0.7 > 0.5, but
                    # \alpha_{1, 2} = 0.2, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.45 < 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.2
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7
        else:
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) not in [[0, 10], [6, 7], [7, 11],
                                                [10, 11], [1, 5], [2, 3],
                                                [3, 8], [4, 13], [1, 2]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{1, 2} = 0.4 < 0.5, but
                    # \alpha_{2, 1} = 0.7, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.55 > 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.4
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 6)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0, 6, 7, 10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [2, 3, 8]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node == 9):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [[0, 6, 7, 10, 11], [1, 5],
                                                    [4, 13], [2, 3, 8], 9, 12]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->5 / 5->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->4 / 4->3.
        self.assertEqual(num_new_primal_edges, 16)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 5], [1, 2], [1, 3], [2, 3],
                                [2, 5], [3, 4]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertEqual(new_petdni_batch[tuple(new_primal_edge)],
                             new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 0--1, 0--5, 2--3 and 3--4.
        idx_new_dual_node = new_petdni_batch[(0, 1)]
        idx_old_dual_node_1 = petdni_batch[(0, 1)]
        idx_old_dual_node_2 = petdni_batch[(5, 6)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(0, 5)]
        idx_old_dual_node_1 = petdni_batch[(6, 12)]
        idx_old_dual_node_2 = petdni_batch[(11, 12)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(2, 3)]
        idx_old_dual_node_1 = petdni_batch[(3, 4)]
        idx_old_dual_node_2 = petdni_batch[(8, 13)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        idx_new_dual_node = new_petdni_batch[(3, 4)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(8, 9)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 4), (1, 2), (1, 3), (2, 5)]
        old_dual_nodes = [(9, 10), (4, 5), (1, 2), (12, 13)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (2, 3), (1, 2), (
                0, 1), (0, 5), (3, 4), (2, 3), (0, 4), (0, 5), (2, 5)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - {0, 1} -> {0, 4};
        #   - {0, 1} -> {0, 5};
        #   - {0, 1} -> {1, 2};
        #   - {0, 1} -> {1, 3};
        #   - {0, 4} -> {0, 1};
        #   - {0, 4} -> {0, 5};
        #   - {0, 4} -> {3, 4};
        #   - {0, 5} -> {0, 1};
        #   - {0, 5} -> {0, 4};
        #   - {0, 5} -> {2, 5};
        #   - {1, 2} -> {0, 1};
        #   - {1, 2} -> {1, 3};
        #   - {1, 2} -> {2, 3};
        #   - {1, 2} -> {2, 5};
        #   - {1, 3} -> {0, 1};
        #   - {1, 3} -> {1, 2};
        #   - {1, 3} -> {2, 3};
        #   - {1, 3} -> {3, 4};
        #   - {2, 3} -> {1, 2};
        #   - {2, 3} -> {2, 5};
        #   - {2, 3} -> {1, 3};
        #   - {2, 3} -> {3, 4};
        #   - {2, 5} -> {1, 2};
        #   - {2, 5} -> {2, 3};
        #   - {2, 5} -> {0, 5};
        #   - {3, 4} -> {1, 3};
        #   - {3, 4} -> {2, 3};
        #   - {3, 4} -> {0, 4}.
        self.assertEqual(num_new_dual_edges, 28 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()
        dual_node_1 = (0, 1)
        other_dual_nodes = [(0, 4), (0, 5), (1, 2), (1, 3)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (0, 4)
        other_dual_nodes = [(0, 1), (0, 5), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (0, 5)
        other_dual_nodes = [(0, 1), (0, 4), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (1, 2)
        other_dual_nodes = [(0, 1), (1, 3), (2, 3), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (1, 3)
        other_dual_nodes = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (2, 3)
        other_dual_nodes = [(1, 2), (2, 5), (1, 3), (3, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (2, 5)
        other_dual_nodes = [(1, 2), (2, 3), (0, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (3, 4)
        other_dual_nodes = [(1, 3), (2, 3), (0, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[dual_node_1], new_petdni_batch[other_dual_node]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

    def __test_large_simple_mesh_config_B_with_output_self_loops(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1):
        # - Dual-graph configuration B.
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            # Configuration B has double dual nodes.
            self.assertNotEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so as to pool the following
        # primal edges:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.
        # (cf. file `../../common_data/simple_mesh_large_pool_1.png`)
        if (primal_att_coeff_threshold is not None):
            attention_threshold = primal_att_coeff_threshold
        else:
            attention_threshold = 0.5
        primal_attention_coeffs = torch.rand(
            [num_primal_edges, num_heads],
            dtype=torch.float) * attention_threshold

        if (use_decreasing_attention_coefficient):
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in [[0, 10], [6, 7], [7, 11], [10, 11],
                                            [1, 5], [2, 3], [3, 8], [4, 13]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{2, 1} = 0.7 > 0.5, but
                    # \alpha_{1, 2} = 0.2, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.45 < 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.2
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7
        else:
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) not in [[0, 10], [6, 7], [7, 11],
                                                [10, 11], [1, 5], [2, 3],
                                                [3, 8], [4, 13], [1, 2]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{1, 2} = 0.4 < 0.5, but
                    # \alpha_{2, 1} = 0.7, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.55 > 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.4
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 6)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0, 6, 7, 10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [2, 3, 8]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node == 9):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [[0, 6, 7, 10, 11], [1, 5],
                                                    [4, 13], [2, 3, 8], 9, 12]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->5 / 5->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->4 / 4->3.
        self.assertEqual(num_new_primal_edges, 16)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 5], [1, 2], [1, 3], [2, 3],
                                [2, 5], [3, 4]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are not associated to the same
            # dual node (configuration with double dual nodes).
            self.assertNotEqual(new_petdni_batch[tuple(new_primal_edge)],
                                new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 0--1, 0--5, 2--3 and 3--4, in both
        #   directions.
        #   - New (directed) primal edge 0->1 corresponds to old (directed)
        #     primal edges 0->1 and 6->5.
        idx_new_dual_node = new_petdni_batch[(0, 1)]
        idx_old_dual_node_1 = petdni_batch[(0, 1)]
        idx_old_dual_node_2 = petdni_batch[(6, 5)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 1->0 corresponds to old (directed)
        #     primal edges 1->0 and 5->6.
        idx_new_dual_node = new_petdni_batch[(1, 0)]
        idx_old_dual_node_1 = petdni_batch[(1, 0)]
        idx_old_dual_node_2 = petdni_batch[(5, 6)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 0->5 corresponds to old (directed)
        #     primal edges 6->12 and 11->12.
        idx_new_dual_node = new_petdni_batch[(0, 5)]
        idx_old_dual_node_1 = petdni_batch[(6, 12)]
        idx_old_dual_node_2 = petdni_batch[(11, 12)]
        #   - New (directed) primal edge 5->0 corresponds to old (directed)
        #     primal edges 12->6 and 12->11.
        idx_new_dual_node = new_petdni_batch[(5, 0)]
        idx_old_dual_node_1 = petdni_batch[(12, 6)]
        idx_old_dual_node_2 = petdni_batch[(12, 11)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 2->3 corresponds to old (directed)
        #     primal edges 4->3 and 13->8.
        idx_new_dual_node = new_petdni_batch[(2, 3)]
        idx_old_dual_node_1 = petdni_batch[(4, 3)]
        idx_old_dual_node_2 = petdni_batch[(13, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 3->2 corresponds to old (directed)
        #     primal edges 3->4 and 8->13.
        idx_new_dual_node = new_petdni_batch[(3, 2)]
        idx_old_dual_node_1 = petdni_batch[(3, 4)]
        idx_old_dual_node_2 = petdni_batch[(8, 13)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 3->4 corresponds to old (directed)
        #     primal edges 2->9 and 8->9.
        idx_new_dual_node = new_petdni_batch[(3, 4)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(8, 9)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 4->3 corresponds to old (directed)
        #     primal edges 9->2 and 9->8.
        idx_new_dual_node = new_petdni_batch[(4, 3)]
        idx_old_dual_node_1 = petdni_batch[(9, 2)]
        idx_old_dual_node_2 = petdni_batch[(9, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 4), (1, 2), (1, 3), (2, 5)]
        old_dual_nodes = [(10, 9), (5, 4), (1, 2), (13, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            # 'Forward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)
            # 'Backward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node[::-1]]
            idx_old_dual_node = petdni_batch[old_dual_node[::-1]]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - (0->1) -> (4->0);
        #   - (0->1) -> (5->0);
        #   - (0->1) -> (1->2);
        #   - (0->1) -> (1->3);
        #   - (1->0) -> (0->4);
        #   - (1->0) -> (0->5);
        #   - (1->0) -> (2->1);
        #   - (1->0) -> (3->1);
        #   - (0->4) -> (1->0);
        #   - (0->4) -> (5->0);
        #   - (0->4) -> (4->3);
        #   - (4->0) -> (0->1);
        #   - (4->0) -> (0->5);
        #   - (4->0) -> (3->4);
        #   - (0->5) -> (1->0);
        #   - (0->5) -> (4->0);
        #   - (0->5) -> (5->2);
        #   - (5->0) -> (0->1);
        #   - (5->0) -> (0->4);
        #   - (5->0) -> (2->5);
        #   - (1->2) -> (0->1);
        #   - (1->2) -> (3->1);
        #   - (1->2) -> (2->3);
        #   - (1->2) -> (2->5);
        #   - (2->1) -> (1->0);
        #   - (2->1) -> (1->3);
        #   - (2->1) -> (3->2);
        #   - (2->1) -> (5->2);
        #   - (1->3) -> (0->1);
        #   - (1->3) -> (2->1);
        #   - (1->3) -> (3->2);
        #   - (1->3) -> (3->4);
        #   - (3->1) -> (1->0);
        #   - (3->1) -> (1->2);
        #   - (3->1) -> (2->3);
        #   - (3->1) -> (4->3);
        #   - (2->3) -> (1->2);
        #   - (2->3) -> (5->2);
        #   - (2->3) -> (3->1);
        #   - (2->3) -> (3->4);
        #   - (3->2) -> (2->1);
        #   - (3->2) -> (2->5);
        #   - (3->2) -> (1->3);
        #   - (3->2) -> (4->3);
        #   - (2->5) -> (1->2);
        #   - (2->5) -> (3->2);
        #   - (2->5) -> (5->0);
        #   - (5->2) -> (2->1);
        #   - (5->2) -> (2->3);
        #   - (5->2) -> (0->5);
        #   - (3->4) -> (1->3);
        #   - (3->4) -> (2->3);
        #   - (3->4) -> (4->0);
        #   - (4->3) -> (3->1);
        #   - (4->3) -> (3->2);
        #   - (4->3) -> (0->4).
        self.assertEqual(num_new_dual_edges, 56 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()

        dual_node_to_neighbors = {
            (0, 1): [(4, 0), (5, 0), (1, 2), (1, 3)],
            (0, 4): [(1, 0), (5, 0), (4, 3)],
            (0, 5): [(1, 0), (4, 0), (5, 2)],
            (1, 2): [(0, 1), (3, 1), (2, 3), (2, 5)],
            (1, 3): [(0, 1), (2, 1), (3, 2), (3, 4)],
            (2, 3): [(1, 2), (5, 2), (3, 1), (3, 4)],
            (2, 5): [(1, 2), (3, 2), (5, 0)],
            (3, 4): [(1, 3), (2, 3), (4, 0)]
        }
        for new_dual_node, other_dual_nodes in dual_node_to_neighbors.items():
            for other_dual_node in other_dual_nodes:
                self.assertTrue([
                    new_petdni_batch[new_dual_node],
                    new_petdni_batch[other_dual_node]
                ] in new_dual_edge_index_list)
                # 'Opposite' dual node.
                self.assertTrue([
                    new_petdni_batch[new_dual_node[::-1]], new_petdni_batch[
                        other_dual_node[::-1]]
                ] in new_dual_edge_index_list)
            # Self-loop.
            self.assertTrue([
                new_petdni_batch[new_dual_node], new_petdni_batch[new_dual_node]
            ] in new_dual_edge_index_list)
            # Self-loop of 'opposite' dual node.
            self.assertTrue([
                new_petdni_batch[new_dual_node[::-1]], new_petdni_batch[
                    new_dual_node[::-1]]
            ] in new_dual_edge_index_list)

    def __test_large_simple_mesh_config_C_with_output_self_loops(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1):
        # - Dual-graph configuration C.
        single_dual_nodes = False
        undirected_dual_edges = False
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            # Configuration C has double dual nodes.
            self.assertNotEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, but by definition of dual-graph configuration C each
        #   node in the dual graph has 2 incoming edges  and 2 outgoing edges.
        #   However, since there are no self-loops in the dual graph, each
        #   incoming edge for a certain dual node is also an outgoing edge for
        #   another dual node, and the total number of (directed) edges in the
        #   dual graph is 2 times the number of dual nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 2)
        self.assertEqual(num_dual_nodes, num_primal_edges)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so as to pool the following
        # primal edges:
        # - 0->10  / 10->0;
        # - 6->7   / 7->6;
        # - 7->11  / 11->7;
        # - 10->11 / 11->10;
        # - 1->5   / 5->1;
        # - 2->3   / 3->2;
        # - 3->8   / 8->3;
        # - 4->13  / 13->4.
        # (cf. file `../../common_data/simple_mesh_large_pool_1.png`)
        if (primal_att_coeff_threshold is not None):
            attention_threshold = primal_att_coeff_threshold
        else:
            attention_threshold = 0.5
        primal_attention_coeffs = torch.rand(
            [num_primal_edges, num_heads],
            dtype=torch.float) * attention_threshold

        if (use_decreasing_attention_coefficient):
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in [[0, 10], [6, 7], [7, 11], [10, 11],
                                            [1, 5], [2, 3], [3, 8], [4, 13]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{2, 1} = 0.7 > 0.5, but
                    # \alpha_{1, 2} = 0.2, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.45 < 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.2
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7
        else:
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) not in [[0, 10], [6, 7], [7, 11],
                                                [10, 11], [1, 5], [2, 3],
                                                [3, 8], [4, 13], [1, 2]]):
                    primal_attention_coeffs[edge_idx] += (1 -
                                                          attention_threshold)
                elif (primal_edge == [1, 2]):
                    # Further test: set \alpha_{1, 2} = 0.4 < 0.5, but
                    # \alpha_{2, 1} = 0.7, so that
                    # (\alpha_{1, 2} + \alpha_{2, 1}) / 2 = 0.55 > 0.5, and the
                    # edges 1->2 / 2->1 do not get pooled.
                    primal_attention_coeffs[edge_idx] = 0.4
                elif (primal_edge == [2, 1]):
                    primal_attention_coeffs[edge_idx] = 0.7

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 6)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0, 6, 7, 10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [2, 3, 8]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node == 9):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [[0, 6, 7, 10, 11], [1, 5],
                                                    [4, 13], [2, 3, 8], 9, 12]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->5 / 5->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->4 / 4->3.
        self.assertEqual(num_new_primal_edges, 16)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 5], [1, 2], [1, 3], [2, 3],
                                [2, 5], [3, 4]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are not associated to the same
            # dual node (configuration with double dual nodes).
            self.assertNotEqual(new_petdni_batch[tuple(new_primal_edge)],
                                new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 0--1, 0--5, 2--3 and 3--4, in both
        #   directions.
        #   - New (directed) primal edge 0->1 corresponds to old (directed)
        #     primal edges 0->1 and 6->5.
        idx_new_dual_node = new_petdni_batch[(0, 1)]
        idx_old_dual_node_1 = petdni_batch[(0, 1)]
        idx_old_dual_node_2 = petdni_batch[(6, 5)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 1->0 corresponds to old (directed)
        #     primal edges 1->0 and 5->6.
        idx_new_dual_node = new_petdni_batch[(1, 0)]
        idx_old_dual_node_1 = petdni_batch[(1, 0)]
        idx_old_dual_node_2 = petdni_batch[(5, 6)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 0->5 corresponds to old (directed)
        #     primal edges 6->12 and 11->12.
        idx_new_dual_node = new_petdni_batch[(0, 5)]
        idx_old_dual_node_1 = petdni_batch[(6, 12)]
        idx_old_dual_node_2 = petdni_batch[(11, 12)]
        #   - New (directed) primal edge 5->0 corresponds to old (directed)
        #     primal edges 12->6 and 12->11.
        idx_new_dual_node = new_petdni_batch[(5, 0)]
        idx_old_dual_node_1 = petdni_batch[(12, 6)]
        idx_old_dual_node_2 = petdni_batch[(12, 11)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 2->3 corresponds to old (directed)
        #     primal edges 4->3 and 13->8.
        idx_new_dual_node = new_petdni_batch[(2, 3)]
        idx_old_dual_node_1 = petdni_batch[(4, 3)]
        idx_old_dual_node_2 = petdni_batch[(13, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 3->2 corresponds to old (directed)
        #     primal edges 3->4 and 8->13.
        idx_new_dual_node = new_petdni_batch[(3, 2)]
        idx_old_dual_node_1 = petdni_batch[(3, 4)]
        idx_old_dual_node_2 = petdni_batch[(8, 13)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 3->4 corresponds to old (directed)
        #     primal edges 2->9 and 8->9.
        idx_new_dual_node = new_petdni_batch[(3, 4)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(8, 9)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 4->3 corresponds to old (directed)
        #     primal edges 9->2 and 9->8.
        idx_new_dual_node = new_petdni_batch[(4, 3)]
        idx_old_dual_node_1 = petdni_batch[(9, 2)]
        idx_old_dual_node_2 = petdni_batch[(9, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 4), (1, 2), (1, 3), (2, 5)]
        old_dual_nodes = [(10, 9), (5, 4), (1, 2), (13, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            # 'Forward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)
            # 'Backward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node[::-1]]
            idx_old_dual_node = petdni_batch[old_dual_node[::-1]]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - (0->1) -> (1->2);
        #   - (0->1) -> (1->3);
        #   - (1->0) -> (0->4);
        #   - (1->0) -> (0->5);
        #   - (0->4) -> (4->3);
        #   - (4->0) -> (0->1);
        #   - (4->0) -> (0->5);
        #   - (0->5) -> (5->2);
        #   - (5->0) -> (0->1);
        #   - (5->0) -> (0->4);
        #   - (1->2) -> (2->3);
        #   - (1->2) -> (2->5);
        #   - (2->1) -> (1->0);
        #   - (2->1) -> (1->3);
        #   - (1->3) -> (3->2);
        #   - (1->3) -> (3->4);
        #   - (3->1) -> (1->0);
        #   - (3->1) -> (1->2);
        #   - (2->3) -> (3->1);
        #   - (2->3) -> (3->4);
        #   - (3->2) -> (2->1);
        #   - (3->2) -> (2->5);
        #   - (2->5) -> (5->0);
        #   - (5->2) -> (2->1);
        #   - (5->2) -> (2->3);
        #   - (3->4) -> (4->0);
        #   - (4->3) -> (3->1);
        #   - (4->3) -> (3->2);
        self.assertEqual(num_new_dual_edges, 28 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()

        dual_node_to_neighbors = {
            (0, 1): [(1, 2), (1, 3)],
            (1, 0): [(0, 4), (0, 5)],
            (0, 4): [(4, 3)],
            (4, 0): [(0, 1), (0, 5)],
            (0, 5): [(5, 2)],
            (5, 0): [(0, 1), (0, 4)],
            (1, 2): [(2, 3), (2, 5)],
            (2, 1): [(1, 0), (1, 3)],
            (1, 3): [(3, 2), (3, 4)],
            (3, 1): [(1, 0), (1, 2)],
            (2, 3): [(3, 1), (3, 4)],
            (3, 2): [(2, 1), (2, 5)],
            (2, 5): [(5, 0)],
            (5, 2): [(2, 1), (2, 3)],
            (3, 4): [(4, 0)],
            (4, 3): [(3, 1), (3, 2)]
        }
        for new_dual_node, other_dual_nodes in dual_node_to_neighbors.items():
            for other_dual_node in other_dual_nodes:
                self.assertTrue([
                    new_petdni_batch[new_dual_node],
                    new_petdni_batch[other_dual_node]
                ] in new_dual_edge_index_list)
            # Self-loop.
            self.assertTrue([
                new_petdni_batch[new_dual_node], new_petdni_batch[new_dual_node]
            ] in new_dual_edge_index_list)

    # * Allow only non-consecutive edges.
    def __test_config_A_no_output_self_loops_nonconsecutive(
        self, use_decreasing_attention_coefficient=True, num_heads=1):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges // 2)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so that the primal edges have
        # associated attention coefficients in this order:
        # - 4->13  / 13->4;
        # - 10->11 / 11->10;
        # - 0->10  / 10->0 [not pooled, because 10->11 / 11->10 was pooled];
        # - 2->3   / 3->2;
        # - 3->8   / 8->3  [not pooled, because 2->3 / 3->2 was pooled];
        # - 6->7   / 7->6;
        # - 1->5   / 5->1;
        # - 7->11  / 11->7 [not pooled, because 10->11 / 11->10 and 6->7 / 7->6
        #                   were pooled];
        # - 1->2   / 2->1  [not pooled, because 2->3 / 3->2 and 1->5 / 5->1 were
        #                   pooled];
        # - 8->9   / 9->8;
        # - ...            [other edges that are not pooled]
        # (cf. file `../../common_data/simple_mesh_large_pool_2.png`)
        attention_threshold = 0.5

        edges_to_pool = [[8, 9], [1, 2], [7, 11], [1, 5], [6, 7], [3, 8],
                         [2, 3], [0, 10], [10, 11], [4, 13]]
        if (use_decreasing_attention_coefficient):
            primal_attention_coeffs = torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * attention_threshold
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = attention_threshold + (
                        1 - attention_threshold) * (
                            float(pooling_idx) / len(edges_to_pool) +
                            torch.rand([num_heads], dtype=torch.float) * 1. /
                            len(edges_to_pool))
        else:
            primal_attention_coeffs = attention_threshold + torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * (1 - attention_threshold)
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = (
                        attention_threshold - attention_threshold *
                        (float(pooling_idx) / len(edges_to_pool) +
                         torch.rand([num_heads], dtype=torch.float) * 1. /
                         len(edges_to_pool)))

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=False,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 8)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [2, 3]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node in [6, 7]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node in [8, 9]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
            elif (old_primal_node in [10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 6)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 7)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [
                0, [1, 5], [2, 3], [4, 13], [6, 7], [8, 9], [10, 11], 12
            ]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->6 / 6->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 1->4 / 4->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->5 / 5->3;
        #   - 3->7 / 7->3;
        #   - 4->6 / 6->4;
        #   - 4->7 / 7->4;
        #   - 5->6 / 6->5;
        #   - 6->7 / 7->6.
        self.assertEqual(num_new_primal_edges, 28)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 4],
                                [2, 3], [2, 5], [3, 5], [3, 7], [4, 6], [4, 7],
                                [5, 6], [6, 7]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertEqual(new_petdni_batch[tuple(new_primal_edge)],
                             new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 2--5.
        #   - New (directed) primal edge 2->5 corresponds to old (directed)
        #     primal edges 2->9 and 3->8.
        idx_new_dual_node = new_petdni_batch[(2, 5)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(3, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4),
                          (2, 3), (3, 5), (3, 7), (4, 6), (4, 7), (5, 6),
                          (6, 7)]
        old_dual_nodes = [(0, 1), (0, 7), (0, 10), (1, 2), (4, 5), (5, 6),
                          (3, 4), (8, 13), (12, 13), (7, 11), (6, 12), (9, 10),
                          (11, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (
                3, 4), (3, 8), (4,
                                5), (5,
                                     6), (6,
                                          12), (7,
                                                11), (8,
                                                      13), (9,
                                                            10), (11,
                                                                  12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (
                2, 3), (2, 5), (1, 3), (1, 4), (4, 7), (4,
                                                        6), (3,
                                                             5), (5,
                                                                  6), (6,
                                                                       7), (3,
                                                                            7)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set):
        #   - {0, 1} -> {0, 4};
        #   - {0, 1} -> {0, 6};
        #   - {0, 1} -> {1, 2};
        #   - {0, 1} -> {1, 3};
        #   - {0, 1} -> {1, 4};
        #   - {0, 4} -> {0, 1};
        #   - {0, 4} -> {0, 6};
        #   - {0, 4} -> {1, 4};
        #   - {0, 4} -> {4, 6};
        #   - {0, 4} -> {4, 7};
        #   - {0, 6} -> {0, 1};
        #   - {0, 6} -> {0, 4};
        #   - {0, 6} -> {4, 6};
        #   - {0, 6} -> {5, 6};
        #   - {0, 6} -> {6, 7};
        #   - {1, 2} -> {0, 1};
        #   - {1, 2} -> {1, 3};
        #   - {1, 2} -> {1, 4};
        #   - {1, 2} -> {2, 3};
        #   - {1, 2} -> {2, 5};
        #   - {1, 3} -> {0, 1};
        #   - {1, 3} -> {1, 2};
        #   - {1, 3} -> {1, 4};
        #   - {1, 3} -> {2, 3};
        #   - {1, 3} -> {3, 5};
        #   - {1, 3} -> {3, 7};
        #   - {1, 4} -> {0, 1};
        #   - {1, 4} -> {0, 4};
        #   - {1, 4} -> {1, 2};
        #   - {1, 4} -> {1, 3};
        #   - {1, 4} -> {4, 6};
        #   - {1, 4} -> {4, 7};
        #   - {2, 3} -> {1, 2};
        #   - {2, 3} -> {1, 3};
        #   - {2, 3} -> {2, 5};
        #   - {2, 3} -> {3, 5};
        #   - {2, 3} -> {3, 7};
        #   - {2, 5} -> {1, 2};
        #   - {2, 5} -> {2, 3};
        #   - {2, 5} -> {3, 5};
        #   - {2, 5} -> {5, 6};
        #   - {3, 5} -> {1, 3};
        #   - {3, 5} -> {2, 3};
        #   - {3, 5} -> {2, 5};
        #   - {3, 5} -> {3, 7};
        #   - {3, 5} -> {5, 6};
        #   - {3, 7} -> {1, 3};
        #   - {3, 7} -> {2, 3};
        #   - {3, 7} -> {3, 5};
        #   - {3, 7} -> {4, 7};
        #   - {3, 7} -> {6, 7};
        #   - {4, 6} -> {0, 4};
        #   - {4, 6} -> {0, 6};
        #   - {4, 6} -> {1, 4};
        #   - {4, 6} -> {4, 7};
        #   - {4, 6} -> {5, 6};
        #   - {4, 6} -> {6, 7};
        #   - {4, 7} -> {0, 4};
        #   - {4, 7} -> {1, 4};
        #   - {4, 7} -> {3, 7};
        #   - {4, 7} -> {4, 6};
        #   - {4, 7} -> {6, 7};
        #   - {5, 6} -> {0, 6};
        #   - {5, 6} -> {2, 5};
        #   - {5, 6} -> {3, 5};
        #   - {5, 6} -> {4, 6};
        #   - {5, 6} -> {6, 7};
        #   - {6, 7} -> {0, 6};
        #   - {6, 7} -> {3, 7};
        #   - {6, 7} -> {4, 6};
        #   - {6, 7} -> {4, 7};
        #   - {6, 7} -> {5, 6}.
        self.assertEqual(num_new_dual_edges, 72)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()
        dual_node_1 = (0, 1)
        other_dual_nodes = [(0, 4), (0, 6), (1, 2), (1, 3), (1, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (0, 4)
        other_dual_nodes = [(0, 1), (0, 6), (1, 4), (4, 6), (4, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (0, 6)
        other_dual_nodes = [(0, 1), (0, 4), (4, 6), (5, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (1, 2)
        other_dual_nodes = [(0, 1), (1, 3), (1, 4), (2, 3), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (1, 3)
        other_dual_nodes = [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (3, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (1, 4)
        other_dual_nodes = [(0, 1), (0, 4), (1, 2), (1, 3), (4, 6), (4, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (2, 3)
        other_dual_nodes = [(1, 2), (1, 3), (2, 5), (3, 5), (3, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (2, 5)
        other_dual_nodes = [(1, 2), (2, 3), (3, 5), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (3, 5)
        other_dual_nodes = [(1, 3), (2, 3), (2, 5), (3, 7), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (3, 7)
        other_dual_nodes = [(1, 3), (2, 3), (3, 5), (4, 7), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (4, 6)
        other_dual_nodes = [(0, 4), (0, 6), (1, 4), (4, 7), (5, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (4, 7)
        other_dual_nodes = [(0, 4), (1, 4), (3, 7), (4, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (5, 6)
        other_dual_nodes = [(0, 6), (2, 5), (3, 5), (4, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        dual_node_1 = (6, 7)
        other_dual_nodes = [(0, 6), (3, 7), (4, 6), (4, 7), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

    def __test_config_A_with_output_self_loops_nonconsecutive(
        self, use_decreasing_attention_coefficient=True, num_heads=1):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges // 2)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so that the primal edges have
        # associated attention coefficients in this order:
        # - 4->13  / 13->4;
        # - 10->11 / 11->10;
        # - 0->10  / 10->0 [not pooled, because 10->11 / 11->10 was pooled];
        # - 2->3   / 3->2;
        # - 3->8   / 8->3  [not pooled, because 2->3 / 3->2 was pooled];
        # - 6->7   / 7->6;
        # - 1->5   / 5->1;
        # - 7->11  / 11->7 [not pooled, because 10->11 / 11->10 and 6->7 / 7->6
        #                   were pooled];
        # - 1->2   / 2->1  [not pooled, because 2->3 / 3->2 and 1->5 / 5->1 were
        #                   pooled];
        # - 8->9   / 9->8;
        # - ...            [other edges that are not pooled]
        # (cf. file `../../common_data/simple_mesh_large_pool_2.png`)
        attention_threshold = 0.5

        edges_to_pool = [[8, 9], [1, 2], [7, 11], [1, 5], [6, 7], [3, 8],
                         [2, 3], [0, 10], [10, 11], [4, 13]]
        if (use_decreasing_attention_coefficient):
            primal_attention_coeffs = torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * attention_threshold
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = attention_threshold + (
                        1 - attention_threshold) * (
                            float(pooling_idx) / len(edges_to_pool) +
                            torch.rand([num_heads], dtype=torch.float) * 1. /
                            len(edges_to_pool))
        else:
            primal_attention_coeffs = attention_threshold + torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * (1 - attention_threshold)
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = (
                        attention_threshold - attention_threshold *
                        (float(pooling_idx) / len(edges_to_pool) +
                         torch.rand([num_heads], dtype=torch.float) * 1. /
                         len(edges_to_pool)))

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 8)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [2, 3]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node in [6, 7]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node in [8, 9]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
            elif (old_primal_node in [10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 6)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 7)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [
                0, [1, 5], [2, 3], [4, 13], [6, 7], [8, 9], [10, 11], 12
            ]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->6 / 6->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 1->4 / 4->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->5 / 5->3;
        #   - 3->7 / 7->3;
        #   - 4->6 / 6->4;
        #   - 4->7 / 7->4;
        #   - 5->6 / 6->5;
        #   - 6->7 / 7->6.
        self.assertEqual(num_new_primal_edges, 28)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 4],
                                [2, 3], [2, 5], [3, 5], [3, 7], [4, 6], [4, 7],
                                [5, 6], [6, 7]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertEqual(new_petdni_batch[tuple(new_primal_edge)],
                             new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 2--5.
        #   - New (directed) primal edge 2->5 corresponds to old (directed)
        #     primal edges 2->9 and 3->8.
        idx_new_dual_node = new_petdni_batch[(2, 5)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(3, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4),
                          (2, 3), (3, 5), (3, 7), (4, 6), (4, 7), (5, 6),
                          (6, 7)]
        old_dual_nodes = [(0, 1), (0, 7), (0, 10), (1, 2), (4, 5), (5, 6),
                          (3, 4), (8, 13), (12, 13), (7, 11), (6, 12), (9, 10),
                          (11, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (
                3, 4), (3, 8), (4,
                                5), (5,
                                     6), (6,
                                          12), (7,
                                                11), (8,
                                                      13), (9,
                                                            10), (11,
                                                                  12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (
                2, 3), (2, 5), (1, 3), (1, 4), (4, 7), (4,
                                                        6), (3,
                                                             5), (5,
                                                                  6), (6,
                                                                       7), (3,
                                                                            7)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - {0, 1} -> {0, 4};
        #   - {0, 1} -> {0, 6};
        #   - {0, 1} -> {1, 2};
        #   - {0, 1} -> {1, 3};
        #   - {0, 1} -> {1, 4};
        #   - {0, 4} -> {0, 1};
        #   - {0, 4} -> {0, 6};
        #   - {0, 4} -> {1, 4};
        #   - {0, 4} -> {4, 6};
        #   - {0, 4} -> {4, 7};
        #   - {0, 6} -> {0, 1};
        #   - {0, 6} -> {0, 4};
        #   - {0, 6} -> {4, 6};
        #   - {0, 6} -> {5, 6};
        #   - {0, 6} -> {6, 7};
        #   - {1, 2} -> {0, 1};
        #   - {1, 2} -> {1, 3};
        #   - {1, 2} -> {1, 4};
        #   - {1, 2} -> {2, 3};
        #   - {1, 2} -> {2, 5};
        #   - {1, 3} -> {0, 1};
        #   - {1, 3} -> {1, 2};
        #   - {1, 3} -> {1, 4};
        #   - {1, 3} -> {2, 3};
        #   - {1, 3} -> {3, 5};
        #   - {1, 3} -> {3, 7};
        #   - {1, 4} -> {0, 1};
        #   - {1, 4} -> {0, 4};
        #   - {1, 4} -> {1, 2};
        #   - {1, 4} -> {1, 3};
        #   - {1, 4} -> {4, 6};
        #   - {1, 4} -> {4, 7};
        #   - {2, 3} -> {1, 2};
        #   - {2, 3} -> {1, 3};
        #   - {2, 3} -> {2, 5};
        #   - {2, 3} -> {3, 5};
        #   - {2, 3} -> {3, 7};
        #   - {2, 5} -> {1, 2};
        #   - {2, 5} -> {2, 3};
        #   - {2, 5} -> {3, 5};
        #   - {2, 5} -> {5, 6};
        #   - {3, 5} -> {1, 3};
        #   - {3, 5} -> {2, 3};
        #   - {3, 5} -> {2, 5};
        #   - {3, 5} -> {3, 7};
        #   - {3, 5} -> {5, 6};
        #   - {3, 7} -> {1, 3};
        #   - {3, 7} -> {2, 3};
        #   - {3, 7} -> {3, 5};
        #   - {3, 7} -> {4, 7};
        #   - {3, 7} -> {6, 7};
        #   - {4, 6} -> {0, 4};
        #   - {4, 6} -> {0, 6};
        #   - {4, 6} -> {1, 4};
        #   - {4, 6} -> {4, 7};
        #   - {4, 6} -> {5, 6};
        #   - {4, 6} -> {6, 7};
        #   - {4, 7} -> {0, 4};
        #   - {4, 7} -> {1, 4};
        #   - {4, 7} -> {3, 7};
        #   - {4, 7} -> {4, 6};
        #   - {4, 7} -> {6, 7};
        #   - {5, 6} -> {0, 6};
        #   - {5, 6} -> {2, 5};
        #   - {5, 6} -> {3, 5};
        #   - {5, 6} -> {4, 6};
        #   - {5, 6} -> {6, 7};
        #   - {6, 7} -> {0, 6};
        #   - {6, 7} -> {3, 7};
        #   - {6, 7} -> {4, 6};
        #   - {6, 7} -> {4, 7};
        #   - {6, 7} -> {5, 6}.
        self.assertEqual(num_new_dual_edges, 72 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()
        dual_node_1 = (0, 1)
        other_dual_nodes = [(0, 4), (0, 6), (1, 2), (1, 3), (1, 4)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (0, 4)
        other_dual_nodes = [(0, 1), (0, 6), (1, 4), (4, 6), (4, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (0, 6)
        other_dual_nodes = [(0, 1), (0, 4), (4, 6), (5, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (1, 2)
        other_dual_nodes = [(0, 1), (1, 3), (1, 4), (2, 3), (2, 5)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (1, 3)
        other_dual_nodes = [(0, 1), (1, 2), (1, 4), (2, 3), (3, 5), (3, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (1, 4)
        other_dual_nodes = [(0, 1), (0, 4), (1, 2), (1, 3), (4, 6), (4, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (2, 3)
        other_dual_nodes = [(1, 2), (1, 3), (2, 5), (3, 5), (3, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (2, 5)
        other_dual_nodes = [(1, 2), (2, 3), (3, 5), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (3, 5)
        other_dual_nodes = [(1, 3), (2, 3), (2, 5), (3, 7), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (3, 7)
        other_dual_nodes = [(1, 3), (2, 3), (3, 5), (4, 7), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (4, 6)
        other_dual_nodes = [(0, 4), (0, 6), (1, 4), (4, 7), (5, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (4, 7)
        other_dual_nodes = [(0, 4), (1, 4), (3, 7), (4, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (5, 6)
        other_dual_nodes = [(0, 6), (2, 5), (3, 5), (4, 6), (6, 7)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

        dual_node_1 = (6, 7)
        other_dual_nodes = [(0, 6), (3, 7), (4, 6), (4, 7), (5, 6)]
        for other_dual_node in other_dual_nodes:
            self.assertTrue([
                new_petdni_batch[other_dual_node], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)
        # Self-loop.
        self.assertTrue(
            [new_petdni_batch[dual_node_1], new_petdni_batch[dual_node_1]
            ] in new_dual_edge_index_list)

    def __test_config_B_with_output_self_loops_nonconsecutive(
        self, use_decreasing_attention_coefficient=True, num_heads=1):
        # - Dual-graph configuration B.
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertNotEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, hence each node in the dual graph has 4 incoming edges
        #   and 4 outgoing edges. However, since there are no self-loops in the
        #   dual graph, each incoming edge for a certain dual node is also an
        #   outgoing edge for another dual node, and the total number of
        #   (directed) edges in the dual graph is 4 times the number of dual
        #   nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 4)
        self.assertEqual(num_dual_nodes, num_primal_edges)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so that the primal edges have
        # associated attention coefficients in this order:
        # - 4->13  / 13->4;
        # - 10->11 / 11->10;
        # - 0->10  / 10->0 [not pooled, because 10->11 / 11->10 was pooled];
        # - 2->3   / 3->2;
        # - 3->8   / 8->3  [not pooled, because 2->3 / 3->2 was pooled];
        # - 6->7   / 7->6;
        # - 1->5   / 5->1;
        # - 7->11  / 11->7 [not pooled, because 10->11 / 11->10 and 6->7 / 7->6
        #                   were pooled];
        # - 1->2   / 2->1  [not pooled, because 2->3 / 3->2 and 1->5 / 5->1 were
        #                   pooled];
        # - 8->9   / 9->8;
        # - ...            [other edges that are not pooled]
        # (cf. file `../../common_data/simple_mesh_large_pool_2.png`)
        attention_threshold = 0.5

        edges_to_pool = [[8, 9], [1, 2], [7, 11], [1, 5], [6, 7], [3, 8],
                         [2, 3], [0, 10], [10, 11], [4, 13]]
        if (use_decreasing_attention_coefficient):
            primal_attention_coeffs = torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * attention_threshold
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = attention_threshold + (
                        1 - attention_threshold) * (
                            float(pooling_idx) / len(edges_to_pool) +
                            torch.rand([num_heads], dtype=torch.float) * 1. /
                            len(edges_to_pool))
        else:
            primal_attention_coeffs = attention_threshold + torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * (1 - attention_threshold)
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = (
                        attention_threshold - attention_threshold *
                        (float(pooling_idx) / len(edges_to_pool) +
                         torch.rand([num_heads], dtype=torch.float) * 1. /
                         len(edges_to_pool)))

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 8)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [2, 3]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node in [6, 7]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node in [8, 9]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
            elif (old_primal_node in [10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 6)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 7)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [
                0, [1, 5], [2, 3], [4, 13], [6, 7], [8, 9], [10, 11], 12
            ]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->6 / 6->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 1->4 / 4->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->5 / 5->3;
        #   - 3->7 / 7->3;
        #   - 4->6 / 6->4;
        #   - 4->7 / 7->4;
        #   - 5->6 / 6->5;
        #   - 6->7 / 7->6.
        self.assertEqual(num_new_primal_edges, 28)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 4],
                                [2, 3], [2, 5], [3, 5], [3, 7], [4, 6], [4, 7],
                                [5, 6], [6, 7]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertNotEqual(new_petdni_batch[tuple(new_primal_edge)],
                                new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 2--5, in both directions.
        #   - New (directed) primal edge 2->5 corresponds to old (directed)
        #     primal edges 2->9 and 3->8.
        idx_new_dual_node = new_petdni_batch[(2, 5)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(3, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 5->2 corresponds to old (directed)
        #     primal edges 9->2 and 8->3.
        idx_new_dual_node = new_petdni_batch[(5, 2)]
        idx_old_dual_node_1 = petdni_batch[(9, 2)]
        idx_old_dual_node_2 = petdni_batch[(8, 3)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4),
                          (2, 3), (3, 5), (3, 7), (4, 6), (4, 7), (5, 6),
                          (6, 7)]
        old_dual_nodes = [(0, 1), (0, 7), (0, 10), (1, 2), (5, 4), (5, 6),
                          (3, 4), (13, 8), (13, 12), (7, 11), (6, 12), (9, 10),
                          (11, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            # 'Forward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)
            # 'Backward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node[::-1]]
            idx_old_dual_node = petdni_batch[old_dual_node[::-1]]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - (0->1) -> (1->2);
        #   - (0->1) -> (1->3);
        #   - (0->1) -> (1->4);
        #   - (0->1) -> (4->0);
        #   - (0->1) -> (6->0);
        #   - (1->0) -> (2->1);
        #   - (1->0) -> (3->1);
        #   - (1->0) -> (4->1);
        #   - (1->0) -> (0->4);
        #   - (1->0) -> (0->6);
        #   - (0->4) -> (4->1);
        #   - (0->4) -> (4->6);
        #   - (0->4) -> (4->7);
        #   - (0->4) -> (1->0);
        #   - (0->4) -> (6->0);
        #   - (4->0) -> (1->4);
        #   - (4->0) -> (6->4);
        #   - (4->0) -> (7->4);
        #   - (4->0) -> (0->1);
        #   - (4->0) -> (0->6);
        #   - (0->6) -> (6->4);
        #   - (0->6) -> (6->5);
        #   - (0->6) -> (6->7);
        #   - (0->6) -> (1->0);
        #   - (0->6) -> (4->0);
        #   - (6->0) -> (4->6);
        #   - (6->0) -> (5->6);
        #   - (6->0) -> (7->6);
        #   - (6->0) -> (0->1);
        #   - (6->0) -> (0->4);
        #   - (1->2) -> (2->3);
        #   - (1->2) -> (2->5);
        #   - (1->2) -> (0->1);
        #   - (1->2) -> (3->1);
        #   - (1->2) -> (4->1);
        #   - (2->1) -> (3->2);
        #   - (2->1) -> (5->2);
        #   - (2->1) -> (1->0);
        #   - (2->1) -> (1->3);
        #   - (2->1) -> (1->4);
        #   - (1->3) -> (3->2);
        #   - (1->3) -> (3->5);
        #   - (1->3) -> (3->7);
        #   - (1->3) -> (0->1);
        #   - (1->3) -> (2->1);
        #   - (1->3) -> (4->1);
        #   - (3->1) -> (2->3);
        #   - (3->1) -> (5->3);
        #   - (3->1) -> (7->3);
        #   - (3->1) -> (1->0);
        #   - (3->1) -> (1->2);
        #   - (3->1) -> (1->4);
        #   - (1->4) -> (4->0);
        #   - (1->4) -> (4->6);
        #   - (1->4) -> (4->7);
        #   - (1->4) -> (0->1);
        #   - (1->4) -> (2->1);
        #   - (1->4) -> (3->1);
        #   - (4->1) -> (0->4);
        #   - (4->1) -> (6->4);
        #   - (4->1) -> (7->4);
        #   - (4->1) -> (1->0);
        #   - (4->1) -> (1->2);
        #   - (4->1) -> (1->3);
        #   - (2->3) -> (3->1);
        #   - (2->3) -> (3->5);
        #   - (2->3) -> (3->7);
        #   - (2->3) -> (1->2);
        #   - (2->3) -> (5->2);
        #   - (3->2) -> (1->3);
        #   - (3->2) -> (5->3);
        #   - (3->2) -> (7->3);
        #   - (3->2) -> (2->1);
        #   - (3->2) -> (2->5);
        #   - (2->5) -> (5->3);
        #   - (2->5) -> (5->6);
        #   - (2->5) -> (1->2);
        #   - (2->5) -> (3->2);
        #   - (5->2) -> (3->5);
        #   - (5->2) -> (6->5);
        #   - (5->2) -> (2->1);
        #   - (5->2) -> (2->3);
        #   - (3->5) -> (5->2);
        #   - (3->5) -> (5->6);
        #   - (3->5) -> (1->3);
        #   - (3->5) -> (2->3);
        #   - (3->5) -> (7->3);
        #   - (5->3) -> (2->5);
        #   - (5->3) -> (6->2);
        #   - (5->3) -> (3->1);
        #   - (5->3) -> (3->2);
        #   - (5->3) -> (3->7);
        #   - (3->7) -> (7->4);
        #   - (3->7) -> (7->6);
        #   - (3->7) -> (1->3);
        #   - (3->7) -> (2->3);
        #   - (3->7) -> (5->3);
        #   - (7->3) -> (4->7);
        #   - (7->3) -> (6->7);
        #   - (7->3) -> (3->1);
        #   - (7->3) -> (3->2);
        #   - (7->3) -> (3->5);
        #   - (4->6) -> (6->0);
        #   - (4->6) -> (6->5);
        #   - (4->6) -> (6->7);
        #   - (4->6) -> (0->4);
        #   - (4->6) -> (1->4);
        #   - (4->6) -> (7->4);
        #   - (6->4) -> (0->6);
        #   - (6->4) -> (5->6);
        #   - (6->4) -> (7->6);
        #   - (6->4) -> (4->0);
        #   - (6->4) -> (4->1);
        #   - (6->4) -> (4->7);
        #   - (4->7) -> (7->3);
        #   - (4->7) -> (7->6);
        #   - (4->7) -> (0->4);
        #   - (4->7) -> (1->4);
        #   - (4->7) -> (6->4);
        #   - (7->4) -> (3->7);
        #   - (7->4) -> (6->7);
        #   - (7->4) -> (4->0);
        #   - (7->4) -> (4->1);
        #   - (7->4) -> (4->6);
        #   - (5->6) -> (6->0);
        #   - (5->6) -> (6->4);
        #   - (5->6) -> (6->7);
        #   - (5->6) -> (2->5);
        #   - (5->6) -> (3->5);
        #   - (6->5) -> (0->6);
        #   - (6->5) -> (4->6);
        #   - (6->5) -> (7->6);
        #   - (6->5) -> (5->2);
        #   - (6->5) -> (5->3);
        #   - (6->7) -> (7->3);
        #   - (6->7) -> (7->4);
        #   - (6->7) -> (0->6);
        #   - (6->7) -> (4->6);
        #   - (6->7) -> (5->6).
        #   - (7->6) -> (3->7);
        #   - (7->6) -> (4->7);
        #   - (7->6) -> (6->0);
        #   - (7->6) -> (6->4);
        #   - (7->6) -> (6->5).
        self.assertEqual(num_new_dual_edges, 144 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()

        dual_node_to_neighbors = {
            (0, 1): [(1, 2), (1, 3), (1, 4), (4, 0), (6, 0)],
            (0, 4): [(4, 1), (4, 6), (4, 7), (1, 0), (6, 0)],
            (0, 6): [(6, 4), (6, 5), (6, 7), (1, 0), (4, 0)],
            (1, 2): [(2, 3), (2, 5), (0, 1), (3, 1), (4, 1)],
            (1, 3): [(3, 2), (3, 5), (3, 7), (0, 1), (2, 1), (4, 1)],
            (1, 4): [(4, 0), (4, 6), (4, 7), (0, 1), (2, 1), (3, 1)],
            (2, 3): [(3, 1), (3, 5), (3, 7), (1, 2), (5, 2)],
            (2, 5): [(5, 3), (5, 6), (1, 2), (3, 2)],
            (3, 5): [(5, 2), (5, 6), (1, 3), (2, 3), (7, 3)],
            (3, 7): [(7, 4), (7, 6), (1, 3), (2, 3), (5, 3)],
            (4, 6): [(6, 0), (6, 5), (6, 7), (0, 4), (1, 4), (7, 4)],
            (4, 7): [(7, 3), (7, 6), (0, 4), (1, 4), (6, 4)],
            (5, 6): [(6, 0), (6, 4), (6, 7), (2, 5), (3, 5)],
            (6, 7): [(7, 3), (7, 4), (0, 6), (4, 6), (5, 6)]
        }
        for new_dual_node, other_dual_nodes in dual_node_to_neighbors.items():
            for other_dual_node in other_dual_nodes:
                self.assertTrue([
                    new_petdni_batch[new_dual_node],
                    new_petdni_batch[other_dual_node]
                ] in new_dual_edge_index_list)
            # - Self-loop.
            self.assertTrue([
                new_petdni_batch[new_dual_node], new_petdni_batch[new_dual_node]
            ] in new_dual_edge_index_list)
            # 'Opposite' dual node.
            for other_dual_node in other_dual_nodes:
                self.assertTrue([
                    new_petdni_batch[new_dual_node[::-1]], new_petdni_batch[
                        other_dual_node[::-1]]
                ] in new_dual_edge_index_list)
            # - Self-loop of 'opposite' dual node.
            self.assertTrue([
                new_petdni_batch[new_dual_node[::-1]], new_petdni_batch[
                    new_dual_node[::-1]]
            ] in new_dual_edge_index_list)

    def __test_config_C_with_output_self_loops_nonconsecutive(
        self, use_decreasing_attention_coefficient=True, num_heads=1):
        # - Dual-graph configuration C.
        single_dual_nodes = False
        undirected_dual_edges = False
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh_large.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        petdni = graph_creator.primal_edge_to_dual_node_idx

        (primal_graph_batch, dual_graph_batch,
         petdni_batch) = create_dual_primal_batch(
             primal_graphs_list=[primal_graph],
             dual_graphs_list=[dual_graph],
             primal_edge_to_dual_node_idx_list=[petdni])

        # Primal graph.
        num_primal_edges = primal_graph_batch.num_edges
        num_primal_nodes = maybe_num_nodes(primal_graph_batch.edge_index)
        self.assertEqual(num_primal_edges, 42)
        self.assertEqual(num_primal_nodes, 14)
        # - Check existence of primal edges.
        for edge in [(0, 1), (0, 7), (0, 10), (1, 2), (1, 5), (2, 3), (2, 9),
                     (3, 4), (3, 8), (4, 5), (4, 13), (5, 6), (6, 7), (6, 12),
                     (7, 11), (8, 9), (8, 13), (9, 10), (10, 11), (11, 12),
                     (12, 13)]:
            self.assertNotEqual(petdni_batch[edge], petdni_batch[edge[::-1]])
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
        # - Since the mesh is watertight, the medial graph of the triangulation
        #   is 4-regular, but by definition of dual-graph configuration C each
        #   node in the dual graph has 2 incoming edges  and 2 outgoing edges.
        #   However, since there are no self-loops in the dual graph, each
        #   incoming edge for a certain dual node is also an outgoing edge for
        #   another dual node, and the total number of (directed) edges in the
        #   dual graph is 2 times the number of dual nodes.
        self.assertEqual(num_dual_edges, num_dual_nodes * 2)
        self.assertEqual(num_dual_nodes, num_primal_edges)
        #  - Set the features of each dual node randomly.
        dim_dual_features = dual_graph_batch.num_node_features
        for dual_feature in dual_graph_batch.x:
            dual_feature[:] = torch.rand(dim_dual_features,
                                         dtype=torch.float) * 3

        # Randomly shuffle the primal edge-index matrix.
        permutation = np.random.permutation(num_primal_edges)
        primal_graph_batch.edge_index = (
            primal_graph_batch.edge_index[:, permutation])

        # Set the attention coefficients manually, so that the primal edges have
        # associated attention coefficients in this order:
        # - 4->13  / 13->4;
        # - 10->11 / 11->10;
        # - 0->10  / 10->0 [not pooled, because 10->11 / 11->10 was pooled];
        # - 2->3   / 3->2;
        # - 3->8   / 8->3  [not pooled, because 2->3 / 3->2 was pooled];
        # - 6->7   / 7->6;
        # - 1->5   / 5->1;
        # - 7->11  / 11->7 [not pooled, because 10->11 / 11->10 and 6->7 / 7->6
        #                   were pooled];
        # - 1->2   / 2->1  [not pooled, because 2->3 / 3->2 and 1->5 / 5->1 were
        #                   pooled];
        # - 8->9   / 9->8;
        # - ...            [other edges that are not pooled]
        # (cf. file `../../common_data/simple_mesh_large_pool_2.png`)
        attention_threshold = 0.5

        edges_to_pool = [[8, 9], [1, 2], [7, 11], [1, 5], [6, 7], [3, 8],
                         [2, 3], [0, 10], [10, 11], [4, 13]]
        if (use_decreasing_attention_coefficient):
            primal_attention_coeffs = torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * attention_threshold
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = attention_threshold + (
                        1 - attention_threshold) * (
                            float(pooling_idx) / len(edges_to_pool) +
                            torch.rand([num_heads], dtype=torch.float) * 1. /
                            len(edges_to_pool))
        else:
            primal_attention_coeffs = attention_threshold + torch.rand(
                [num_primal_edges, num_heads],
                dtype=torch.float) * (1 - attention_threshold)
            for edge_idx, primal_edge in enumerate(
                    primal_graph_batch.edge_index.t().tolist()):
                if (sorted(primal_edge) in edges_to_pool):
                    pooling_idx = edges_to_pool.index(sorted(primal_edge))
                    primal_attention_coeffs[edge_idx] = (
                        attention_threshold - attention_threshold *
                        (float(pooling_idx) / len(edges_to_pool) +
                         torch.rand([num_heads], dtype=torch.float) * 1. /
                         len(edges_to_pool)))

        # Create a single dual-primal edge-pooling layer.
        pool = DualPrimalEdgePooling(
            self_loops_in_output_dual_graph=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (new_primal_graph_batch, new_dual_graph_batch, new_petdni_batch,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, 8)
        # - Check correspondence of the old primal nodes with the new primal
        #   nodes (i.e., node clusters).
        old_primal_node_to_new_one = pooling_log.old_primal_node_to_new_one
        for old_primal_node in range(num_primal_nodes):
            if (old_primal_node in [0]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 0)
            elif (old_primal_node in [1, 5]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 1)
            elif (old_primal_node in [2, 3]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 2)
            elif (old_primal_node in [4, 13]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 3)
            elif (old_primal_node in [6, 7]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 4)
            elif (old_primal_node in [8, 9]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 5)
            elif (old_primal_node in [10, 11]):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 6)
            elif (old_primal_node == 12):
                self.assertEqual(old_primal_node_to_new_one[old_primal_node], 7)
        # - Check that the features of each new primal node correspond to the
        #   average of the features of the primal nodes merged together into
        #   that node.
        for new_primal_node in range(num_new_primal_nodes):
            old_primal_nodes_per_new_primal_node = [
                0, [1, 5], [2, 3], [4, 13], [6, 7], [8, 9], [10, 11], 12
            ]
            old_primal_nodes = old_primal_nodes_per_new_primal_node[
                new_primal_node]
            self.assertAlmostEqual(
                new_primal_graph_batch.x[new_primal_node, 0].item(),
                primal_graph_batch.x[old_primal_nodes, 0].mean().item(), 5)
        # - Check the edges between the new primal nodes, which should be the
        #   following:
        #   - 0->1 / 1->0;
        #   - 0->4 / 4->0;
        #   - 0->6 / 6->0;
        #   - 1->2 / 2->1;
        #   - 1->3 / 3->1;
        #   - 1->4 / 4->1;
        #   - 2->3 / 3->2;
        #   - 2->5 / 5->2;
        #   - 3->5 / 5->3;
        #   - 3->7 / 7->3;
        #   - 4->6 / 6->4;
        #   - 4->7 / 7->4;
        #   - 5->6 / 6->5;
        #   - 6->7 / 7->6.
        self.assertEqual(num_new_primal_edges, 28)
        new_primal_edge_index_list = new_primal_graph_batch.edge_index.t(
        ).tolist()
        for new_primal_edge in [[0, 1], [0, 4], [0, 6], [1, 2], [1, 3], [1, 4],
                                [2, 3], [2, 5], [3, 5], [3, 7], [4, 6], [4, 7],
                                [5, 6], [6, 7]]:
            self.assertTrue(new_primal_edge in new_primal_edge_index_list)
            self.assertTrue(new_primal_edge[::-1] in new_primal_edge_index_list)
            # Check that opposite primal edges are associated to the same dual
            # node.
            self.assertNotEqual(new_petdni_batch[tuple(new_primal_edge)],
                                new_petdni_batch[tuple(new_primal_edge[::-1])])

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that in case the border between two new face clusters is made
        #   of multiple edges of the original mesh, the dual feature associated
        #   to the new primal edge is the average of the dual features
        #   associated with the 'multiple edges of the original mesh'. This
        #   happens between new primal nodes 2--5, in both directions.
        #   - New (directed) primal edge 2->5 corresponds to old (directed)
        #     primal edges 2->9 and 3->8.
        idx_new_dual_node = new_petdni_batch[(2, 5)]
        idx_old_dual_node_1 = petdni_batch[(2, 9)]
        idx_old_dual_node_2 = petdni_batch[(3, 8)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        #   - New (directed) primal edge 5->2 corresponds to old (directed)
        #     primal edges 9->2 and 8->3.
        idx_new_dual_node = new_petdni_batch[(5, 2)]
        idx_old_dual_node_1 = petdni_batch[(9, 2)]
        idx_old_dual_node_2 = petdni_batch[(8, 3)]
        self.assertAlmostEqual(
            new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
            dual_graph_batch.x[[idx_old_dual_node_1, idx_old_dual_node_2],
                               0].mean().item(), 5)
        # - For all other cases, check that the dual feature associated to the
        #   new primal edge is the dual feature associated with edge of the
        #   original mesh that is now between the new primal nodes.
        new_dual_nodes = [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4),
                          (2, 3), (3, 5), (3, 7), (4, 6), (4, 7), (5, 6),
                          (6, 7)]
        old_dual_nodes = [(0, 1), (0, 7), (0, 10), (1, 2), (5, 4), (5, 6),
                          (3, 4), (13, 8), (13, 12), (7, 11), (6, 12), (9, 10),
                          (11, 12)]

        for new_dual_node, old_dual_node in zip(new_dual_nodes, old_dual_nodes):
            # 'Forward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node]
            idx_old_dual_node = petdni_batch[old_dual_node]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)
            # 'Backward' edge.
            idx_new_dual_node = new_petdni_batch[new_dual_node[::-1]]
            idx_old_dual_node = petdni_batch[old_dual_node[::-1]]
            self.assertAlmostEqual(
                new_dual_graph_batch.x[idx_new_dual_node, 0].item(),
                dual_graph_batch.x[idx_old_dual_node, 0].item(), 5)

        # - Check that the mapping between old and new dual nodes is correct.
        old_dual_node_to_new_one = pooling_log.old_dual_node_to_new_one
        self.assertEqual(len(old_dual_node_to_new_one), num_dual_nodes)
        old_dual_nodes_index_with_corresponding_new_one = [
            petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_new_dual_nodes = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ]
        for dual_node_idx in range(num_dual_nodes):
            if (dual_node_idx in old_dual_nodes_index_with_corresponding_new_one
               ):
                # - The old dual node has a corresponding new dual node.
                self.assertEqual(
                    old_dual_node_to_new_one[dual_node_idx],
                    corresponding_new_dual_nodes[
                        old_dual_nodes_index_with_corresponding_new_one.index(
                            dual_node_idx)])
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(old_dual_node_to_new_one[dual_node_idx], -1)

        # - Check the edges between the new dual nodes, which should be the
        #   following (with dual nodes indicated by the corresponding primal
        #   nodes as a set), plus the self-loops:
        #   - (0->1) -> (1->2);
        #   - (0->1) -> (1->3);
        #   - (0->1) -> (1->4);
        #   - (1->0) -> (0->4);
        #   - (1->0) -> (0->6);
        #   - (0->4) -> (4->1);
        #   - (0->4) -> (4->6);
        #   - (0->4) -> (4->7);
        #   - (4->0) -> (0->1);
        #   - (4->0) -> (0->6);
        #   - (0->6) -> (6->4);
        #   - (0->6) -> (6->5);
        #   - (0->6) -> (6->7);
        #   - (6->0) -> (0->1);
        #   - (6->0) -> (0->4);
        #   - (1->2) -> (2->3);
        #   - (1->2) -> (2->5);
        #   - (2->1) -> (1->0);
        #   - (2->1) -> (1->3);
        #   - (2->1) -> (1->4);
        #   - (1->3) -> (3->2);
        #   - (1->3) -> (3->5);
        #   - (1->3) -> (3->7);
        #   - (3->1) -> (1->0);
        #   - (3->1) -> (1->2);
        #   - (3->1) -> (1->4);
        #   - (1->4) -> (4->0);
        #   - (1->4) -> (4->6);
        #   - (1->4) -> (4->7);
        #   - (4->1) -> (1->0);
        #   - (4->1) -> (1->2);
        #   - (4->1) -> (1->3);
        #   - (2->3) -> (3->1);
        #   - (2->3) -> (3->5);
        #   - (2->3) -> (3->7);
        #   - (3->2) -> (2->1);
        #   - (3->2) -> (2->5);
        #   - (2->5) -> (5->3);
        #   - (2->5) -> (5->6);
        #   - (5->2) -> (2->1);
        #   - (5->2) -> (2->3);
        #   - (3->5) -> (5->2);
        #   - (3->5) -> (5->6);
        #   - (5->3) -> (3->1);
        #   - (5->3) -> (3->2);
        #   - (5->3) -> (3->7);
        #   - (3->7) -> (7->4);
        #   - (3->7) -> (7->6);
        #   - (7->3) -> (3->1);
        #   - (7->3) -> (3->2);
        #   - (7->3) -> (3->5);
        #   - (4->6) -> (6->0);
        #   - (4->6) -> (6->5);
        #   - (4->6) -> (6->7);
        #   - (6->4) -> (4->0);
        #   - (6->4) -> (4->1);
        #   - (6->4) -> (4->7);
        #   - (4->7) -> (7->3);
        #   - (4->7) -> (7->6);
        #   - (7->4) -> (4->0);
        #   - (7->4) -> (4->1);
        #   - (7->4) -> (4->6);
        #   - (5->6) -> (6->0);
        #   - (5->6) -> (6->4);
        #   - (5->6) -> (6->7);
        #   - (6->5) -> (5->2);
        #   - (6->5) -> (5->3);
        #   - (6->7) -> (7->3);
        #   - (6->7) -> (7->4);
        #   - (7->6) -> (6->0);
        #   - (7->6) -> (6->4);
        #   - (7->6) -> (6->5).
        self.assertEqual(num_new_dual_edges, 72 + num_new_dual_nodes)
        new_dual_edge_index_list = new_dual_graph_batch.edge_index.t().tolist()

        dual_node_to_neighbors = {
            (0, 1): [(1, 2), (1, 3), (1, 4)],
            (1, 0): [(0, 4), (0, 6)],
            (0, 4): [(4, 1), (4, 6), (4, 7)],
            (4, 0): [(0, 1), (0, 6)],
            (0, 6): [(6, 4), (6, 5), (6, 7)],
            (6, 0): [(0, 1), (0, 4)],
            (1, 2): [(2, 3), (2, 5)],
            (2, 1): [(1, 0), (1, 3), (1, 4)],
            (1, 3): [(3, 2), (3, 5), (3, 7)],
            (3, 1): [(1, 0), (1, 2), (1, 4)],
            (1, 4): [(4, 0), (4, 6), (4, 7)],
            (4, 1): [(1, 0), (1, 2), (1, 3)],
            (2, 3): [(3, 1), (3, 5), (3, 7)],
            (3, 2): [(2, 1), (2, 5)],
            (2, 5): [(5, 3), (5, 6)],
            (5, 2): [(2, 1), (2, 3)],
            (3, 5): [(5, 2), (5, 6)],
            (5, 3): [(3, 1), (3, 2), (3, 7)],
            (3, 7): [(7, 4), (7, 6)],
            (7, 3): [(3, 1), (3, 2), (3, 5)],
            (4, 6): [(6, 0), (6, 5), (6, 7)],
            (6, 4): [(4, 0), (4, 1), (4, 7)],
            (4, 7): [(7, 3), (7, 6)],
            (7, 4): [(4, 0), (4, 1), (4, 6)],
            (5, 6): [(6, 0), (6, 4), (6, 7)],
            (6, 5): [(5, 2), (5, 3)],
            (6, 7): [(7, 3), (7, 4)],
            (7, 6): [(6, 0), (6, 4), (6, 5)]
        }
        for new_dual_node, other_dual_nodes in dual_node_to_neighbors.items():
            for other_dual_node in other_dual_nodes:
                self.assertTrue([
                    new_petdni_batch[new_dual_node],
                    new_petdni_batch[other_dual_node]
                ] in new_dual_edge_index_list)
            # Self-loop.
            self.assertTrue([
                new_petdni_batch[new_dual_node], new_petdni_batch[new_dual_node]
            ] in new_dual_edge_index_list)
