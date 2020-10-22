import numpy as np
import os.path as osp
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch
import unittest

from pd_mesh_net.nn import DualPrimalEdgePooling, DualPrimalEdgeUnpooling
from pd_mesh_net.utils import create_graphs, create_dual_primal_batch

current_dir = osp.dirname(__file__)


class TestDualEdgeUnpooling(unittest.TestCase):

    def test_large_simple_mesh_config_A(self):
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
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    # Test with number of primal edges to keep.
                    self.__test_large_simple_mesh_config_A(
                        num_primal_edges_to_keep=21 - 8,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with fraction of primal edges to keep. Pooling the
                    # top-8 out of the 21 primal-edge pairs corresponds to
                    # keeping a fraction of the primal edges around
                    # (21 - 8) / 21 = 0.6190...
                    # Since the pooling layer internally finds the number of
                    # primal edges to pool as
                    # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                    # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one
                    # needs to have:
                    #     8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                    # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                    # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                    # <=> 0.5714... < fraction_primal_edges_to_keep <=
                    #        0.6190...;
                    # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                    self.__test_large_simple_mesh_config_A(
                        fraction_primal_edges_to_keep=0.619,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with minimal attention coefficient.
                    self.__test_large_simple_mesh_config_A(
                        primal_att_coeff_threshold=0.5,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def test_large_simple_mesh_config_A_nonconsecutive(self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    self.__test_config_A_nonconsecutive(
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def test_large_simple_mesh_config_B(self):
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
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    # Test with number of primal edges to keep.
                    self.__test_large_simple_mesh_config_B(
                        num_primal_edges_to_keep=21 - 8,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with fraction of primal edges to keep. Pooling the
                    # top-8 out of the 21 primal-edge pairs corresponds to
                    # keeping a fraction of the primal edges around
                    # (21 - 8) / 21 = 0.6190...
                    # Since the pooling layer internally finds the number of
                    # primal edges to pool as
                    # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                    # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one
                    # needs to have:
                    #      8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                    # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                    # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                    # <=> 0.5714... < fraction_primal_edges_to_keep
                    #       <= 0.6190...;
                    # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                    self.__test_large_simple_mesh_config_B(
                        fraction_primal_edges_to_keep=0.619,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with minimal attention coefficient.
                    self.__test_large_simple_mesh_config_B(
                        primal_att_coeff_threshold=0.5,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def test_large_simple_mesh_config_B_nonconsecutive(self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    self.__test_config_B_nonconsecutive(
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def test_large_simple_mesh_config_C(self):
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
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    # Test with number of primal edges to keep.
                    self.__test_large_simple_mesh_config_C(
                        num_primal_edges_to_keep=21 - 8,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with fraction of primal edges to keep. Pooling the
                    # top-8 out of the 21 primal-edge pairs corresponds to
                    # keeping a fraction of the primal edges around
                    # (21 - 8) / 21 = 0.6190...
                    # Since the pooling layer internally finds the number of
                    # primal edges to pool as
                    # floor((1 - fraction_primal_edges_to_keep) * num_edges) =
                    # floor((1 - fraction_primal_edges_to_keep) * 21) = 8, one
                    # needs to have:
                    #     8 <= (1 - fraction_primal_edges_to_keep) * 21 < 9;
                    # <=> -13 <= -21* fraction_primal_edges_to_keep < -12;
                    # <=> 12 / 21 < fraction_primal_edges_to_keep <= 13/21;
                    # <=> 0.5714... < fraction_primal_edges_to_keep
                    #       <= 0.6190...;
                    # e.g., 0.5715 < fraction_primal_edges_to_keep < 0.6190.
                    self.__test_large_simple_mesh_config_C(
                        fraction_primal_edges_to_keep=0.619,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)
                    # Test with minimal attention coefficient.
                    self.__test_large_simple_mesh_config_C(
                        primal_att_coeff_threshold=0.5,
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def test_large_simple_mesh_config_C_nonconsecutive(self):
        # Repeat the experiment by considering once pooling based on decreasing
        # attention coefficients and in the other pooling based on increasing
        # attention coefficient (cf.
        # `pd_mesh_net.nn.pool.DualPrimalEdgePooling`).
        for use_decreasing_attention_coefficient in [True, False]:
            # Test both with and without output self-loops in the dual graph.
            for self_loops_in_output_dual_graph in [True, False]:
                # Test also with more than one attention head.
                for num_heads in range(1, 4):
                    self.__test_config_C_nonconsecutive(
                        use_decreasing_attention_coefficient=
                        use_decreasing_attention_coefficient,
                        num_heads=num_heads,
                        self_loops_in_output_dual_graph=
                        self_loops_in_output_dual_graph)

    def __test_large_simple_mesh_config_A(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0, 6, 7, 10, 11],
                                                              [1, 5], [4, 13],
                                                              [2, 3, 8], [9],
                                                              [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])

        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (2, 3), (1, 2), (
                0, 1), (0, 5), (3, 4), (2, 3), (0, 4), (0, 5), (2, 5)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))

    def __test_large_simple_mesh_config_B(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0, 6, 7, 10, 11],
                                                              [1, 5], [4, 13],
                                                              [2, 3, 8], [9],
                                                              [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])
        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ] + [
            petdni_batch_after_pooling[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))

    def __test_large_simple_mesh_config_C(
        self,
        num_primal_edges_to_keep=None,
        fraction_primal_edges_to_keep=None,
        primal_att_coeff_threshold=None,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=num_primal_edges_to_keep,
            fraction_primal_edges_to_keep=fraction_primal_edges_to_keep,
            primal_att_coeff_threshold=primal_att_coeff_threshold,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0, 6, 7, 10, 11],
                                                              [1, 5], [4, 13],
                                                              [2, 3, 8], [9],
                                                              [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])
        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)
        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 2), (2, 9), (3, 4), (4, 5), (
                5, 6), (6, 12), (8, 9), (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ] + [
            petdni_batch_after_pooling[primal_edge[::-1]]
            for primal_edge in [(0, 1), (1, 3), (3, 4), (3, 2), (2, 1), (1, 0),
                                (0, 5), (3, 4), (3, 2), (4, 0), (0, 5), (5, 2)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))

    # * Allow only non-consecutive edges.
    def __test_config_A_nonconsecutive(
        self,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0], [1, 5],
                                                              [2, 3], [4, 13],
                                                              [6, 7], [8, 9],
                                                              [10, 11], [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])

        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges // 2)
        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
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
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (
                2, 3), (2, 5), (1, 3), (1, 4), (4, 7), (4,
                                                        6), (3,
                                                             5), (5,
                                                                  6), (6,
                                                                       7), (3,
                                                                            7)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))

    def __test_config_B_nonconsecutive(
        self,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0], [1, 5],
                                                              [2, 3], [4, 13],
                                                              [6, 7], [8, 9],
                                                              [10, 11], [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])
        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)

        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ] + [
            petdni_batch_after_pooling[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))

    def __test_config_C_nonconsecutive(
        self,
        use_decreasing_attention_coefficient=True,
        num_heads=1,
        self_loops_in_output_dual_graph=False):
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
        #  - Set the features of each primal node randomly.
        dim_primal_features = primal_graph_batch.num_node_features
        for primal_feature in primal_graph_batch.x:
            primal_feature[:] = torch.rand(dim_primal_features,
                                           dtype=torch.float)

        # Dual graph.
        num_dual_edges = dual_graph_batch.num_edges
        num_dual_nodes = maybe_num_nodes(dual_graph_batch.edge_index)
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
            self_loops_in_output_dual_graph=self_loops_in_output_dual_graph,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            num_primal_edges_to_keep=15,
            use_decreasing_attention_coefficient=
            use_decreasing_attention_coefficient,
            allow_pooling_consecutive_edges=False,
            return_old_dual_node_to_new_dual_node=True)
        # Perform primal-edge pooling.
        (primal_graph_batch_after_pooling, dual_graph_batch_after_pooling,
         petdni_batch_after_pooling,
         pooling_log) = pool(primal_graph_batch=primal_graph_batch,
                             dual_graph_batch=dual_graph_batch,
                             primal_edge_to_dual_node_idx_batch=petdni_batch,
                             primal_attention_coeffs=primal_attention_coeffs)
        num_primal_nodes_after_pooling = maybe_num_nodes(
            primal_graph_batch_after_pooling.edge_index)
        # Create a single dual-primal edge-unpooling layer.
        unpool = DualPrimalEdgeUnpooling(out_channels_dual=dim_dual_features)
        # Perform unpooling.
        (new_primal_graph_batch, new_dual_graph_batch,
         new_petdni_batch) = unpool(
             primal_graph_batch=primal_graph_batch_after_pooling,
             dual_graph_batch=dual_graph_batch_after_pooling,
             pooling_log=pooling_log)
        # Tests on the new primal graph.
        num_new_primal_nodes = maybe_num_nodes(
            new_primal_graph_batch.edge_index)
        num_new_primal_edges = new_primal_graph_batch.num_edges
        self.assertEqual(num_new_primal_nodes, num_primal_nodes)
        # - Check that the features of each new primal node correspond to the
        #   features of the primal nodes in which they were merged before
        #   pooling.
        for primal_node_after_pooling in range(num_primal_nodes_after_pooling):
            new_primal_nodes_per_primal_node_after_pooling = [[0], [1, 5],
                                                              [2, 3], [4, 13],
                                                              [6, 7], [8, 9],
                                                              [10, 11], [12]]
            for primal_node_after_pooling, new_primal_nodes in enumerate(
                    new_primal_nodes_per_primal_node_after_pooling):
                for new_primal_node in new_primal_nodes:
                    self.assertEqual(
                        new_primal_graph_batch.x[new_primal_node],
                        primal_graph_batch_after_pooling.
                        x[primal_node_after_pooling])
        # - Check the edges between the new primal nodes, which should be the
        #   same as the original graph.
        self.assertEqual(num_new_primal_edges, num_primal_edges)
        self.assertTrue(
            torch.equal(new_primal_graph_batch.edge_index,
                        primal_graph_batch.edge_index))

        # - Check that the primal-edge-to-new-node-index dictionary is the same
        #   as the original one.
        self.assertEqual(petdni_batch, new_petdni_batch)

        # Tests on the new dual graph.
        num_new_dual_nodes = maybe_num_nodes(new_dual_graph_batch.edge_index)
        num_new_dual_edges = new_dual_graph_batch.num_edges
        self.assertEqual(num_new_dual_nodes, num_new_primal_edges)

        # - Check that the mapping between old and new dual nodes is correct.
        new_dual_node_to_dual_node_after_pooling = (
            pooling_log.old_dual_node_to_new_one)
        self.assertEqual(len(new_dual_node_to_dual_node_after_pooling),
                         num_dual_nodes)
        new_dual_nodes_with_corresponding_one_after_pooling = [
            new_petdni_batch[primal_edge]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ] + [
            new_petdni_batch[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 7), (0, 10), (1, 2), (2, 9), (3, 4),
                                (3, 8), (4, 5), (5, 6), (6, 12), (7, 11),
                                (8, 13), (9, 10), (11, 12), (12, 13)]
        ]
        corresponding_dual_nodes_after_pooling = [
            petdni_batch_after_pooling[primal_edge]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ] + [
            petdni_batch_after_pooling[primal_edge[::-1]]
            for primal_edge in [(0, 1), (0, 4), (0, 6), (1, 2), (2, 5), (2, 3),
                                (2, 5), (3, 1), (1, 4), (4, 7), (4, 6), (5, 3),
                                (5, 6), (6, 7), (7, 3)]
        ]
        # This will contain the dual feature common to all the new dual nodes
        # that do not have a corresponding dual node after pooling.
        common_dual_feature = None
        for dual_node_idx in range(num_new_dual_nodes):
            if (dual_node_idx in
                    new_dual_nodes_with_corresponding_one_after_pooling):
                # - The old dual node has a corresponding new dual node.
                corresponding_node_after_pooling = (
                    corresponding_dual_nodes_after_pooling[(
                        new_dual_nodes_with_corresponding_one_after_pooling.
                        index(dual_node_idx))])
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx],
                    corresponding_node_after_pooling)
                # - Check that the two features match.
                self.assertTrue(
                    torch.equal(
                        new_dual_graph_batch.x[dual_node_idx],
                        dual_graph_batch_after_pooling.
                        x[corresponding_node_after_pooling]))
            else:
                # - The old dual node has no corresponding new dual node.
                self.assertEqual(
                    new_dual_node_to_dual_node_after_pooling[dual_node_idx], -1)
                if (common_dual_feature is None):
                    common_dual_feature = new_dual_graph_batch.x[dual_node_idx]
                else:
                    self.assertTrue(
                        torch.equal(new_dual_graph_batch.x[dual_node_idx],
                                    common_dual_feature))

        # - Check the edges between the new dual nodes, which should be the same
        #   as the original graph.
        self.assertEqual(num_new_dual_edges, num_dual_edges)
        self.assertTrue(
            torch.equal(new_dual_graph_batch.edge_index,
                        dual_graph_batch.edge_index))