import numpy as np
import os.path as osp
import torch
import unittest

from pd_mesh_net.nn import DualPrimalConv, DualPrimalResConv
from pd_mesh_net.utils import create_graphs

current_dir = osp.dirname(__file__)


class TestDualPrimalConv(unittest.TestCase):

    def test_simple_mesh_config_A_features_not_from_dual_with_self_loops(self):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 7
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=True)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last four features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 3:7] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios both equal to 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i, j} of the dual node {i, j}
        #   becomes \tilde{f}_{i, j} = f_{i, j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 2 / sqrt(3), 0, 0, 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 4 / sqrt(3), pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #      pi + 4 / sqrt(3), pi + 4 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge {j, n}->{i, j} and {i, m}->{i, j} - with m neighbor of i and n
        #   neighbor of j - the quantities (|| indicates concatenation):
        #   * \tilde{\beta}_{{j, n}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{j, n} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3).
        #   * \tilde{\beta}_{{i, m}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, m} || \tilde{f}_{i, j}) =
        #     = pi + 4 / sqrt(3), by the symmetry of the dual features.
        #   NOTE: We also have self-loops in the dual graph! Hence, one has:
        #   - \tilde{\beta}_{{i, j}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, j} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #   = pi + 4 / sqrt(3), for all i, j.
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 4 / sqrt(3) > 0). Then, compute the softmax over the {j, n}'s
        #   and {i, m}'s neighboring nodes of {i, j}, including also {i, j}.
        #   Since all \tilde{\beta}_{i, j}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{{j, n}, {i, j}} and
        #   \tilde{\alpha}_{{i, m}, {i, j}} are simply
        #   1 / #(neighboring nodes {j, n} + neighboring nodes {i, m} + 1 (for
        #   self-loop)). Therefore:
        #   - \tilde{\alpha}_{{0, m}, {0, 1}} = 1 / 4 for m in {1, 7};
        #   - \tilde{\alpha}_{{1, n}, {0, 1}} = 1 / 4 for m in {2, 5};
        #   - \tilde{\alpha}_{{0, m}, {0, 7}} = 1 / 3 for m in {1, 7};
        #   - \tilde{\alpha}_{{7, n}, {0, 7}} = 1 / 3 for m in {6};
        #   - \tilde{\alpha}_{{1, m}, {1, 2}} = 1 / 4 for m in {0, 2, 5};
        #   - \tilde{\alpha}_{{2, n}, {1, 2}} = 1 / 4 for m in {3};
        #   - \tilde{\alpha}_{{1, m}, {1, 5}} = 1 / 5 for m in {0, 2, 5};
        #   - \tilde{\alpha}_{{5, n}, {1, 5}} = 1 / 5 for m in {4, 6};
        #   - \tilde{\alpha}_{{6, m}, {6, 7}} = 1 / 3 for m in {5, 7};
        #   - \tilde{\alpha}_{{7, n}, {6, 7}} = 1 / 3 for m in {0};
        #   - \tilde{\alpha}_{{2, m}, {2, 3}} = 1 / 3 for m in {1, 3};
        #   - \tilde{\alpha}_{{3, n}, {2, 3}} = 1 / 3 for m in {4};
        #   - \tilde{\alpha}_{{4, m}, {4, 5}} = 1 / 4 for m in {3, 5};
        #   - \tilde{\alpha}_{{5, n}, {4, 5}} = 1 / 4 for m in {1, 6};
        #   - \tilde{\alpha}_{{5, m}, {5, 6}} = 1 / 4 for m in {1, 4, 6};
        #   - \tilde{\alpha}_{{6, n}, {5, 6}} = 1 / 4 for m in {7};
        #   - \tilde{\alpha}_{{3, m}, {3, 4}} = 1 / 3 for m in {2, 4};
        #   - \tilde{\alpha}_{{4, n}, {3, 4}} = 1 / 3 for m in {5}.
        #
        #  - The output features are then obtained as f_{i, j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{{i, m}, {i, j}} * \tilde{f}_{i, m}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}} * \tilde{f}_{j, n})
        #           \tilde{\alpha}_{{i, j}, {i, j}} * \tilde{f}_{i, j}).
        #    = ReLU(\tilde{f}_{i, j} * sum_m(\tilde{\alpha}_{{i, m}, {i, j}}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}} +
        #           \tilde{\alpha}_{{i, j}, {i, j}}) =
        #    = ReLU(\tilde{f}_{i, j}) =
        #    = \tilde{f}_{i, j},
        #    where the third-last equality holds since the \tilde{f}_{i, m} =
        #    \tilde{f}_{j, n} = \tilde{f}_{i, j} for all valid i, j, m, n (cf.
        #    above), the second-last holds since the sum all the attention
        #    coefficients over the neighborhood of each dual node (dual node
        #    included) is 1 by construction, and the last one holds because
        #    \tilde{f}_{i, j} > 0 for all valid i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i, j} = att_{primal}^T * f_{i, j}^' =
        #   = (1 / 5 * (pi + 4 / sqrt(3))) + (1 / 5 * (pi + 4 / sqrt(3))) +
        #     (1 / 5 * (pi + 4 / sqrt(3))) + (1 / 5 * (pi + 4 / sqrt(3))) +
        #     (1 / 5 * (pi + 4 / sqrt(3))) =
        #   = pi + 4 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 4 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node {i, j}.
        #   Since all \beta_{i, j}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{i, j} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{m, 0} = 1 / 2     for m in {1, 7};
        #   - \alpha_{m, 1} = 1 / 3     for m in {0, 2, 5};
        #   - \alpha_{m, 2} = 1 / 2     for m in {1, 3};
        #   - \alpha_{m, 3} = 1 / 2     for m in {2, 4};
        #   - \alpha_{m, 4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{m, 5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{m, 6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{m, 7} = 1 / 2     for m in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j, i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j, i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 4 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes {6, 7}, {1, 5} and {3, 4}.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        assert (dual_idx_67 == graph_creator.primal_edge_to_dual_node_idx[(7,
                                                                           6)])
        dual_idx_15 = graph_creator.primal_edge_to_dual_node_idx[(1, 5)]
        assert (dual_idx_15 == graph_creator.primal_edge_to_dual_node_idx[(5,
                                                                           1)])
        dual_idx_34 = graph_creator.primal_edge_to_dual_node_idx[(3, 4)]
        assert (dual_idx_34 == graph_creator.primal_edge_to_dual_node_idx[(4,
                                                                           3)])
        dual_graph.x[dual_idx_67, :] = torch.Tensor(
            [np.pi / 2, 1., 1., 0., 0., 0., 0.])
        dual_graph.x[dual_idx_15, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0.5, 0., 0., 0., 0.])
        dual_graph.x[dual_idx_34, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0.25, 0., 0., 0., 0.])
        # - As previously, to simplify computations, we manually set the last
        #   four features of each dual node (i.e., the edge-to-previous-edge-
        #   and edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 3:7] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node {i, j}           Nodes {j, n}/{i, m} with edges
        #                         {j, n}->{i, j} or {i, m}->{i, j}
        #   _____________________________________________________
        #       {0, 1}            {0, 7}, {1, 2}, {1, 5}
        #       {0, 7}            {0, 1}, {6, 7}
        #       {1, 2}            {0, 1}, {1, 5}, {2, 3}
        #       {1, 5}            {0, 1}, {1, 2}, {4, 5}, {5, 6}
        #       {2, 3}            {1, 2}, {3, 4}
        #       {3, 4}            {2, 3}, {4, 5}
        #       {4, 5}            {1, 5}, {3, 4}, {5, 6}
        #       {5, 6}            {1, 5}, {4, 5}, {6, 7}
        #       {6, 7}            {5, 6}, {0, 7}
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i, j} of the dual node {i, j}
        #   becomes \tilde{f}_{i, j} = f_{i, j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 2 / sqrt(3), 0, 0, 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 4 / sqrt(3), pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #      pi + 4 / sqrt(3), pi + 4 / sqrt(3)], for all {i, j} not in
        #   {{6, 7}, {1, 5}, {3, 4}}.
        #   We have also:
        #   - \tilde{f}_{6, 7} = [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #       pi / 2 + 2, pi / 2 + 2];
        #   - \tilde{f}_{1, 5} = [3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #       3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1];
        #   - \tilde{f}_{3, 4} = [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #       5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   edge {j, n}->{i, j} or {i, m}->{i, j} the quantities (|| indicates
        #   concatenation):
        #   * \tilde{\beta}_{{j, n}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{j, n} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all {j, n}, {i, j} not in {{6, 7}, {1, 5},
        #     {3, 4}}.
        #   * \tilde{\beta}_{{i, m}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, m} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all {i, m}, {i, j} not in {{6, 7}, {1, 5},
        #     {3, 4}}.
        #   Likewise, we have also:
        #   - \tilde{beta}_{{5, 6}, {6, 7}} =
        #     = \tilde{beta}_{{0, 7}, {6, 7}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (pi / 2 + 2) =
        #     = 3 * pi / 4 + 2 / sqrt(3) + 1;
        #   - \tilde{beta}_{{6, 7}, {5, 6}} =
        #     = \tilde{beta}_{{6, 7}, {0, 7}} =
        #     = (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) +
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (pi / 2 + 2) =
        #     = 3 * pi / 4 + 2 / sqrt(3) + 1;
        #   - \tilde{beta}_{{0, 1}, {1, 5}} =
        #     = \tilde{beta}_{{1, 2}, {1, 5}} =
        #     = \tilde{beta}_{{4, 5}, {1, 5}} =
        #     = \tilde{beta}_{{5, 6}, {1, 5}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 1) =
        #     = 7 * pi / 8 + 2 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{{1, 5}, {0, 1}} =
        #     = \tilde{beta}_{{1, 5}, {1, 2}} =
        #     = \tilde{beta}_{{1, 5}, {4, 5}} =
        #     = \tilde{beta}_{{1, 5}, {5, 6}} =
        #     = (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 1)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 1) =
        #     = 7 * pi / 8 + 2 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{{2, 3}, {3, 4}} =
        #     = \tilde{beta}_{{4, 5}, {3, 4}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.5) =
        #     = 9 * pi / 8 + 2 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{{3, 4}, {2, 3}} =
        #     = \tilde{beta}_{{3, 4}, {4, 5}} =
        #     = (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.5) =
        #     = 9 * pi / 8 + 2 / sqrt(3) + 1 / 4;
        #   NOTE: We also have self-loops in the dual graph! Hence, one has:
        #   - \tilde{\beta}_{{i, j}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, j} || \tilde{f}_{i, j}) =
        #   = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #     (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #     (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #     (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #     (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #   = pi + 4 / sqrt(3), for all {i, j} not in {{6, 7}, {1, 5}, {3, 4}}.
        #   - \tilde{beta}_{{6, 7}, {6, 7}} =
        #     = (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) =
        #     = pi / 2 + 2;
        #   - \tilde{beta}_{{1, 5}, {1, 5}} =
        #     = (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) =
        #     = 3 * pi / 4 + 1;
        #   - \tilde{beta}_{{3, 4}, {3, 4}} =
        #     = (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) =
        #     = 5 * pi / 4 + 0.5.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{{j, n}, {i, j}} > 0, \tilde{beta}_{{i, m}, {i, j}} > 0
        #   and \tilde{beta}_{{i, j}, {i, j}} > 0 for all i, j, m, n). Then,
        #   compute the softmax over the neighboring nodes {j, n} and {i, m}. We
        #   have (cf. adjacency matrix + self-loops):
        #   - \tilde{\alpha}_{{0, 1}, {0, 1}} =
        #     = \tilde{\alpha}_{{0, 7}, {0, 1}} =
        #     = \tilde{\alpha}_{{1, 2}, {0, 1}} =
        #     = exp(\tilde{beta}_{{0, 1}, {0, 1}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 1}}) +
        #        exp(\tilde{beta}_{{0, 7}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 2}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 5}, {0, 1}}))
        #     = exp(pi + 4 / sqrt(3)) / (3 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.2984318
        #   - \tilde{\alpha}_{{1, 5}, {0, 1}} =
        #     = exp(\tilde{beta}_{{1, 5}, {0, 1}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 1}}) +
        #        exp(\tilde{beta}_{{0, 7}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 2}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 5}, {0, 1}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (3 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1047045
        #   - \tilde{\alpha}_{{0, 1}, {0, 7}} =
        #     = \tilde{\alpha}_{{0, 7}, {0, 7}} =
        #     = exp(\tilde{beta}_{{0, 1}, {0, 7}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 7}}) +
        #        exp(\tilde{beta}_{{0, 7}, {0, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {0, 7}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.4183069
        #   - \tilde{\alpha}_{{6, 7}, {0, 7}} =
        #     = exp(\tilde{beta}_{{6, 1}, {0, 7}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 7}}) +
        #        exp(\tilde{beta}_{{0, 7}, {0, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {0, 7}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.1633862
        #   - \tilde{\alpha}_{{0, 1}, {1, 2}} =
        #     = \tilde{\alpha}_{{1, 2}, {1, 2}} =
        #     = \tilde{\alpha}_{{2, 3}, {1, 2}} =
        #     = exp(\tilde{beta}_{{0, 1}, {1, 2}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 2}, {1, 2}}) +
        #        exp(\tilde{beta}_{{2, 3}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 2}}))
        #     = exp(pi + 4 / sqrt(3)) / (3 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.2984318
        #   - \tilde{\alpha}_{{1, 5}, {1, 2}} =
        #     = exp(\tilde{beta}_{{1, 5}, {1, 2}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 2}, {1, 2}}) +
        #        exp(\tilde{beta}_{{2, 3}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 2}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (3 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1047045
        #   - \tilde{\alpha}_{{0, 1}, {1, 5}} =
        #     = \tilde{\alpha}_{{1, 2}, {1, 5}} =
        #     = \tilde{\alpha}_{{4, 5}, {1, 5}} =
        #     = \tilde{\alpha}_{{5, 6}, {1, 5}} =
        #     = exp(\tilde{beta}_{{0, 1}, {1, 5}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 5}}) +
        #        exp(\tilde{beta}_{{1, 2}, {1, 5}}) +
        #        exp(\tilde{beta}_{{4, 5}, {1, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {1, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 5}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (4 * exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 1)) =
        #     ~= 0.2298402
        #   - \tilde{\alpha}_{{1, 5}, {1, 5}} =
        #     = exp(\tilde{beta}_{{1, 5}, {1, 5}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 5}}) +
        #        exp(\tilde{beta}_{{1, 2}, {1, 5}}) +
        #        exp(\tilde{beta}_{{4, 5}, {1, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {1, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 5}}))
        #     = exp(3 * pi / 4 + 1) /
        #       (4 * exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 1)) =
        #     ~= 0.0806392
        #   - \tilde{\alpha}_{{1, 2}, {2, 3}} =
        #     = \tilde{\alpha}_{{2, 3}, {2, 3}} =
        #     = exp(\tilde{beta}_{{1, 2}, {2, 3}}) /
        #       (exp(\tilde{beta}_{{1, 2}, {2, 3}}) +
        #        exp(\tilde{beta}_{{2, 3}, {2, 3}}) +
        #        exp(\tilde{beta}_{{3, 4}, {2, 3}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4)) =
        #     ~= 0.3847197
        #   - \tilde{\alpha}_{{3, 4}, {2, 3}} =
        #     = exp(\tilde{beta}_{{3, 4}, {2, 3}}) /
        #       (exp(\tilde{beta}_{{1, 2}, {2, 3}}) +
        #        exp(\tilde{beta}_{{2, 3}, {2, 3}}) +
        #        exp(\tilde{beta}_{{3, 4}, {2, 3}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4)) =
        #     ~= 0.2305606
        #   - \tilde{\alpha}_{{2, 3}, {3, 4}} =
        #     = \tilde{\alpha}_{{4, 5}, {3, 4}} =
        #     = exp(\tilde{beta}_{{2, 3}, {3, 4}}) /
        #       (exp(\tilde{beta}_{{2, 3}, {3, 4}}) +
        #        exp(\tilde{beta}_{{4, 5}, {3, 4}}) +
        #        exp(\tilde{beta}_{{3, 4}, {3, 4}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (2 * exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(5 * pi / 4 + 0.5)) =
        #     ~= 0.3847197
        #   - \tilde{\alpha}_{{3, 4}, {3, 4}} =
        #     = exp(\tilde{beta}_{{3, 4}, {3, 4}}) /
        #       (exp(\tilde{beta}_{{2, 3}, {3, 4}}) +
        #        exp(\tilde{beta}_{{4, 5}, {3, 4}}) +
        #        exp(\tilde{beta}_{{3, 4}, {3, 4}}))
        #     = exp(5 * pi / 4 + 0.5) /
        #       (2 * exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(5 * pi / 4 + 0.5)) =
        #     ~= 0.2305606
        #   - \tilde{\alpha}_{{4, 5}, {4, 5}} =
        #     = \tilde{\alpha}_{{5, 6}, {4, 5}} =
        #     = exp(\tilde{beta}_{{4, 5}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {4, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.3389665
        #   - \tilde{\alpha}_{{3, 4}, {4, 5}} =
        #     = exp(\tilde{beta}_{{3, 4}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {4, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.2031409
        #   - \tilde{\alpha}_{{1, 5}, {4, 5}} =
        #     = exp(\tilde{beta}_{{1, 5}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {4, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1189260
        #   - \tilde{\alpha}_{{4, 5}, {5, 6}} =
        #     = \tilde{\alpha}_{{5, 6}, {5, 6}} =
        #     = exp(\tilde{beta}_{{4, 5}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{5, 6}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.3647720
        #   - \tilde{\alpha}_{{1, 5}, {5, 6}} =
        #     = exp(\tilde{beta}_{{1, 5}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{5, 6}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.1279799
        #   - \tilde{\alpha}_{{6, 7}, {5, 6}} =
        #     = exp(\tilde{beta}_{{6, 7}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{5, 6}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.1424760
        #   - \tilde{\alpha}_{{0, 7}, {6, 7}} =
        #     = \tilde{\alpha}_{{5, 6}, {6, 7}} =
        #     = exp(\tilde{beta}_{{0, 7}, {6, 7}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {6, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {6, 7}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) /
        #       (2 * exp(3 * pi / 4 + 2 / sqrt(3) + 1) +
        #        exp(pi / 2 + 2)) =
        #     ~= 0.4183069
        #   - \tilde{\alpha}_{{6, 7}, {6, 7}} =
        #     = exp(\tilde{beta}_{{6, 7}, {6, 7}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {6, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {6, 7}}))
        #     = exp(pi / 2 + 2) /
        #       (2 * exp(3 * pi / 4 + 2 / sqrt(3) + 1) +
        #        exp(pi / 2 + 2)) =
        #     ~= 0.1633862

        # - The output features are then obtained as f_{i, j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{{i, m}, {i, j}} * \tilde{f}_{i, m}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}} * \tilde{f}_{j, n}) +
        #           \tilde{\alpha}_{{i, j}, {i, j}} * \tilde{f}_{i, j}).
        #    We thus have:
        #    - f_{0, 1}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {0, 1}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{0, 7}, {0, 1}} * \tilde{f}_{0, 7} +
        #                   \tilde{\alpha}_{{1, 2}, {0, 1}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{1, 5}, {0, 1}} * \tilde{f}_{1, 5})
        #                 = ReLU((0.2984318 * 3) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1047045 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1]) =
        #                ~= [5.231658, 5.231658, 5.231658, 5.231658, 5.231658];
        #    - f_{0, 7}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {0, 7}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{0, 7}, {0, 7}} * \tilde{f}_{0, 7} +
        #                   \tilde{\alpha}_{{6, 7}, {0, 7}} * \tilde{f}_{6, 7})
        #                 = ReLU((0.4183069 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1633862 *
        #                   [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #                    pi / 2 + 2]) =
        #                ~= [5.143795, 5.143795, 5.143795, 5.143795, 5.143795];
        #    - f_{1, 2}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {1, 2}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{1, 2}, {1, 2}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{1, 5}, {1, 2}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{2, 3}, {1, 2}} * \tilde{f}_{2, 3})
        #                 = ReLU((0.2984318 * 3) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1047045 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1]) =
        #                ~= [5.231658, 5.231658, 5.231658, 5.231658, 5.231658];
        #    - f_{1, 5}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {1, 5}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{1, 2}, {1, 5}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{1, 5}, {1, 5}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{4, 5}, {1, 5}} * \tilde{f}_{4, 5} +
        #                   \tilde{\alpha}_{{5, 6}, {1, 5}} * \tilde{f}_{5, 6})
        #                 = ReLU((0.2298402 * 4) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.0806392 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1]) =
        #                ~= [5.282071, 5.282071, 5.282071, 5.282071, 5.282071];
        #    - f_{2, 3}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 2}, {2, 3}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{2, 3}, {2, 3}} * \tilde{f}_{2, 3} +
        #                   \tilde{\alpha}_{{3, 4}, {2, 3}} * \tilde{f}_{3, 4})
        #                 = ReLU((0.3847197 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.2305606 *
        #                   [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5]) =
        #                ~= [5.214899, 5.214899, 5.214899, 5.214899, 5.214899];
        #    - f_{3, 4}^' = ReLU(
        #                   \tilde{\alpha}_{{2, 3}, {3, 4}} * \tilde{f}_{2, 3} +
        #                   \tilde{\alpha}_{{3, 4}, {3, 4}} * \tilde{f}_{3, 4} +
        #                   \tilde{\alpha}_{{4, 5}, {3, 4}} * \tilde{f}_{4, 5})
        #                 = ReLU((0.3847197 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.2305606 *
        #                   [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5]) =
        #                ~= [5.214899, 5.214899, 5.214899, 5.214899, 5.214899];
        #    - f_{4, 5}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 5}, {4, 5}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{3, 4}, {4, 5}} * \tilde{f}_{3, 4} +
        #                   \tilde{\alpha}_{{4, 5}, {4, 5}} * \tilde{f}_{4, 5} +
        #                   \tilde{\alpha}_{{5, 6}, {4, 5}} * \tilde{f}_{5, 6})
        #                 = ReLU((0.3389665 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.2031409 *
        #                   [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5] + 0.1189260 * [3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1]) =
        #                ~= [4.993850, 4.993850, 4.993850, 4.993850, 4.993850];
        #    - f_{5, 6}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 5}, {5, 6}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{4, 5}, {5, 6}} * \tilde{f}_{4, 5} +
        #                   \tilde{\alpha}_{{5, 6}, {5, 6}} * \tilde{f}_{5, 6} +
        #                   \tilde{\alpha}_{{6, 7}, {5, 6}} * \tilde{f}_{6, 7})
        #                 = ReLU((0.3647720 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1279799 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1] + 0.1424760 *
        #                   [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #                    pi / 2 + 2]) =
        #                ~= [4.915018, 4.915018, 4.915018, 4.915018, 4.915018];
        #    - f_{6, 7}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 7}, {6, 7}} * \tilde{f}_{0, 7} +
        #                   \tilde{\alpha}_{{5, 6}, {6, 7}} * \tilde{f}_{5, 6} +
        #                   \tilde{\alpha}_{{6, 7}, {6, 7}} * \tilde{f}_{6, 7})
        #                 = ReLU((0.4183069 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1633862 *
        #                   [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #                    pi / 2 + 2]) =
        #                ~= [5.143795, 5.143795, 5.143795, 5.143795, 5.143795].
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i, j} = att_{primal}^T * f_{i, j}^'. We thus have:
        #
        #   - \beta_{0, 1} = att_{primal}^T * f_{0, 1}^' =
        #                  = (1 / 5 * 5.231658) + (1 / 5 * 5.231658) +
        #                    (1 / 5 * 5.231658) + (1 / 5 * 5.231658) +
        #                    (1 / 5 * 5.231658) =
        #                  = 5.231658;
        #   - \beta_{0, 7} = att_{primal}^T * f_{0, 7}^' =
        #                  = (1 / 5 * 5.143795) + (1 / 5 * 5.143795) +
        #                    (1 / 5 * 5.143795) + (1 / 5 * 5.143795) +
        #                    (1 / 5 * 5.143795) =
        #                  = 5.143795;
        #   - \beta_{1, 2} = att_{primal}^T * f_{1, 2}^' =
        #                  = (1 / 5 * 5.231658) + (1 / 5 * 5.231658) +
        #                    (1 / 5 * 5.231658) + (1 / 5 * 5.231658) +
        #                    (1 / 5 * 5.231658) =
        #                  = 5.231658;
        #   - \beta_{1, 5} = att_{primal}^T * f_{1, 5}^' =
        #                  = (1 / 5 * 5.282071) + (1 / 5 * 5.282071) +
        #                    (1 / 5 * 5.282071) + (1 / 5 * 5.282071) +
        #                    (1 / 5 * 5.282071) =
        #                  = 5.282071;
        #   - \beta_{2, 3} = att_{primal}^T * f_{2, 3}^' =
        #                  = (1 / 5 * 5.214899) + (1 / 5 * 5.214899) +
        #                    (1 / 5 * 5.214899) + (1 / 5 * 5.214899) +
        #                    (1 / 5 * 5.214899) =
        #                  = 5.214899;
        #   - \beta_{3, 4} = att_{primal}^T * f_{3, 4}^' =
        #                  = (1 / 5 * 5.214899) + (1 / 5 * 5.214899) +
        #                    (1 / 5 * 5.214899) + (1 / 5 * 5.214899) +
        #                    (1 / 5 * 5.214899) =
        #                  = 5.214899;
        #   - \beta_{4, 5} = att_{primal}^T * f_{4, 5}^' =
        #                  = (1 / 5 * 4.993850) + (1 / 5 * 4.993850) +
        #                    (1 / 5 * 4.993850) + (1 / 5 * 4.993850) +
        #                    (1 / 5 * 4.993850) =
        #                  = 4.993850;
        #   - \beta_{5, 6} = att_{primal}^T * f_{5, 6}^' =
        #                  = (1 / 5 * 4.915018) + (1 / 5 * 4.915018) +
        #                    (1 / 5 * 4.915018) + (1 / 5 * 4.915018) +
        #                    (1 / 5 * 4.915018) =
        #                  = 4.915018;
        #   - \beta_{6, 7} = att_{primal}^T * f_{6, 7}^' =
        #                  = (1 / 5 * 5.143795) + (1 / 5 * 5.143795) +
        #                    (1 / 5 * 5.143795) + (1 / 5 * 5.143795) +
        #                    (1 / 5 * 5.143795) =
        #                  = 5.143795.
        #
        #   Then LeakyReLU is applied (with no effect, since \beta_{i, j} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{0, 1}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{0, 7})) =
        #                   = exp(5.231658) / (exp(5.231658) + exp(5.143795)) =
        #                   ~= 0.5219516;
        #   - \alpha_{7->0} = exp(\beta_{0, 7}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{0, 7})) =
        #                   = exp(5.143795) / (exp(5.231658) + exp(5.143795)) =
        #                   ~= 0.4780484;
        #   - \alpha_{0->1} = exp(\beta_{0, 1}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(5.231658) / (exp(5.231658) + exp(5.231658) +
        #                     exp(5.282071)) =
        #                   ~= 0.3276856;
        #   - \alpha_{2->1} = exp(\beta_{1, 2}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(5.231658) / (exp(5.231658) + exp(5.231658) +
        #                      exp(5.282071)) =
        #                   ~= 0.3276856;
        #   - \alpha_{5->1} = exp(\beta_{1, 5}) / (exp(\beta_{0, 1}) +
        #                      exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(5.282071) / (exp(5.231658) + exp(5.231658) +
        #                     exp(5.282071)) =
        #                   ~= 0.3446287;
        #   - \alpha_{1->2} = exp(\beta_{1, 2}) / (exp(\beta_{1, 2} +
        #                     exp(\beta_{2, 3})) =
        #                   = exp(5.231658) / (exp(5.231658) + exp(5.214899)) =
        #                   ~= 0.5041897;
        #   - \alpha_{3->2} = exp(\beta_{2, 3}) / (exp(\beta_{1, 2} +
        #                     exp(\beta_{2, 3})) =
        #                   = exp(5.214899) / (exp(5.231658) + exp(5.214899)) =
        #                   ~= 0.4958103;
        #   - \alpha_{2->3} = exp(\beta_{2, 3}) / (exp(\beta_{2, 3} +
        #                     exp(\beta_{3, 4})) =
        #                   = exp(5.214899) / (exp(5.214899) + exp(5.214899)) =
        #                   = 0.5;
        #   - \alpha_{4->3} = exp(\beta_{3, 4}) / (exp(\beta_{2, 3} +
        #                     exp(\beta_{3, 4})) =
        #                   = exp(5.214899) / (exp(5.214899) + exp(5.214899)) =
        #                   = 0.5;
        #   - \alpha_{3->4} = exp(\beta_{3, 4}) / (exp(\beta_{3, 4} +
        #                     exp(\beta_{4, 5})) =
        #                   = exp(5.214899) / (exp(5.214899) + exp(4.993850)) =
        #                   = 0.5550383;
        #   - \alpha_{5->4} = exp(\beta_{4, 5}) / (exp(\beta_{3, 4} +
        #                     exp(\beta_{4, 5})) =
        #                   = exp(4.993850) / (exp(5.214899) + exp(4.993850)) =
        #                   = 0.4449617;
        #   - \alpha_{1->5} = exp(\beta_{1, 5}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(5.282071) / (exp(5.282071) + exp(4.993850) +
        #                      exp(4.915018)) =
        #                   = 0.4094386;
        #   - \alpha_{4->5} = exp(\beta_{4, 5}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(4.993850) / (exp(5.282071) + exp(4.993850) +
        #                      exp(4.915018)) =
        #                   = 0.3069135;
        #   - \alpha_{6->5} = exp(\beta_{5, 6}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(4.915018) / (exp(5.282071) + exp(4.993850) +
        #                      exp(4.915018)) =
        #                   = 0.2836480;
        #   - \alpha_{5->6} = exp(\beta_{5, 6}) / (exp(\beta_{5, 6} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(4.915018) / (exp(4.915018) + exp(5.143795)) =
        #                   = 0.4430539;
        #   - \alpha_{7->6} = exp(\beta_{6, 7}) / (exp(\beta_{5, 6} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(5.143795) / (exp(4.915018) + exp(5.143795)) =
        #                   = 0.5569461;
        #   - \alpha_{0->7} = exp(\beta_{0, 7}) / (exp(\beta_{0, 7} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(5.143795) / (exp(5.143795) + exp(5.143795)) =
        #                   = 0.5;
        #   - \alpha_{6->7} = exp(\beta_{6, 7}) / (exp(\beta_{0, 7} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(5.143795) / (exp(5.143795) + exp(5.143795)) =
        #                   = 0.5.
        #
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.5219516 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4780484 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(0.3276856 * [6., 6., 6.] +
        #                   (0.3276856 + 0.3446287) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            ~= [2.050153, 2.050153, 2.050153];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.5041897 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4958103 * [0.5, 0.5, 0.5]) =
        #            ~= [0.3109289, 0.3109289, 0.3109289];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(0.5 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(0.5550383 * [0.5, 0.5, 0.5] +
        #                   0.4449617 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [0.3331394, 0.3331394, 0.3331394];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU((0.4094386 + 0.3069135 + 0.2836480) * [1 / 8,
        #               1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(0.4430539 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5569461 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(0.5 * [6., 6., 6.] +
        #                   0.5 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [49 /16, 49 /16, 49 /16];
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 2.050153, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3109289, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3331394, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 49 / 16, 5)
        # Dual features.
        dual_node_to_feature = {
            (0, 1): 5.231658,
            (0, 7): 5.143795,
            (1, 2): 5.231658,
            (1, 5): 5.282071,
            (2, 3): 5.214899,
            (3, 4): 5.214899,
            (4, 5): 4.993850,
            (5, 6): 4.915018,
            (6, 7): 5.143795
        }

        for dual_node, dual_node_feature in dual_node_to_feature.items():
            dual_node_idx = graph_creator.primal_edge_to_dual_node_idx[
                dual_node]
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), dual_node_feature, 5)

    def test_simple_mesh_config_A_features_not_from_dual_no_self_loops(self):
        # - Dual-graph configuration A.
        single_dual_nodes = True
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 7
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=False)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last four features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 3:7] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios both equal to 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i, j} of the dual node {i, j}
        #   becomes \tilde{f}_{i, j} = f_{i, j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 2 / sqrt(3), 0, 0, 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 4 / sqrt(3), pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #      pi + 4 / sqrt(3), pi + 4 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge {j, n}->{i, j} and {i, m}->{i, j} - with m neighbor of i
        #   excluding j and n neighbor of j excluding i - the quantities (||
        #   indicates concatenation):
        #   * \tilde{\beta}_{{j, n}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{j, n} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3).
        #   * \tilde{\beta}_{{i, m}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, m} || \tilde{f}_{i, j}) =
        #     = pi + 4 / sqrt(3), by the symmetry of the dual features.
        #   NOTE: There are no self-loops in the dual graph.
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 4 / sqrt(3) > 0). Then, compute the softmax over the {j, n}'s
        #   and {i, m}'s neighboring nodes of {i, j}.
        #   Since all \tilde{\beta}_{i, j}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{{j, n}, {i, j}} and
        #   \tilde{\alpha}_{{i, m}, {i, j}} are simply
        #   1 / #(neighboring nodes {j, n} + neighboring nodes {i, m}).
        #   Therefore:
        #   - \tilde{\alpha}_{{0, m}, {0, 1}} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{{1, n}, {0, 1}} = 1 / 3 for m in {2, 5};
        #   - \tilde{\alpha}_{{0, m}, {0, 7}} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{{7, n}, {0, 7}} = 1 / 2 for m in {6};
        #   - \tilde{\alpha}_{{1, m}, {1, 2}} = 1 / 3 for m in {0, 5};
        #   - \tilde{\alpha}_{{2, n}, {1, 2}} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{{1, m}, {1, 5}} = 1 / 4 for m in {0, 2};
        #   - \tilde{\alpha}_{{5, n}, {1, 5}} = 1 / 4 for m in {4, 6};
        #   - \tilde{\alpha}_{{6, m}, {6, 7}} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{{7, n}, {6, 7}} = 1 / 2 for m in {0};
        #   - \tilde{\alpha}_{{2, m}, {2, 3}} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{{3, n}, {2, 3}} = 1 / 2 for m in {4};
        #   - \tilde{\alpha}_{{4, m}, {4, 5}} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{{5, n}, {4, 5}} = 1 / 3 for m in {1, 6};
        #   - \tilde{\alpha}_{{5, m}, {5, 6}} = 1 / 3 for m in {1, 4};
        #   - \tilde{\alpha}_{{6, n}, {5, 6}} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{{3, m}, {3, 4}} = 1 / 2 for m in {2};
        #   - \tilde{\alpha}_{{4, n}, {3, 4}} = 1 / 2 for m in {5}.
        #
        #  - The output features are then obtained as f_{i, j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{{i, m}, {i, j}} * \tilde{f}_{i, m}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}} * \tilde{f}_{j, n})).
        #    = ReLU(\tilde{f}_{i, j} * sum_m(\tilde{\alpha}_{{i, m}, {i, j}}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}}) =
        #    = ReLU(\tilde{f}_{i, j}) =
        #    = \tilde{f}_{i, j},
        #    where the third-last equality holds since the \tilde{f}_{i, m} =
        #    \tilde{f}_{j, n} = \tilde{f}_{i, j} for all valid i, j, m, n (cf.
        #    above), the second-last holds since the sum all the attention
        #    coefficients over the neighborhood of each dual node is 1 by
        #    construction, and the last one holds because \tilde{f}_{i, j} > 0
        #    for all valid i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i, j} = att_{primal}^T * f_{i, j}^' =
        #   = (1 / 5 * (pi + 4 / sqrt(3))) + (1 / 5 * (pi + 4 / sqrt(3))) +
        #     (1 / 5 * (pi + 4 / sqrt(3))) + (1 / 5 * (pi + 4 / sqrt(3))) +
        #     (1 / 5 * (pi + 4 / sqrt(3))) =
        #   = pi + 4 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 4 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node {i, j}.
        #   Since all \beta_{i, j}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{i, j} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{m, 0} = 1 / 2     for m in {1, 7};
        #   - \alpha_{m, 1} = 1 / 3     for m in {0, 2, 5};
        #   - \alpha_{m, 2} = 1 / 2     for m in {1, 3};
        #   - \alpha_{m, 3} = 1 / 2     for m in {2, 4};
        #   - \alpha_{m, 4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{m, 5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{m, 6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{m, 7} = 1 / 2     for m in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j, i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j, i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 4 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes {6, 7}, {1, 5} and {3, 4}.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        assert (dual_idx_67 == graph_creator.primal_edge_to_dual_node_idx[(7,
                                                                           6)])
        dual_idx_15 = graph_creator.primal_edge_to_dual_node_idx[(1, 5)]
        assert (dual_idx_15 == graph_creator.primal_edge_to_dual_node_idx[(5,
                                                                           1)])
        dual_idx_34 = graph_creator.primal_edge_to_dual_node_idx[(3, 4)]
        assert (dual_idx_34 == graph_creator.primal_edge_to_dual_node_idx[(4,
                                                                           3)])
        dual_graph.x[dual_idx_67, :] = torch.Tensor(
            [np.pi / 2, 1., 1., 0., 0., 0., 0.])
        dual_graph.x[dual_idx_15, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0.5, 0., 0., 0., 0.])
        dual_graph.x[dual_idx_34, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0.25, 0., 0., 0., 0.])
        # - As previously, to simplify computations, we manually set the last
        #   four features of each dual node (i.e., the edge-to-previous-edge-
        #   and edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 3:7] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node {i, j}           Nodes {j, n}/{i, m} with edges
        #                         {j, n}->{i, j} or {i, m}->{i, j}
        #   _____________________________________________________
        #       {0, 1}            {0, 7}, {1, 2}, {1, 5}
        #       {0, 7}            {0, 1}, {6, 7}
        #       {1, 2}            {0, 1}, {1, 5}, {2, 3}
        #       {1, 5}            {0, 1}, {1, 2}, {4, 5}, {5, 6}
        #       {2, 3}            {1, 2}, {3, 4}
        #       {3, 4}            {2, 3}, {4, 5}
        #       {4, 5}            {1, 5}, {3, 4}, {5, 6}
        #       {5, 6}            {1, 5}, {4, 5}, {6, 7}
        #       {6, 7}            {5, 6}, {0, 7}
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i, j} of the dual node {i, j}
        #   becomes \tilde{f}_{i, j} = f_{i, j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 2 / sqrt(3), 0, 0, 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 4 / sqrt(3), pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #      pi + 4 / sqrt(3), pi + 4 / sqrt(3)], for all {i, j} not in
        #   {{6, 7}, {1, 5}, {3, 4}}.
        #   We have also:
        #   - \tilde{f}_{6, 7} = [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #       pi / 2 + 2, pi / 2 + 2];
        #   - \tilde{f}_{1, 5} = [3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #       3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1];
        #   - \tilde{f}_{3, 4} = [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #       5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   edge {j, n}->{i, j} or {i, m}->{i, j} the quantities (|| indicates
        #   concatenation):
        #   * \tilde{\beta}_{{j, n}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{j, n} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all {j, n}, {i, j} not in {{6, 7}, {1, 5},
        #     {3, 4}}.
        #   * \tilde{\beta}_{{i, m}, {i, j}} = att_{dual}^T *
        #     (\tilde{f}_{i, m} || \tilde{f}_{i, j}) =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all {i, m}, {i, j} not in {{6, 7}, {1, 5},
        #     {3, 4}}.
        #   Likewise, we have also:
        #   - \tilde{beta}_{{5, 6}, {6, 7}} =
        #     = \tilde{beta}_{{0, 7}, {6, 7}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (pi / 2 + 2) =
        #     = 3 * pi / 4 + 2 / sqrt(3) + 1;
        #   - \tilde{beta}_{{6, 7}, {5, 6}} =
        #     = \tilde{beta}_{{6, 7}, {0, 7}} =
        #     = (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi / 2 + 2)) +
        #       (1 / 10 * (pi / 2 + 2)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) +
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (pi / 2 + 2) =
        #     = 3 * pi / 4 + 2 / sqrt(3) + 1;
        #   - \tilde{beta}_{{0, 1}, {1, 5}} =
        #     = \tilde{beta}_{{1, 2}, {1, 5}} =
        #     = \tilde{beta}_{{4, 5}, {1, 5}} =
        #     = \tilde{beta}_{{5, 6}, {1, 5}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 1) =
        #     = 7 * pi / 8 + 2 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{{1, 5}, {0, 1}} =
        #     = \tilde{beta}_{{1, 5}, {1, 2}} =
        #     = \tilde{beta}_{{1, 5}, {4, 5}} =
        #     = \tilde{beta}_{{1, 5}, {5, 6}} =
        #     = (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (3 * pi / 4 + 1)) +
        #       (1 / 10 * (3 * pi / 4 + 1)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 1)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 1) =
        #     = 7 * pi / 8 + 2 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{{2, 3}, {3, 4}} =
        #     = \tilde{beta}_{{4, 5}, {3, 4}} =
        #     = (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.5) =
        #     = 9 * pi / 8 + 2 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{{3, 4}, {2, 3}} =
        #     = \tilde{beta}_{{3, 4}, {4, 5}} =
        #     = (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (5 * pi / 4 + 0.5)) +
        #       (1 / 10 * (5 * pi / 4 + 0.5)) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 4 / sqrt(3))) =
        #     = 1 / 2 * (pi + 4 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.5) =
        #     = 9 * pi / 8 + 2 / sqrt(3) + 1 / 4;
        #   NOTE: We do not have self-loops in the dual graph.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{{j, n}, {i, j}} > 0, \tilde{beta}_{{i, m}, {i, j}} > 0
        #   and \tilde{beta}_{{i, j}, {i, j}} > 0 for all i, j, m!=j, n!=i).
        #   Then, compute the softmax over the neighboring nodes {j, n} and
        #   {i, m}. We have (cf. adjacency matrix):
        #   - \tilde{\alpha}_{{0, 7}, {0, 1}} =
        #     = \tilde{\alpha}_{{1, 2}, {0, 1}} =
        #     = exp(\tilde{beta}_{{0, 7}, {0, 1}}) /
        #       (exp(\tilde{beta}_{{0, 7}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 2}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 5}, {0, 1}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.4253783
        #   - \tilde{\alpha}_{{1, 5}, {0, 1}} =
        #     = exp(\tilde{beta}_{{1, 5}, {0, 1}}) /
        #       (exp(\tilde{beta}_{{0, 7}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 2}, {0, 1}}) +
        #        exp(\tilde{beta}_{{1, 5}, {0, 1}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1492435
        #   - \tilde{\alpha}_{{0, 1}, {0, 7}} =
        #     = exp(\tilde{beta}_{{0, 1}, {0, 7}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {0, 7}}))
        #     = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.7191196
        #   - \tilde{\alpha}_{{6, 7}, {0, 7}} =
        #     = exp(\tilde{beta}_{{6, 1}, {0, 7}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {0, 7}}) +
        #        exp(\tilde{beta}_{{6, 7}, {0, 7}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) / (exp(pi + 4 / sqrt(3)) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.2808804
        #   - \tilde{\alpha}_{{0, 1}, {1, 2}} =
        #     = \tilde{\alpha}_{{2, 3}, {1, 2}} =
        #     = exp(\tilde{beta}_{{0, 1}, {1, 2}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 2}}) +
        #        exp(\tilde{beta}_{{2, 3}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 2}}))
        #     = exp(pi + 4 / sqrt(3)) / (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.4253783
        #   - \tilde{\alpha}_{{1, 5}, {1, 2}} =
        #     = exp(\tilde{beta}_{{1, 5}, {1, 2}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 2}}) +
        #        exp(\tilde{beta}_{{2, 3}, {1, 2}}) +
        #        exp(\tilde{beta}_{{1, 5}, {1, 2}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (2 * exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1492435
        #   - \tilde{\alpha}_{{0, 1}, {1, 5}} =
        #     = \tilde{\alpha}_{{1, 2}, {1, 5}} =
        #     = \tilde{\alpha}_{{4, 5}, {1, 5}} =
        #     = \tilde{\alpha}_{{5, 6}, {1, 5}} =
        #     = exp(\tilde{beta}_{{0, 1}, {1, 5}}) /
        #       (exp(\tilde{beta}_{{0, 1}, {1, 5}}) +
        #        exp(\tilde{beta}_{{1, 2}, {1, 5}}) +
        #        exp(\tilde{beta}_{{4, 5}, {1, 5}}) +
        #        exp(\tilde{beta}_{{5, 6}, {1, 5}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (4 * exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     = 0.25
        #   - \tilde{\alpha}_{{1, 2}, {2, 3}} =
        #     = exp(\tilde{beta}_{{1, 2}, {2, 3}}) /
        #       (exp(\tilde{beta}_{{1, 2}, {2, 3}}) +
        #        exp(\tilde{beta}_{{3, 4}, {2, 3}}))
        #     = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4)) =
        #     ~= 0.6252755
        #   - \tilde{\alpha}_{{3, 4}, {2, 3}} =
        #     = exp(\tilde{beta}_{{3, 4}, {2, 3}}) /
        #       (exp(\tilde{beta}_{{1, 2}, {2, 3}}) +
        #        exp(\tilde{beta}_{{3, 4}, {2, 3}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4)) =
        #     ~= 0.3747245
        #   - \tilde{\alpha}_{{2, 3}, {3, 4}} =
        #     = \tilde{\alpha}_{{4, 5}, {3, 4}} =
        #     = exp(\tilde{beta}_{{2, 3}, {3, 4}}) /
        #       (exp(\tilde{beta}_{{2, 3}, {3, 4}}) +
        #        exp(\tilde{beta}_{{4, 5}, {3, 4}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (2 * exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4)) =
        #     = 0.5
        #   - \tilde{\alpha}_{{5, 6}, {4, 5}} =
        #     = exp(\tilde{beta}_{{5, 6}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.5127827
        #   - \tilde{\alpha}_{{3, 4}, {4, 5}} =
        #     = exp(\tilde{beta}_{{3, 4}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) /
        #       (exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.3073081
        #   - \tilde{\alpha}_{{1, 5}, {4, 5}} =
        #     = exp(\tilde{beta}_{{1, 5}, {4, 5}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {4, 5}}) +
        #        exp(\tilde{beta}_{{3, 4}, {4, 5}}) +
        #        exp(\tilde{beta}_{{1, 5}, {4, 5}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (exp(pi + 4 / sqrt(3)) +
        #        exp(9 * pi / 8 + 2 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2)) =
        #     ~= 0.1799092
        #   - \tilde{\alpha}_{{4, 5}, {5, 6}} =
        #     = exp(\tilde{beta}_{{4, 5}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.5742380
        #   - \tilde{\alpha}_{{1, 5}, {5, 6}} =
        #     = exp(\tilde{beta}_{{1, 5}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) /
        #       (exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.2014708
        #   - \tilde{\alpha}_{{6, 7}, {5, 6}} =
        #     = exp(\tilde{beta}_{{6, 7}, {5, 6}}) /
        #       (exp(\tilde{beta}_{{4, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{1, 5}, {5, 6}}) +
        #        exp(\tilde{beta}_{{6, 7}, {5, 6}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) /
        #       (exp(pi + 4 / sqrt(3)) +
        #        exp(7 * pi / 8 + 2 / sqrt(3) + 1 / 2) +
        #        exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     ~= 0.2242912
        #   - \tilde{\alpha}_{{0, 7}, {6, 7}} =
        #     = \tilde{\alpha}_{{5, 6}, {6, 7}} =
        #     = exp(\tilde{beta}_{{0, 7}, {6, 7}}) /
        #       (exp(\tilde{beta}_{{5, 6}, {6, 7}}))
        #     = exp(3 * pi / 4 + 2 / sqrt(3) + 1) /
        #       (2 * exp(3 * pi / 4 + 2 / sqrt(3) + 1)) =
        #     = 0.5
        #
        # - The output features are then obtained as f_{i, j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{{i, m}, {i, j}} * \tilde{f}_{i, m}) +
        #           sum_n(\tilde{\alpha}_{{j, n}, {i, j}} * \tilde{f}_{j, n})).
        #    We thus have:
        #    - f_{0, 1}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 7}, {0, 1}} * \tilde{f}_{0, 7} +
        #                   \tilde{\alpha}_{{1, 2}, {0, 1}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{1, 5}, {0, 1}} * \tilde{f}_{1, 5})
        #                 = ReLU((0.4253783 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1492435 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1]) =
        #                ~= [5.138359, 5.138359, 5.138359, 5.138359, 5.138359];
        #    - f_{0, 7}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {0, 7}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{6, 7}, {0, 7}} * \tilde{f}_{6, 7})
        #                 = ReLU(0.7191196 * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.2808804 *
        #                   [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #                    pi / 2 + 2]) =
        #                ~= [4.922883, 4.922883, 4.922883, 4.922883, 4.922883];
        #    - f_{1, 2}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {1, 2}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{1, 5}, {1, 2}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{2, 3}, {1, 2}} * \tilde{f}_{2, 3})
        #                 = ReLU((0.4253783 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.1492435 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1]) =
        #                ~= [5.138359, 5.138359, 5.138359, 5.138359, 5.138359];
        #    - f_{1, 5}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 1}, {1, 5}} * \tilde{f}_{0, 1} +
        #                   \tilde{\alpha}_{{1, 2}, {1, 5}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{4, 5}, {1, 5}} * \tilde{f}_{4, 5} +
        #                   \tilde{\alpha}_{{5, 6}, {1, 5}} * \tilde{f}_{5, 6})
        #                 = ReLU((0.25 * 4) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)]) =
        #                 = [pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3)];
        #    - f_{2, 3}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 2}, {2, 3}} * \tilde{f}_{1, 2} +
        #                   \tilde{\alpha}_{{3, 4}, {2, 3}} * \tilde{f}_{3, 4})
        #                 = ReLU(0.6252755 * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.3747245 *
        #                   [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5]) =
        #                ~= [5.067275, 5.067275, 5.067275, 5.067275, 5.067275];
        #    - f_{3, 4}^' = ReLU(
        #                   \tilde{\alpha}_{{2, 3}, {3, 4}} * \tilde{f}_{2, 3} +
        #                   \tilde{\alpha}_{{4, 5}, {3, 4}} * \tilde{f}_{4, 5})
        #                 = ReLU((0.5 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)]) =
        #                 = [pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3)];
        #    - f_{4, 5}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 5}, {4, 5}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{3, 4}, {4, 5}} * \tilde{f}_{3, 4} +
        #                   \tilde{\alpha}_{{5, 6}, {4, 5}} * \tilde{f}_{5, 6})
        #                 = ReLU(0.5127827 * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.3073081 *
        #                   [5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5, 5 * pi / 4 + 0.5,
        #                    5 * pi / 4 + 0.5] + 0.1799092 * [3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1]) =
        #                ~= [4.759436, 4.759436, 4.759436, 4.759436, 4.759436];
        #    - f_{5, 6}^' = ReLU(
        #                   \tilde{\alpha}_{{1, 5}, {5, 6}} * \tilde{f}_{1, 5} +
        #                   \tilde{\alpha}_{{4, 5}, {5, 6}} * \tilde{f}_{4, 5} +
        #                   \tilde{\alpha}_{{6, 7}, {5, 6}} * \tilde{f}_{6, 7})
        #                 = ReLU(0.5742380 * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)] + 0.2014708 *
        #                   [3 * pi / 4 + 1, 3 * pi / 4 + 1, 3 * pi / 4 + 1,
        #                    3 * pi / 4 + 1, 3 * pi / 4 + 1] + 0.2242912 *
        #                   [pi / 2 + 2, pi / 2 + 2, pi / 2 + 2, pi / 2 + 2,
        #                    pi / 2 + 2]) =
        #                ~= [4.607241, 4.607241, 4.607241, 4.607241, 4.607241];
        #    - f_{6, 7}^' = ReLU(
        #                   \tilde{\alpha}_{{0, 7}, {6, 7}} * \tilde{f}_{0, 7} +
        #                   \tilde{\alpha}_{{5, 6}, {6, 7}} * \tilde{f}_{5, 6})
        #                 = ReLU((0.5 * 2) * [pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                   pi + 4 / sqrt(3), pi + 4 / sqrt(3)]) =
        #                 = [pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3), pi + 4 / sqrt(3),
        #                    pi + 4 / sqrt(3)].
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i, j} = att_{primal}^T * f_{i, j}^'. We thus have:
        #
        #   - \beta_{0, 1} = att_{primal}^T * f_{0, 1}^' =
        #                  = (1 / 5 * 5.138359) + (1 / 5 * 5.138359) +
        #                    (1 / 5 * 5.138359) + (1 / 5 * 5.138359) +
        #                    (1 / 5 * 5.138359) =
        #                  = 5.138359;
        #   - \beta_{0, 7} = att_{primal}^T * f_{0, 7}^' =
        #                  = (1 / 5 * 4.922883) + (1 / 5 * 4.922883) +
        #                    (1 / 5 * 4.922883) + (1 / 5 * 4.922883) +
        #                    (1 / 5 * 4.922883) =
        #                  = 4.922883;
        #   - \beta_{1, 2} = att_{primal}^T * f_{1, 2}^' =
        #                  = (1 / 5 * 5.138359) + (1 / 5 * 5.138359) +
        #                    (1 / 5 * 5.138359) + (1 / 5 * 5.138359) +
        #                    (1 / 5 * 5.138359) =
        #                  = 5.138359;
        #   - \beta_{1, 5} = att_{primal}^T * f_{1, 5}^' =
        #                  = (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) =
        #                  = pi + 4 / sqrt(3);
        #   - \beta_{2, 3} = att_{primal}^T * f_{2, 3}^' =
        #                  = (1 / 5 * 5.067275) + (1 / 5 * 5.067275) +
        #                    (1 / 5 * 5.067275) + (1 / 5 * 5.067275) +
        #                    (1 / 5 * 5.067275) =
        #                  = 5.067275;
        #   - \beta_{3, 4} = att_{primal}^T * f_{3, 4}^' =
        #                  = (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) =
        #                  = pi + 4 / sqrt(3);
        #   - \beta_{4, 5} = att_{primal}^T * f_{4, 5}^' =
        #                  = (1 / 5 * 4.759436) + (1 / 5 * 4.759436) +
        #                    (1 / 5 * 4.759436) + (1 / 5 * 4.759436) +
        #                    (1 / 5 * 4.759436) =
        #                  = 4.759436;
        #   - \beta_{5, 6} = att_{primal}^T * f_{5, 6}^' =
        #                  = (1 / 5 * 4.607241) + (1 / 5 * 4.607241) +
        #                    (1 / 5 * 4.607241) + (1 / 5 * 4.607241) +
        #                    (1 / 5 * 4.607241) =
        #                  = 4.607241;
        #   - \beta_{6, 7} = att_{primal}^T * f_{6, 7}^' =
        #                  = (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) +
        #                    (1 / 5 * pi + 4 / sqrt(3)) =
        #                  = pi + 4 / sqrt(3);
        #
        #   Then LeakyReLU is applied (with no effect, since \beta_{i, j} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{0, 1}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{0, 7})) =
        #                   = exp(5.138359) / (exp(5.138359) + exp(4.922883)) =
        #                   ~= 0.5536615;
        #   - \alpha_{7->0} = exp(\beta_{0, 7}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{0, 7})) =
        #                   = exp(4.922883) / (exp(5.138359) + exp(4.922883)) =
        #                   ~= 0.4463385;
        #   - \alpha_{0->1} = exp(\beta_{0, 1}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(5.138359) / (exp(5.138359) + exp(5.138359) +
        #                     exp(pi + 4 / sqrt(3))) =
        #                   ~= 0.2969983;
        #   - \alpha_{2->1} = exp(\beta_{1, 2}) / (exp(\beta_{0, 1}) +
        #                     exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(5.138359) / (exp(5.138359) + exp(5.138359) +
        #                     exp(pi + 4 / sqrt(3))) =
        #                   ~= 0.2969983;
        #   - \alpha_{5->1} = exp(\beta_{1, 5}) / (exp(\beta_{0, 1}) +
        #                      exp(\beta_{1, 2}) + exp(\beta_{1, 5})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(5.138359) +
        #                      exp(5.138359) + exp(pi + 4 / sqrt(3))) =
        #                   ~= 0.4060033;
        #   - \alpha_{1->2} = exp(\beta_{1, 2}) / (exp(\beta_{1, 2} +
        #                     exp(\beta_{2, 3})) =
        #                   = exp(5.138359) / (exp(5.138359) + exp(5.067275)) =
        #                   ~= 0.5177635;
        #   - \alpha_{3->2} = exp(\beta_{2, 3}) / (exp(\beta_{1, 2} +
        #                     exp(\beta_{2, 3})) =
        #                   = exp(5.067275) / (exp(5.138359) + exp(5.067275)) =
        #                   ~= 0.4822365;
        #   - \alpha_{2->3} = exp(\beta_{2, 3}) / (exp(\beta_{2, 3} +
        #                     exp(\beta_{3, 4})) =
        #                   = exp(5.067275) / (exp(5.067275) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.4052303;
        #   - \alpha_{4->3} = exp(\beta_{3, 4}) / (exp(\beta_{2, 3} +
        #                     exp(\beta_{3, 4})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(5.067275) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.5947697;
        #   - \alpha_{3->4} = exp(\beta_{3, 4}) / (exp(\beta_{3, 4} +
        #                     exp(\beta_{4, 5})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #                      exp(4.759436)) =
        #                   = 0.6663134;
        #   - \alpha_{5->4} = exp(\beta_{4, 5}) / (exp(\beta_{3, 4} +
        #                     exp(\beta_{4, 5})) =
        #                   = exp(4.759436) / (exp(pi + 4 / sqrt(3)) +
        #                      exp(4.759436)) =
        #                   = 0.3336866;
        #   - \alpha_{1->5} = exp(\beta_{1, 5}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(pi + 4 / sqrt(3)) +
        #                      exp(4.759436) + exp(4.607241)) =
        #                   = 0.5178962;
        #   - \alpha_{4->5} = exp(\beta_{4, 5}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(4.759436) / (exp(pi + 4 / sqrt(3)) +
        #                      exp(4.759436) + exp(4.607241)) =
        #                   = 0.2593600;
        #   - \alpha_{6->5} = exp(\beta_{5, 6}) / (exp(\beta_{1, 5} +
        #                     exp(\beta_{4, 5}) + exp(\beta_{5, 6})) =
        #                   = exp(4.607241) / (exp(pi + 4 / sqrt(3)) +
        #                      exp(4.759436) + exp(4.607241)) =
        #                   = 0.2227438;
        #   - \alpha_{5->6} = exp(\beta_{5, 6}) / (exp(\beta_{5, 6} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(4.607241) / (exp(4.607241) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.3007450;
        #   - \alpha_{7->6} = exp(\beta_{6, 7}) / (exp(\beta_{5, 6} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(4.607241) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.6992550;
        #   - \alpha_{0->7} = exp(\beta_{0, 7}) / (exp(\beta_{0, 7} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(4.922883) / (exp(4.922883) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.3709576;
        #   - \alpha_{6->7} = exp(\beta_{6, 7}) / (exp(\beta_{0, 7} +
        #                     exp(\beta_{6, 7})) =
        #                   = exp(pi + 4 / sqrt(3)) / (exp(4.922883) +
        #                      exp(pi + 4 / sqrt(3))) =
        #                   = 0.6290424.
        #
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.5536615 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4463385 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(0.2969983 * [6., 6., 6.] +
        #                   (0.2969983 + 0.4060033) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            ~= [1.869865, 1.869865, 1.869865];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.5177635 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4822365 * [0.5, 0.5, 0.5]) =
        #            ~= [0.3058387, 0.3058387, 0.3058387];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(0.4052303 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5947697 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(0.6663134 * [0.5, 0.5, 0.5] +
        #                   0.3336866 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [0.3748675, 0.3748675, 0.3748675];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU((0.5178962 + 0.2593600 + 0.3007450) * [1 / 8,
        #               1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(0.3007450 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.6992550 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(0.3709576 * [6., 6., 6.] +
        #                   0.6290424 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [2.304376, 2.304376, 2.304376].
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))

        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 1.869865, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3058387, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3748675, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 2.304376, 5)

        # Dual features.
        dual_node_to_feature = {
            (0, 1): 5.138359,
            (0, 7): 4.922883,
            (1, 2): 5.138359,
            (1, 5): np.pi + 4 / np.sqrt(3),
            (2, 3): 5.067275,
            (3, 4): np.pi + 4 / np.sqrt(3),
            (4, 5): 4.759436,
            (5, 6): 4.607241,
            (6, 7): np.pi + 4 / np.sqrt(3)
        }

        for dual_node, dual_node_feature in dual_node_to_feature.items():
            dual_node_idx = graph_creator.primal_edge_to_dual_node_idx[
                dual_node]
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), dual_node_feature, 5)

    def test_simple_mesh_config_B_features_not_from_dual_with_self_loops(self):
        # - Dual-graph configuration B.
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=True)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last two features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios both equal to 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i->j} of the dual node i->j
        #   becomes \tilde{f}_{i->j} = f_{i->j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->i)->(i->j), (j->n)->(i->j) - with m neighbor of i and n
        #   neighbor of j - the quantities (|| indicates concatenation):
        #   * \tilde{\beta}_{m->i, i->j} = att_{dual}^T *
        #     (\tilde{f}_{m->i} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 2 / sqrt(3).
        #   * \tilde{\beta}_{{j->n, i->j} = att_{dual}^T *
        #     (\tilde{f}_{j->n} || \tilde{f}_{i->j}) =
        #     = pi + 2 / sqrt(3), by the symmetry of the dual features.
        #   NOTE: We also have self-loops in the dual graph! Hence, one has:
        #   - \tilde{\beta}_{{i->j, i->j} = att_{dual}^T *
        #     (\tilde{f}_{i->j} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all i, j.
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 2 / sqrt(3) > 0). Then, compute the softmax over the (m->i)'s
        #   and (j->n)'s neighboring nodes of (i->j), including also i->j.
        #   Since all \tilde{\beta}_{i->j}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{m->i, i->j} and
        #   \tilde{\alpha}_{j->n, i->j} are simply
        #   1 / #(neighboring nodes (m->i) + neighboring nodes (j->n) + 1 (for
        #   self-loop)). Therefore:
        #   - \tilde{\alpha}_{m->0, 0->1} = 1 / 4 for m in {7};
        #   - \tilde{\alpha}_{1->n, 0->1} = 1 / 4 for m in {2, 5};
        #   - \tilde{\alpha}_{0->1, 0->1} = 1 / 4;
        #   - \tilde{\alpha}_{m->0, 0->7} = 1 / 3 for m in {1};
        #   - \tilde{\alpha}_{7->n, 0->7} = 1 / 3 for m in {6};
        #   - \tilde{\alpha}_{0->7, 0->7} = 1 / 3;
        #   - \tilde{\alpha}_{m->1, 1->0} = 1 / 4 for m in {2, 5};
        #   - \tilde{\alpha}_{0->n, 1->0} = 1 / 4 for m in {7};
        #   - \tilde{\alpha}_{1->0, 1->0} = 1 / 4;
        #   - \tilde{\alpha}_{m->1, 1->2} = 1 / 4 for m in {0, 5};
        #   - \tilde{\alpha}_{2->n, 1->2} = 1 / 4 for m in {3};
        #   - \tilde{\alpha}_{1->2, 1->2} = 1 / 4;
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 5 for m in {0, 2};
        #   - \tilde{\alpha}_{5->n, 1->5} = 1 / 5 for m in {4, 6};
        #   - \tilde{\alpha}_{1->5, 1->5} = 1 / 5;
        #   - \tilde{\alpha}_{m->2, 2->1} = 1 / 4 for m in {3};
        #   - \tilde{\alpha}_{1->n, 2->1} = 1 / 4 for m in {0, 5};
        #   - \tilde{\alpha}_{2->1, 2->1} = 1 / 4;
        #   - \tilde{\alpha}_{m->2, 2->3} = 1 / 3 for m in {1};
        #   - \tilde{\alpha}_{3->n, 2->3} = 1 / 3 for m in {4};
        #   - \tilde{\alpha}_{2->3, 2->3} = 1 / 3;
        #   - \tilde{\alpha}_{m->3, 3->2} = 1 / 3 for m in {4};
        #   - \tilde{\alpha}_{2->n, 3->2} = 1 / 3 for m in {1};
        #   - \tilde{\alpha}_{3->2, 3->2} = 1 / 3;
        #   - \tilde{\alpha}_{m->3, 3->4} = 1 / 3 for m in {2};
        #   - \tilde{\alpha}_{4->n, 3->4} = 1 / 3 for m in {5};
        #   - \tilde{\alpha}_{3->4, 3->4} = 1 / 3.
        #   - \tilde{\alpha}_{m->4, 4->3} = 1 / 3 for m in {5};
        #   - \tilde{\alpha}_{3->n, 4->3} = 1 / 3 for m in {2};
        #   - \tilde{\alpha}_{4->3, 4->3} = 1 / 3.
        #   - \tilde{\alpha}_{m->4, 4->5} = 1 / 4 for m in {3};
        #   - \tilde{\alpha}_{5->n, 4->5} = 1 / 4 for m in {1, 6};
        #   - \tilde{\alpha}_{4->5, 4->5} = 1 / 4;
        #   - \tilde{\alpha}_{m->5, 5->1} = 1 / 5 for m in {4, 6};
        #   - \tilde{\alpha}_{1->n, 5->1} = 1 / 5 for m in {0, 2};
        #   - \tilde{\alpha}_{5->1, 5->1} = 1 / 5;
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 4 for m in {1, 6};
        #   - \tilde{\alpha}_{4->n, 5->4} = 1 / 4 for m in {3};
        #   - \tilde{\alpha}_{5->4, 5->4} = 1 / 4;
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 4 for m in {1, 4};
        #   - \tilde{\alpha}_{6->n, 5->6} = 1 / 4 for m in {7};
        #   - \tilde{\alpha}_{5->6, 5->6} = 1 / 4;
        #   - \tilde{\alpha}_{m->6, 6->5} = 1 / 4 for m in {7};
        #   - \tilde{\alpha}_{5->n, 6->5} = 1 / 4 for m in {1, 4};
        #   - \tilde{\alpha}_{6->5, 6->5} = 1 / 4;
        #   - \tilde{\alpha}_{m->6, 6->7} = 1 / 3 for m in {5};
        #   - \tilde{\alpha}_{7->n, 6->7} = 1 / 3 for m in {0};
        #   - \tilde{\alpha}_{6->7, 6->7} = 1 / 3;
        #   - \tilde{\alpha}_{m->7, 7->0} = 1 / 3 for m in {6};
        #   - \tilde{\alpha}_{0->n, 7->0} = 1 / 3 for m in {1};
        #   - \tilde{\alpha}_{7->0, 7->0} = 1 / 4;
        #   - \tilde{\alpha}_{m->7, 7->6} = 1 / 3 for m in {0};
        #   - \tilde{\alpha}_{6->n, 7->6} = 1 / 3 for m in {5};
        #   - \tilde{\alpha}_{7->6, 7->6} = 1 / 3.
        #
        #  - The output features are then obtained as f_{i->j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->i, i->j} * \tilde{f}_{m->i}) +
        #           sum_n(\tilde{\alpha}_{j->n, i->j} * \tilde{f}_{j->n}) +
        #           \tilde{\alpha}_{i->j, i->j} * \tilde{f}_{i->j})
        #    = ReLU(\tilde{f}_{i->j} * sum_m(\tilde{\alpha}_{m->i, i->j}) +
        #           sum_n(\tilde{\alpha}_{j->n, i->j} +
        #           \tilde{\alpha}_{i->j, i->j}) =
        #    = ReLU(\tilde{f}_{i->j}) =
        #    = \tilde{f}_{i->j},
        #    where the third-last equality holds since the \tilde{f}_{m->i} =
        #    \tilde{f}_{j->n} = \tilde{f}_{i->j} for all valid i, j, m, n (cf.
        #    above), the second-last holds since the sum all the attention
        #    coefficients over the neighborhood of each dual node (dual node
        #    included) is 1 by construction, and the last one holds because
        #    \tilde{f}_{i->j} > 0 for all valid i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i->j} = att_{primal}^T * f_{i->j}^' =
        #   = (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 2 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node {i, j}.
        #   Since all \beta_{i->j}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{i->j} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{m, 0} = 1 / 2     for m in {1, 7};
        #   - \alpha_{m, 1} = 1 / 3     for m in {0, 2, 5};
        #   - \alpha_{m, 2} = 1 / 2     for m in {1, 3};
        #   - \alpha_{m, 3} = 1 / 2     for m in {2, 4};
        #   - \alpha_{m, 4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{m, 5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{m, 6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{m, 7} = 1 / 2     for m in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j, i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j, i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes 6->7, 5->1 and 4->3.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        dual_idx_51 = graph_creator.primal_edge_to_dual_node_idx[(5, 1)]
        dual_idx_43 = graph_creator.primal_edge_to_dual_node_idx[(4, 3)]
        dual_graph.x[dual_idx_67, :] = torch.Tensor([np.pi / 2, 1., 0., 0.])
        dual_graph.x[dual_idx_51, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0., 0.])
        dual_graph.x[dual_idx_43, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0., 0.])
        assert (dual_idx_67 !=
                graph_creator.primal_edge_to_dual_node_idx[(7, 6)])
        assert (dual_idx_51 !=
                graph_creator.primal_edge_to_dual_node_idx[(1, 5)])
        assert (dual_idx_43 !=
                graph_creator.primal_edge_to_dual_node_idx[(3, 4)])
        # - As previously, to simplify computations, we manually set the last
        #   two features of each dual node (i.e., the edge-to-previous-edge-
        #   and edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node i->j           Nodes (m->i)/(j->n) with edges
        #                         (m->i)->(i->j) or (j->n)->(i->j)
        #   _____________________________________________________
        #       0->1            1->2, 1->5, 7->0
        #       0->7            1->0, 7->6
        #       1->0            0->7, 2->1, 5->1
        #       1->2            1->0, 2->3, 5->1
        #       1->5            0->1, 2->1, 5->4, 5->6
        #       2->1            1->0, 3->2, 1->5
        #       2->3            1->2, 3->4
        #       3->2            2->1, 4->3
        #       3->4            2->3, 4->5
        #       4->3            3->2, 5->4
        #       4->5            3->4, 5->1, 5->6
        #       5->1            1->0, 1->2, 4->5, 6->5
        #       5->4            1->5, 4->3, 6->5
        #       5->6            1->5, 4->5, 6->7
        #       6->5            5->1, 5->4, 7->6
        #       6->7            5->6, 7->0
        #       7->0            0->1, 6->7
        #       7->6            0->7, 6->5
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i->j} of the dual node i->j
        #   becomes \tilde{f}_{i->j} = f_{i->j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)], for all i->j not in
        #   {6->7, 5->1, 4->3}.
        #   We have also:
        #   - \tilde{f}_{6->7} = [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #       pi / 2 + 1, pi / 2 + 1];
        #   - \tilde{f}_{5->1} = [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #       3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5];
        #   - \tilde{f}_{4->3} = [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #       5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25];
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->i)->(i->j) or (j->n)->(i->j) the quantities (|| indicates
        #   concatenation):
        #   * \tilde{\beta}_{j->n, i->j} = att_{dual}^T *
        #     (\tilde{f}_{j->n} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 2 / sqrt(3), for all j->n, i->j not in {6->7, 5->1, 4->3}.
        #   * \tilde{\beta}_{m->i, i->j} = att_{dual}^T *
        #     (\tilde{f}_{m->i} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all j->n, i->j not in {6->7, 5->1, 4->3}
        #   Likewise, we have also:
        #   - \tilde{beta}_{5->6, 6->7} =
        #     = \tilde{beta}_{7->0, 6->7} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 0.5;
        #   - \tilde{beta}_{6->7, 5->6} =
        #     = \tilde{beta}_{6->7, 7->0} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) +
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 0.5;
        #   - \tilde{beta}_{1->0, 5->1} =
        #     = \tilde{beta}_{1->2, 5->1} =
        #     = \tilde{beta}_{4->5, 5->1} =
        #     = \tilde{beta}_{6->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->1, 1->0} =
        #     = \tilde{beta}_{5->1, 1->2} =
        #     = \tilde{beta}_{5->1, 4->5} =
        #     = \tilde{beta}_{5->1, 6->5} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3)))) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{3->2, 4->3} =
        #     = \tilde{beta}_{5->4, 4->3} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   - \tilde{beta}_{4->3, 3->2} =
        #     = \tilde{beta}_{4->3, 5->4} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   NOTE: We also have self-loops in the dual graph! Thus, one has:
        #   - \tilde{\beta}_{i->j, i->j} = att_{dual}^T *
        #     (\tilde{f}_i->j || \tilde{f}_i->j) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all i->j not in {6->7, 5->1, 4->3}.
        #   - \tilde{beta}_{6->7, 6->7} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = pi / 2 + 1;
        #   - \tilde{beta}_{5->1, 5->1} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 3 * pi / 4 + 0.5;
        #   - \tilde{beta}_{4->3, 4->3} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 5 * pi / 4 + 0.25.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{m->i, i->j} > 0, \tilde{beta}_{j->n, i->j} > 0
        #   and \tilde{beta}_{i->j, i->j} > 0 for all i, j, m, n). Then,
        #   compute the softmax over the neighboring nodes m->i and j->n. We
        #   have (cf. adjacency matrix + self-loops):
        #   - \tilde{\alpha}_{0->1, 0->1} =
        #     = \tilde{\alpha}_{7->0, 0->1} =
        #     = \tilde{\alpha}_{1->2, 0->1} =
        #     = \tilde{\alpha}_{1->5, 0->1} =
        #     = exp(\tilde{beta}_{0->1, 0->1}) /
        #       (exp(\tilde{beta}_{0->1, 0->1}) +
        #        exp(\tilde{beta}_{7->0, 0->1}) +
        #        exp(\tilde{beta}_{1->2, 0->1}) +
        #        exp(\tilde{beta}_{1->5, 0->1}))
        #     = exp(pi + 2 / sqrt(3)) / (4 * exp(pi + 2 / sqrt(3))) =
        #     = 0.25
        #   - \tilde{\alpha}_{0->7, 0->7} =
        #     = \tilde{\alpha}_{1->0, 0->7} =
        #     = \tilde{\alpha}_{7->6, 0->7} =
        #     = exp(\tilde{beta}_{0->7, 0->7}) /
        #       (exp(\tilde{beta}_{0->7, 0->7}) +
        #        exp(\tilde{beta}_{1->0, 0->7}) +
        #        exp(\tilde{beta}_{7->6, 0->7}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #   - \tilde{\alpha}_{0->7, 1->0} =
        #     = \tilde{\alpha}_{1->0, 1->0} =
        #     = \tilde{\alpha}_{2->1, 1->0} =
        #     = exp(\tilde{beta}_{0->7, 1->0}) /
        #       (exp(\tilde{beta}_{0->7, 1->0}) +
        #        exp(\tilde{beta}_{1->0, 1->0}) +
        #        exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.2868018
        #   - \tilde{\alpha}_{5->1, 1->0} =
        #     = exp(\tilde{beta}_{5->1, 1->0}) /
        #       (exp(\tilde{beta}_{0->7, 1->0}) +
        #        exp(\tilde{beta}_{1->0, 1->0}) +
        #        exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1395945
        #   - \tilde{\alpha}_{0->1, 1->2} =
        #     = \tilde{\alpha}_{1->2, 1->2} =
        #     = \tilde{\alpha}_{2->3, 1->2} =
        #     = exp(\tilde{beta}_{0->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{1->2, 1->2}) +
        #        exp(\tilde{beta}_{2->3, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.2868018
        #   - \tilde{\alpha}_{5->1, 1->2} =
        #     = exp(\tilde{beta}_{5->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{1->2, 1->2}) +
        #        exp(\tilde{beta}_{2->3, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1395945
        #   - \tilde{\alpha}_{0->1, 1->5} =
        #     = \tilde{\alpha}_{1->5, 1->5} =
        #     = \tilde{\alpha}_{2->1, 1->5} =
        #     = \tilde{\alpha}_{5->4, 1->5} =
        #     = \tilde{\alpha}_{5->6, 1->5} =
        #     = exp(\tilde{beta}_{0->1, 1->5}) /
        #       (exp(\tilde{beta}_{0->1, 1->5}) +
        #        exp(\tilde{beta}_{1->5, 1->5}) +
        #        exp(\tilde{beta}_{2->1, 1->5}) +
        #        exp(\tilde{beta}_{5->4, 1->5}) +
        #        exp(\tilde{beta}_{5->6, 1->5}) +
        #     = 0.2
        #   - \tilde{\alpha}_{1->0, 2->1} =
        #     = \tilde{\alpha}_{1->5, 2->1} =
        #     = \tilde{\alpha}_{2->1, 2->1} =
        #     = \tilde{\alpha}_{3->2, 2->1} =
        #     = exp(\tilde{beta}_{1->0, 2->1}) /
        #       (exp(\tilde{beta}_{1->0, 2->1}) +
        #        exp(\tilde{beta}_{1->5, 2->1}) +
        #        exp(\tilde{beta}_{2->1, 2->1}) +
        #        exp(\tilde{beta}_{3->2, 2->1}))
        #     = exp(pi + 2 / sqrt(3)) / (4 * exp(pi + 2 / sqrt(3))) =
        #     = 0.25
        #   - \tilde{\alpha}_{1->2, 2->3} =
        #     = \tilde{\alpha}_{2->3, 2->3} =
        #     = \tilde{\alpha}_{3->4, 2->3} =
        #     = exp(\tilde{beta}_{1->2, 2->3}) /
        #       (exp(\tilde{beta}_{1->2, 2->3}) +
        #        exp(\tilde{beta}_{2->3, 2->3}) +
        #        exp(\tilde{beta}_{3->4, 2->3}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #   - \tilde{\alpha}_{2->1, 3->2} =
        #     = \tilde{\alpha}_{3->2, 3->2} =
        #     = exp(\tilde{beta}_{3->2, 3->2}) /
        #       (exp(\tilde{beta}_{2->1, 3->2}) +
        #        exp(\tilde{beta}_{3->2, 3->2}) +
        #        exp(\tilde{beta}_{4->3, 3->2}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.3398941
        #   - \tilde{\alpha}_{4->3, 3->2} =
        #     = exp(\tilde{beta}_{4->3, 3->2}) /
        #       (exp(\tilde{beta}_{2->1, 3->2}) +
        #        exp(\tilde{beta}_{3->2, 3->2}) +
        #        exp(\tilde{beta}_{4->3, 3->2}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.3202119
        #   - \tilde{\alpha}_{2->3, 3->4} =
        #     = \tilde{\alpha}_{3->4, 3->4} =
        #     = \tilde{\alpha}_{4->5, 3->4} =
        #     = exp(\tilde{beta}_{2->3, 3->4}) /
        #       (exp(\tilde{beta}_{2->3, 3->4}) +
        #        exp(\tilde{beta}_{3->4, 3->4}) +
        #        exp(\tilde{beta}_{4->5, 3->4}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #   - \tilde{\alpha}_{3->2, 4->3} =
        #     = \tilde{\alpha}_{5->4, 4->3} =
        #     = exp(\tilde{beta}_{3->2, 4->3}) /
        #       (exp(\tilde{beta}_{3->2, 4->3}) +
        #        exp(\tilde{beta}_{5->4, 4->3}) +
        #        exp(\tilde{beta}_{4->3, 4->3}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (2 * exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #        exp(5 * pi / 4 + 0.25)) =
        #     ~= 0.3398941
        #   - \tilde{\alpha}_{4->3, 4->3} =
        #     = exp(\tilde{beta}_{4->3, 4->3}) /
        #       (exp(\tilde{beta}_{3->2, 4->3}) +
        #        exp(\tilde{beta}_{5->4, 4->3}) +
        #        exp(\tilde{beta}_{4->3, 4->3}))
        #     = exp(5 * pi / 4 + 0.25) /
        #       (2 * exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #        exp(5 * pi / 4 + 0.25)) =
        #     ~= 0.3202119
        #   - \tilde{\alpha}_{3->4, 4->5} =
        #     = \tilde{\alpha}_{4->5, 4->5} =
        #     = \tilde{\alpha}_{5->6, 4->5} =
        #     = exp(\tilde{beta}_{3->4, 4->5}) /
        #       (exp(\tilde{beta}_{3->4, 4->5}) +
        #        exp(\tilde{beta}_{4->5, 4->5}) +
        #        exp(\tilde{beta}_{5->6, 4->5}) +
        #        exp(\tilde{beta}_{5->1, 4->5}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.2868018
        #   - \tilde{\alpha}_{5->1, 4->5} =
        #     = exp(\tilde{beta}_{5->1, 4->5}) /
        #       (exp(\tilde{beta}_{3->4, 4->5}) +
        #        exp(\tilde{beta}_{4->5, 4->5}) +
        #        exp(\tilde{beta}_{5->6, 4->5}) +
        #        exp(\tilde{beta}_{5->1, 4->5}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1395945
        #   - \tilde{\alpha}_{1->0, 5->1} =
        #     = \tilde{\alpha}_{1->2, 5->1} =
        #     = \tilde{\alpha}_{4->5, 5->1} =
        #     = \tilde{\alpha}_{6->5, 5->1} =
        #     = exp(\tilde{beta}_{1->0, 5->1}) /
        #       (exp(\tilde{beta}_{1->0, 5->1}) +
        #        exp(\tilde{beta}_{1->2, 5->1}) +
        #        exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{5->1, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (4 * exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(3 * pi / 4 + 0.5)) =
        #     ~= 0.2228796
        #   - \tilde{\alpha}_{5->1, 5->1} =
        #     = exp(\tilde{beta}_{5->1, 5->1}) /
        #       (exp(\tilde{beta}_{1->0, 5->1}) +
        #        exp(\tilde{beta}_{1->2, 5->1}) +
        #        exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{5->1, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}))
        #     = exp(3 * pi / 4 + 0.5) /
        #       (4 * exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(3 * pi / 4 + 0.5)) =
        #     ~= 0.1084818
        #   - \tilde{\alpha}_{1->5, 5->4} =
        #     = \tilde{\alpha}_{5->4, 5->4} =
        #     = \tilde{\alpha}_{6->5, 5->4} =
        #     = exp(\tilde{beta}_{1->5, 5->4}) /
        #       (exp(\tilde{beta}_{1->5, 5->4}) +
        #        exp(\tilde{beta}_{4->3, 5->4}) +
        #        exp(\tilde{beta}_{5->4, 5->4}) +
        #        exp(\tilde{beta}_{6->5, 5->4}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.2536723
        #   - \tilde{\alpha}_{4->3, 5->4} =
        #     = exp(\tilde{beta}_{4->3, 5->4}) /
        #       (exp(\tilde{beta}_{1->5, 5->4}) +
        #        exp(\tilde{beta}_{4->3, 5->4}) +
        #        exp(\tilde{beta}_{5->4, 5->4}) +
        #        exp(\tilde{beta}_{6->5, 5->4}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.2389830
        #   - \tilde{\alpha}_{1->5, 5->6} =
        #     = \tilde{\alpha}_{4->5, 5->6} =
        #     = \tilde{\alpha}_{5->6, 5->6} =
        #     = exp(\tilde{beta}_{1->5, 5->6}) /
        #       (exp(\tilde{beta}_{1->5, 5->6}) +
        #        exp(\tilde{beta}_{4->5, 5->6}) +
        #        exp(\tilde{beta}_{5->6, 5->6}) +
        #        exp(\tilde{beta}_{6->7, 5->6}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.2922267
        #   - \tilde{\alpha}_{6->7, 5->6} =
        #     = exp(\tilde{beta}_{6->7, 5->6}) /
        #       (exp(\tilde{beta}_{1->5, 5->6}) +
        #        exp(\tilde{beta}_{4->5, 5->6}) +
        #        exp(\tilde{beta}_{5->6, 5->6}) +
        #        exp(\tilde{beta}_{6->7, 5->6}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #        (3 * exp(pi + 2 / sqrt(3)) +
        #         exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.1233199
        #   - \tilde{\alpha}_{5->1, 6->5} =
        #     = exp(\tilde{beta}_{5->1, 6->5}) /
        #       (exp(\tilde{beta}_{5->1, 6->5}) +
        #        exp(\tilde{beta}_{5->4, 6->5}) +
        #        exp(\tilde{beta}_{6->5, 6->5}) +
        #        exp(\tilde{beta}_{7->6, 6->5}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1395945
        #   - \tilde{\alpha}_{5->4, 6->5} =
        #     = \tilde{\alpha}_{6->5, 6->5} =
        #     = \tilde{\alpha}_{7->6, 6->5} =
        #     = exp(\tilde{beta}_{5->4, 6->5}) /
        #       (exp(\tilde{beta}_{5->1, 6->5}) +
        #        exp(\tilde{beta}_{5->4, 6->5}) +
        #        exp(\tilde{beta}_{6->5, 6->5}) +
        #        exp(\tilde{beta}_{7->6, 6->5}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.2868018
        #   - \tilde{\alpha}_{5->6, 6->7} =
        #     = \tilde{\alpha}_{7->0, 6->7} =
        #     = exp(\tilde{beta}_{5->6, 6->7}) /
        #       (exp(\tilde{beta}_{5->6, 6->7}) +
        #        exp(\tilde{beta}_{6->7, 6->7}) +
        #        exp(\tilde{beta}_{7->0, 6->7}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #       (2 * exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) +
        #        exp(pi / 2 + 1)) =
        #     ~= 0.4128818
        #   - \tilde{\alpha}_{6->7, 6->7} =
        #     = exp(\tilde{beta}_{6->7, 6->7}) /
        #       (exp(\tilde{beta}_{5->6, 6->7}) +
        #        exp(\tilde{beta}_{6->7, 6->7}) +
        #        exp(\tilde{beta}_{7->0, 6->7}))
        #     = exp(pi / 2 + 1) / (2 * exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) +
        #        exp(pi / 2 + 1)) =
        #     ~= 0.1742364
        #   - \tilde{\alpha}_{0->1, 7->0} =
        #     = \tilde{\alpha}_{7->0, 7->0} =
        #     = exp(\tilde{beta}_{0->1, 7->0}) /
        #       (exp(\tilde{beta}_{0->1, 7->0}) +
        #        exp(\tilde{beta}_{6->7, 7->0}) +
        #        exp(\tilde{beta}_{7->0, 7->0}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.4128818
        #   - \tilde{\alpha}_{6->7, 7->0} =
        #     = exp(\tilde{beta}_{6->7, 7->0}) /
        #       (exp(\tilde{beta}_{0->1, 7->0}) +
        #        exp(\tilde{beta}_{6->7, 7->0}) +
        #        exp(\tilde{beta}_{7->0, 7->0}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #        (2 * exp(pi + 2 / sqrt(3)) +
        #         exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.1742364
        #   - \tilde{\alpha}_{0->7, 7->6} =
        #     = \tilde{\alpha}_{6->5, 7->6} =
        #     = \tilde{\alpha}_{7->6, 7->6} =
        #     = exp(\tilde{beta}_{0->7, 7->6}) /
        #       (exp(\tilde{beta}_{0->7, 7->6}) +
        #        exp(\tilde{beta}_{6->5, 7->6}) +
        #        exp(\tilde{beta}_{7->6, 7->6}))
        #     = exp(pi + 2 / sqrt(3)) /
        #       (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #
        # - The output features are then obtained as f_{i->j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->i, i->j} * \tilde{f}_{m->i}) +
        #           sum_n(\tilde{\alpha}_{{j->n, i->j} * \tilde{f}_{j->n}) +
        #           \tilde{\alpha}_{i->j, i->j} * \tilde{f}_{i->j}).
        #    We thus have:
        #    - f_{0->1}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 0->1} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{7->0, 0->1} * \tilde{f}_{7->0} +
        #                   \tilde{\alpha}_{1->2, 0->1} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{1->5, 0->1} * \tilde{f}_{1->5})
        #                 = ReLU((0.25 * 4) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{0->7}^' = ReLU(
        #                   \tilde{\alpha}_{0->7, 0->7} * \tilde{f}_{0->7} +
        #                   \tilde{\alpha}_{1->0, 0->7} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{7->6, 0->7} * \tilde{f}_{7->6})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{1->0}^' = ReLU(
        #                   \tilde{\alpha}_{0->7, 1->0} * \tilde{f}_{0->7} +
        #                   \tilde{\alpha}_{1->0, 1->0} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{2->1, 1->0} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{5->1, 1->0} * \tilde{f}_{5->1})
        #                 = ReLU((0.2868018 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1395945 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                = [4.095263, 4.095263, 4.095263, 4.095263, 4.095263];
        #    - f_{1->2}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 1->2} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{1->2, 1->2} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{5->1, 1->2} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{2->3, 1->2} * \tilde{f}_{2->3})
        #                 = ReLU((0.2868018 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1395945 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                = [4.095263, 4.095263, 4.095263, 4.095263, 4.095263];
        #    - f_{1->5}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 1->5} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{1->5, 1->5} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{2->1, 1->5} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{5->4, 1->5} * \tilde{f}_{5->4} +
        #                   \tilde{\alpha}_{5->6, 1->5} * \tilde{f}_{5->6})
        #                 = ReLU((0.2 * 5) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{2->1}^' = ReLU(
        #                   \tilde{\alpha}_{1->0, 2->1} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{1->5, 2->1} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{2->1, 2->1} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{3->2, 2->1} * \tilde{f}_{3->2})
        #                 = ReLU((0.25 * 4) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{2->3}^' = ReLU(
        #                   \tilde{\alpha}_{1->2, 2->3} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{2->3, 2->3} * \tilde{f}_{2->3} +
        #                   \tilde{\alpha}_{3->4, 2->3} * \tilde{f}_{3->4})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{3->2}^' = ReLU(
        #                   \tilde{\alpha}_{2->1, 3->2} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{3->2, 3->2} * \tilde{f}_{3->2} +
        #                   \tilde{\alpha}_{4->3, 3->2} * \tilde{f}_{4->3})
        #                 = ReLU((0.3398941 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.3202119 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                ~= [4.258092, 4.258092, 4.258092, 4.258092, 4.258092];
        #    - f_{3->4}^' = ReLU(
        #                   \tilde{\alpha}_{2->3, 3->4} * \tilde{f}_{2->3} +
        #                   \tilde{\alpha}_{3->4, 3->4} * \tilde{f}_{3->4} +
        #                   \tilde{\alpha}_{4->5, 3->4} * \tilde{f}_{4->5})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{4->3}^' = ReLU(
        #                   \tilde{\alpha}_{3->2, 4->3} * \tilde{f}_{3->2} +
        #                   \tilde{\alpha}_{4->3, 4->3} * \tilde{f}_{4->3} +
        #                   \tilde{\alpha}_{5->4, 4->3} * \tilde{f}_{5->4})
        #                 = ReLU((0.3398941 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.3202119 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                ~= [4.258092, 4.258092, 4.258092, 4.258092, 4.258092];
        #    - f_{4->5}^' = ReLU(
        #                   \tilde{\alpha}_{3->4, 4->5} * \tilde{f}_{3->4} +
        #                   \tilde{\alpha}_{4->5, 4->5} * \tilde{f}_{4->5} +
        #                   \tilde{\alpha}_{5->1, 4->5} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{5->6, 4->5} * \tilde{f}_{5->6})
        #                 = ReLU((0.2868018 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1395945 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                ~= [4.095263, 4.095263, 4.095263, 4.095263, 4.095263];
        #    - f_{5->1}^' = ReLU(
        #                   \tilde{\alpha}_{1->0, 5->1} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{1->2, 5->1} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{4->5, 5->1} * \tilde{f}_{4->5} +
        #                   \tilde{\alpha}_{5->1, 5->1} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{6->5, 5->1} * \tilde{f}_{6->5})
        #                 = ReLU((0.2228796 * 4) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1084818 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                ~= [4.140070, 4.140070, 4.140070, 4.140070, 4.140070];
        #    - f_{5->4}^' = ReLU(
        #                   \tilde{\alpha}_{1->5, 5->4} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{4->3, 5->4} * \tilde{f}_{4->3} +
        #                   \tilde{\alpha}_{5->4, 5->4} * \tilde{f}_{5->4} +
        #                   \tilde{\alpha}_{6->5, 5->4} * \tilde{f}_{6->5})
        #                 = ReLU((0.2536723 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.2389830 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                ~= [4.267782, 4.267782, 4.267782, 4.267782, 4.267782];
        #    - f_{5->6}^' = ReLU(
        #                   \tilde{\alpha}_{1->5, 5->6} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{4->5, 5->6} * \tilde{f}_{4->5} +
        #                   \tilde{\alpha}_{5->6, 5->6} * \tilde{f}_{5->6} +
        #                   \tilde{\alpha}_{6->7, 5->6} * \tilde{f}_{6->7})
        #                 = ReLU((0.2922267 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1233199 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1]) =
        #                ~= [4.083505, 4.083505, 4.083505, 4.083505, 4.083505];
        #    - f_{6->5}^' = ReLU(
        #                   \tilde{\alpha}_{5->1, 6->5} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{5->4, 6->5} * \tilde{f}_{5->4} +
        #                   \tilde{\alpha}_{6->5, 6->5} * \tilde{f}_{6->5} +
        #                   \tilde{\alpha}_{7->6, 6->5} * \tilde{f}_{7->6})
        #                 = ReLU((0.2868018 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1395945 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                ~= [4.095263, 4.095263, 4.095263, 4.095263, 4.095263];
        #    - f_{6->7}^' = ReLU(
        #                   \tilde{\alpha}_{5->6, 6->7} * \tilde{f}_{5->6} +
        #                   \tilde{\alpha}_{6->7, 6->7} * \tilde{f}_{6->7} +
        #                   \tilde{\alpha}_{7->0, 6->7} * \tilde{f}_{7->0})
        #                 = ReLU((0.4128818 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1742364 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1]) =
        #                ~= [3.995649, 3.995649, 3.995649, 3.995649, 3.995649];
        #    - f_{7->0}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 7->0} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{7->0, 7->0} * \tilde{f}_{7->0} +
        #                   \tilde{\alpha}_{6->7, 7->0} * \tilde{f}_{6->7})
        #                 = ReLU((0.4128818 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1742364 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1] =
        #                ~= [3.995649, 3.995649, 3.995649, 3.995649, 3.995649];
        #    - f_{7->6}^' = ReLU(
        #                   \tilde{\alpha}_{0->7, 7->6} * \tilde{f}_{0->7} +
        #                   \tilde{\alpha}_{6->5, 7->6} * \tilde{f}_{6->5} +
        #                   \tilde{\alpha}_{7->6, 7->6} * \tilde{f}_{7->6})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)].
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node i->j the quantity
        #   \beta_{i->j} = att_{primal}^T * f_{i->j}^'. We thus have:
        #
        #   - \beta_{0->1} = att_{primal}^T * f_{0->1}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{0->7} = att_{primal}^T * f_{0->7}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{1->0} = att_{primal}^T * f_{1->0}^' =
        #                  = (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) =
        #                  = 4.095263;
        #   - \beta_{1->2} = att_{primal}^T * f_{1->2}^' =
        #                  = (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) =
        #                  = 4.095263;
        #   - \beta_{1->5} = att_{primal}^T * f_{1->5}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{2->1} = att_{primal}^T * f_{2->1}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{2->3} = att_{primal}^T * f_{2->3}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{3->2} = att_{primal}^T * f_{3->2}^' =
        #                  = (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) =
        #                  = 4.258092;
        #   - \beta_{3->4} = att_{primal}^T * f_{3->4}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{4->3} = att_{primal}^T * f_{4->3}^' =
        #                  = (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) =
        #                  = 4.258092;
        #   - \beta_{4->5} = att_{primal}^T * f_{4->5}^' =
        #                  = (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) =
        #                  = 4.095263;
        #   - \beta_{5->1} = att_{primal}^T * f_{5->1}^' =
        #                  = (1 / 5 * 4.140070) + (1 / 5 * 4.140070) +
        #                    (1 / 5 * 4.140070) + (1 / 5 * 4.140070) +
        #                    (1 / 5 * 4.140070) =
        #                  = 4.140070;
        #   - \beta_{5->4} = att_{primal}^T * f_{5->4}^' =
        #                  = (1 / 5 * 4.267782) + (1 / 5 * 4.267782) +
        #                    (1 / 5 * 4.267782) + (1 / 5 * 4.267782) +
        #                    (1 / 5 * 4.267782) =
        #                  = 4.267782;
        #   - \beta_{5->6} = att_{primal}^T * f_{5->6}^' =
        #                  = (1 / 5 * 4.083505) + (1 / 5 * 4.083505) +
        #                    (1 / 5 * 4.083505) + (1 / 5 * 4.083505) +
        #                    (1 / 5 * 4.083505) =
        #                  = 4.083505;
        #   - \beta_{6->5} = att_{primal}^T * f_{6->5}^' =
        #                  = (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) + (1 / 5 * 4.095263) +
        #                    (1 / 5 * 4.095263) =
        #                  = 4.095263;
        #   - \beta_{6->7} = att_{primal}^T * f_{6->7}^' =
        #                  = (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) =
        #                  = 3.995649;
        #   - \beta_{7->0} = att_{primal}^T * f_{7->0}^' =
        #                  = (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) =
        #                  = 3.995649.
        #   - \beta_{7->6} = att_{primal}^T * f_{7->6}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3).
        #
        #   Then LeakyReLU is applied (with no effect, since \beta_{i->j} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                     exp(\beta_{7->0})) =
        #                   = exp(4.095263) / (exp(4.095263) + exp(3.995649)) =
        #                   ~= 0.5248829;
        #   - \alpha_{7->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                     exp(\beta_{7->0})) =
        #                   = exp(3.995649) / (exp(4.095263) + exp(3.995649)) =
        #                   ~= 0.4751171;
        #   - \alpha_{0->1} = exp(\beta_{0->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(4.140070)) =
        #                   ~= 0.3502175;
        #   - \alpha_{2->1} = exp(\beta_{2->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(4.140070)) =
        #                   ~= 0.3502175;
        #   - \alpha_{5->1} = exp(\beta_{5->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(4.140070) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(4.140070)) =
        #                   ~= 0.2995650;
        #   - \alpha_{1->2} = exp(\beta_{1->2}) / (exp(\beta_{1->2} +
        #                     exp(\beta_{3->2})) =
        #                   = exp(4.095263) / (exp(4.095263) + exp(4.258092)) =
        #                   ~= 0.4593825;
        #   - \alpha_{3->2} = exp(\beta_{3->2}) / (exp(\beta_{1->2} +
        #                     exp(\beta_{3->2})) =
        #                   = exp(4.258092) / (exp(4.095263) + exp(4.258092)) =
        #                   ~= 0.5406175;
        #   - \alpha_{2->3} = exp(\beta_{2->3}) / (exp(\beta_{2->3} +
        #                     exp(\beta_{4->3})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.258092)) =
        #                   = 0.5095491;
        #   - \alpha_{4->3} = exp(\beta_{4->3}) / (exp(\beta_{2->3} +
        #                     exp(\beta_{4->3})) =
        #                   = exp(4.258092) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.258092)) =
        #                   = 0.4904509;
        #   - \alpha_{3->4} = exp(\beta_{3->4}) / (exp(\beta_{3->4} +
        #                     exp(\beta_{5->4})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.267782)) =
        #                   = 0.5071273;
        #   - \alpha_{5->4} = exp(\beta_{5->4}) / (exp(\beta_{3->4} +
        #                     exp(\beta_{5->4})) =
        #                   = exp(4.267782) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.267782)) =
        #                   = 0.4928727;
        #   - \alpha_{1->5} = exp(\beta_{1->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.095263) + exp(4.095263)) =
        #                   = 0.3793950;
        #   - \alpha_{4->5} = exp(\beta_{4->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(4.095263) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.095263) + exp(4.095263)) =
        #                   = 0.3103025;
        #   - \alpha_{6->5} = exp(\beta_{6->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(4.095263) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.095263) + exp(4.095263)) =
        #                   = 0.3103025;
        #   - \alpha_{5->6} = exp(\beta_{5->6}) / (exp(\beta_{5->6} +
        #                     exp(\beta_{7->6})) =
        #                   = exp(4.083505) / (exp(4.083505) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.4470028;
        #   - \alpha_{7->6} = exp(\beta_{7->6}) / (exp(\beta_{5->6} +
        #                     exp(\beta_{7->6})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(4.083505) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5529972;
        #   - \alpha_{0->7} = exp(\beta_{0->7}) / (exp(\beta_{0->7} +
        #                     exp(\beta_{6->7})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(3.995649)) =
        #                   = 0.5746000;
        #   - \alpha_{6->7} = exp(\beta_{6->7}) / (exp(\beta_{0->7} +
        #                     exp(\beta_{6->7})) =
        #                   = exp(3.995649) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(3.995649)) =
        #                   = 0.4254000;
        #
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.5248829 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4751171 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(0.3502175 * [6., 6., 6.] +
        #                   (0.3502175 + 0.2995650) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            ~= [2.1825278, 2.1825278, 2.1825278];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.4593825 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5406175 * [0.5, 0.5, 0.5]) =
        #            ~= [0.3277316, 0.3277316, 0.3277316];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(0.5095491 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4904509 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(0.5071273 * [0.5, 0.5, 0.5] +
        #                   0.4928727 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [0.3151727, 0.3151727, 0.3151727];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU((0.3793950 + 0.3103025 + 0.3103025) * [1 / 8,
        #               1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(0.4470028 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5529972 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(0.5746000 * [6., 6., 6.] +
        #                   0.4254000 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [3.500775, 3.500775, 3.500775];
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 2.1825278, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3277316, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3151727, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 3.500775, 5)

        # Dual features.
        dual_node_to_feature = {
            (0, 1): np.pi + 2 / np.sqrt(3),
            (0, 7): np.pi + 2 / np.sqrt(3),
            (1, 0): 4.095263,
            (1, 2): 4.095263,
            (1, 5): np.pi + 2 / np.sqrt(3),
            (2, 1): np.pi + 2 / np.sqrt(3),
            (2, 3): np.pi + 2 / np.sqrt(3),
            (3, 2): 4.258092,
            (3, 4): np.pi + 2 / np.sqrt(3),
            (4, 3): 4.258092,
            (4, 5): 4.095263,
            (5, 1): 4.140070,
            (5, 4): 4.267782,
            (5, 6): 4.083505,
            (6, 5): 4.095263,
            (6, 7): 3.995649,
            (7, 0): 3.995649,
            (7, 6): np.pi + 2 / np.sqrt(3)
        }

        for dual_node, dual_node_feature in dual_node_to_feature.items():
            dual_node_idx = graph_creator.primal_edge_to_dual_node_idx[
                dual_node]
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), dual_node_feature, 5)

    def test_simple_mesh_config_B_features_not_from_dual_with_self_loops(self):
        # - Dual-graph configuration B.
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=False)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last two features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios both equal to 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i->j} of the dual node i->j
        #   becomes \tilde{f}_{i->j} = f_{i->j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->i)->(i->j), (j->n)->(i->j) - with m neighbor of i and n
        #   neighbor of j - the quantities (|| indicates concatenation):
        #   * \tilde{\beta}_{m->i, i->j} = att_{dual}^T *
        #     (\tilde{f}_{m->i} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 2 / sqrt(3).
        #   * \tilde{\beta}_{{j->n, i->j} = att_{dual}^T *
        #     (\tilde{f}_{j->n} || \tilde{f}_{i->j}) =
        #     = pi + 2 / sqrt(3), by the symmetry of the dual features.
        #   NOTE: There are no self-loops in the dual graph.
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 2 / sqrt(3) > 0). Then, compute the softmax over the (m->i)'s
        #   and (j->n)'s neighboring nodes of (i->j), including also i->j.
        #   Since all \tilde{\beta}_{i->j}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{m->i, i->j} and
        #   \tilde{\alpha}_{j->n, i->j} are simply
        #   1 / #(neighboring nodes (m->i) + neighboring nodes (j->n) + 1 (for
        #   self-loop)). Therefore:
        #   - \tilde{\alpha}_{m->0, 0->1} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{1->n, 0->1} = 1 / 3 for m in {2, 5};
        #   - \tilde{\alpha}_{m->0, 0->7} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{7->n, 0->7} = 1 / 2 for m in {6};
        #   - \tilde{\alpha}_{m->1, 1->0} = 1 / 3 for m in {2, 5};
        #   - \tilde{\alpha}_{0->n, 1->0} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{m->1, 1->2} = 1 / 3 for m in {0, 5};
        #   - \tilde{\alpha}_{2->n, 1->2} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 4 for m in {0, 2};
        #   - \tilde{\alpha}_{5->n, 1->5} = 1 / 4 for m in {4, 6};
        #   - \tilde{\alpha}_{m->2, 2->1} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{1->n, 2->1} = 1 / 3 for m in {0, 5};
        #   - \tilde{\alpha}_{m->2, 2->3} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{3->n, 2->3} = 1 / 2 for m in {4};
        #   - \tilde{\alpha}_{m->3, 3->2} = 1 / 2 for m in {4};
        #   - \tilde{\alpha}_{2->n, 3->2} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{m->3, 3->4} = 1 / 2 for m in {2};
        #   - \tilde{\alpha}_{4->n, 3->4} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{m->4, 4->3} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{3->n, 4->3} = 1 / 2 for m in {2};
        #   - \tilde{\alpha}_{m->4, 4->5} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{5->n, 4->5} = 1 / 3 for m in {1, 6};
        #   - \tilde{\alpha}_{m->5, 5->1} = 1 / 4 for m in {4, 6};
        #   - \tilde{\alpha}_{1->n, 5->1} = 1 / 4 for m in {0, 2};
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 3 for m in {1, 6};
        #   - \tilde{\alpha}_{4->n, 5->4} = 1 / 3 for m in {3};
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 3 for m in {1, 4};
        #   - \tilde{\alpha}_{6->n, 5->6} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{m->6, 6->5} = 1 / 3 for m in {7};
        #   - \tilde{\alpha}_{5->n, 6->5} = 1 / 3 for m in {1, 4};
        #   - \tilde{\alpha}_{m->6, 6->7} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{7->n, 6->7} = 1 / 2 for m in {0};
        #   - \tilde{\alpha}_{m->7, 7->0} = 1 / 2 for m in {6};
        #   - \tilde{\alpha}_{0->n, 7->0} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{m->7, 7->6} = 1 / 2 for m in {0};
        #   - \tilde{\alpha}_{6->n, 7->6} = 1 / 2 for m in {5};
        #
        #  - The output features are then obtained as f_{i->j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->i, i->j} * \tilde{f}_{m->i}) +
        #           sum_n(\tilde{\alpha}_{j->n, i->j} * \tilde{f}_{j->n})) =
        #    = ReLU(\tilde{f}_{i->j} * sum_m(\tilde{\alpha}_{m->i, i->j}) +
        #           sum_n(\tilde{\alpha}_{j->n, i->j}) =
        #    = ReLU(\tilde{f}_{i->j}) =
        #    = \tilde{f}_{i->j},
        #    where the third-last equality holds since the \tilde{f}_{m->i} =
        #    \tilde{f}_{j->n} = \tilde{f}_{i->j} for all valid i, j, m, n (cf.
        #    above), the second-last holds since the sum all the attention
        #    coefficients over the neighborhood of each dual node is 1 by
        #    construction, and the last one holds because \tilde{f}_{i->j} > 0
        #    for all valid i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node {i, j} the quantity
        #   \beta_{i->j} = att_{primal}^T * f_{i->j}^' =
        #   = (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 2 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node {i, j}.
        #   Since all \beta_{i->j}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{i->j} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{m, 0} = 1 / 2     for m in {1, 7};
        #   - \alpha_{m, 1} = 1 / 3     for m in {0, 2, 5};
        #   - \alpha_{m, 2} = 1 / 2     for m in {1, 3};
        #   - \alpha_{m, 3} = 1 / 2     for m in {2, 4};
        #   - \alpha_{m, 4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{m, 5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{m, 6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{m, 7} = 1 / 2     for m in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j, i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j, i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes 6->7, 5->1 and 4->3.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        dual_idx_51 = graph_creator.primal_edge_to_dual_node_idx[(5, 1)]
        dual_idx_43 = graph_creator.primal_edge_to_dual_node_idx[(4, 3)]
        dual_graph.x[dual_idx_67, :] = torch.Tensor([np.pi / 2, 1., 0., 0.])
        dual_graph.x[dual_idx_51, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0., 0.])
        dual_graph.x[dual_idx_43, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0., 0.])
        assert (dual_idx_67 !=
                graph_creator.primal_edge_to_dual_node_idx[(7, 6)])
        assert (dual_idx_51 !=
                graph_creator.primal_edge_to_dual_node_idx[(1, 5)])
        assert (dual_idx_43 !=
                graph_creator.primal_edge_to_dual_node_idx[(3, 4)])
        # - As previously, to simplify computations, we manually set the last
        #   two features of each dual node (i.e., the edge-to-previous-edge-
        #   and edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node i->j           Nodes (m->i)/(j->n) with edges
        #                         (m->i)->(i->j) or (j->n)->(i->j)
        #   _____________________________________________________
        #       0->1            1->2, 1->5, 7->0
        #       0->7            1->0, 7->6
        #       1->0            0->7, 2->1, 5->1
        #       1->2            1->0, 2->3, 5->1
        #       1->5            0->1, 2->1, 5->4, 5->6
        #       2->1            1->0, 3->2, 1->5
        #       2->3            1->2, 3->4
        #       3->2            2->1, 4->3
        #       3->4            2->3, 4->5
        #       4->3            3->2, 5->4
        #       4->5            3->4, 5->1, 5->6
        #       5->1            1->0, 1->2, 4->5, 6->5
        #       5->4            1->5, 4->3, 6->5
        #       5->6            1->5, 4->5, 6->7
        #       6->5            5->1, 5->4, 7->6
        #       6->7            5->6, 7->0
        #       7->0            0->1, 6->7
        #       7->6            0->7, 6->5
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratios 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{i->j} of the dual node i->j
        #   becomes \tilde{f}_{i->j} = f_{i->j} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)], for all i->j not in
        #   {6->7, 5->1, 4->3}.
        #   We have also:
        #   - \tilde{f}_{6->7} = [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #       pi / 2 + 1, pi / 2 + 1];
        #   - \tilde{f}_{5->1} = [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #       3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5];
        #   - \tilde{f}_{4->3} = [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #       5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25];
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->i)->(i->j) or (j->n)->(i->j) the quantities (|| indicates
        #   concatenation):
        #   * \tilde{\beta}_{j->n, i->j} = att_{dual}^T *
        #     (\tilde{f}_{j->n} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 2 / sqrt(3), for all j->n, i->j not in {6->7, 5->1, 4->3}.
        #   * \tilde{\beta}_{m->i, i->j} = att_{dual}^T *
        #     (\tilde{f}_{m->i} || \tilde{f}_{i->j}) =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = pi + 4 / sqrt(3), for all j->n, i->j not in {6->7, 5->1, 4->3}
        #   Likewise, we have also:
        #   - \tilde{beta}_{5->6, 6->7} =
        #     = \tilde{beta}_{7->0, 6->7} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 0.5;
        #   - \tilde{beta}_{6->7, 5->6} =
        #     = \tilde{beta}_{6->7, 7->0} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) +
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 0.5;
        #   - \tilde{beta}_{1->0, 5->1} =
        #     = \tilde{beta}_{1->2, 5->1} =
        #     = \tilde{beta}_{4->5, 5->1} =
        #     = \tilde{beta}_{6->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->1, 1->0} =
        #     = \tilde{beta}_{5->1, 1->2} =
        #     = \tilde{beta}_{5->1, 4->5} =
        #     = \tilde{beta}_{5->1, 6->5} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 4 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3)))) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{3->2, 4->3} =
        #     = \tilde{beta}_{5->4, 4->3} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   - \tilde{beta}_{4->3, 3->2} =
        #     = \tilde{beta}_{4->3, 5->4} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   NOTE: There are no self-loops in the dual graph.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{m->i, i->j} > 0 and \tilde{beta}_{j->n, i->j} > 0 for
        #   all i, j, m, n). Then,
        #   compute the softmax over the neighboring nodes m->i and j->n. We
        #   have (cf. adjacency matrix + self-loops):
        #   - \tilde{\alpha}_{1->2, 0->1} =
        #     = \tilde{\alpha}_{1->5, 0->1} =
        #     = \tilde{\alpha}_{7->0, 0->1} =
        #     = exp(\tilde{beta}_{1->2, 0->1}) /
        #       (exp(\tilde{beta}_{1->2, 0->1}) +
        #        exp(\tilde{beta}_{1->5, 0->1}) +
        #        exp(\tilde{beta}_{7->0, 0->1}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #   - \tilde{\alpha}_{1->0, 0->7} =
        #     = \tilde{\alpha}_{7->6, 0->7} =
        #     = exp(\tilde{beta}_{1->0, 0->7}) /
        #       (exp(\tilde{beta}_{1->0, 0->7}) +
        #        exp(\tilde{beta}_{7->6, 0->7}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3))) =
        #     = 0.5
        #   - \tilde{\alpha}_{0->7, 1->0} =
        #     = \tilde{\alpha}_{2->1, 1->0} =
        #     = exp(\tilde{beta}_{0->7, 1->0}) /
        #       (exp(\tilde{beta}_{0->7, 1->0}) +
        #        exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.4021348
        #   - \tilde{\alpha}_{5->1, 1->0} =
        #     = exp(\tilde{beta}_{5->1, 1->0}) /
        #       (exp(\tilde{beta}_{0->7, 1->0}) +
        #        exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1957304
        #   - \tilde{\alpha}_{0->1, 1->2} =
        #     = \tilde{\alpha}_{2->3, 1->2} =
        #     = exp(\tilde{beta}_{0->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{2->3, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.4021348
        #   - \tilde{\alpha}_{5->1, 1->2} =
        #     = exp(\tilde{beta}_{5->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{2->3, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1957304
        #   - \tilde{\alpha}_{0->1, 1->5} =
        #     = \tilde{\alpha}_{2->1, 1->5} =
        #     = \tilde{\alpha}_{5->4, 1->5} =
        #     = \tilde{\alpha}_{5->6, 1->5} =
        #     = exp(\tilde{beta}_{0->1, 1->5}) /
        #       (exp(\tilde{beta}_{0->1, 1->5}) +
        #        exp(\tilde{beta}_{2->1, 1->5}) +
        #        exp(\tilde{beta}_{5->4, 1->5}) +
        #        exp(\tilde{beta}_{5->6, 1->5}) +
        #     = 0.25
        #   - \tilde{\alpha}_{1->0, 2->1} =
        #     = \tilde{\alpha}_{1->5, 2->1} =
        #     = \tilde{\alpha}_{3->2, 2->1} =
        #     = exp(\tilde{beta}_{1->0, 2->1}) /
        #       (exp(\tilde{beta}_{1->0, 2->1}) +
        #        exp(\tilde{beta}_{1->5, 2->1}) +
        #        exp(\tilde{beta}_{3->2, 2->1}))
        #     = exp(pi + 2 / sqrt(3)) / (3 * exp(pi + 2 / sqrt(3))) =
        #     = 1 / 3
        #   - \tilde{\alpha}_{1->2, 2->3} =
        #     = \tilde{\alpha}_{3->4, 2->3} =
        #     = exp(\tilde{beta}_{1->2, 2->3}) /
        #       (exp(\tilde{beta}_{1->2, 2->3}) +
        #        exp(\tilde{beta}_{3->4, 2->3}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3))) =
        #     = 0.5
        #   - \tilde{\alpha}_{2->1, 3->2} =
        #     = exp(\tilde{beta}_{3->2, 3->2}) /
        #       (exp(\tilde{beta}_{2->1, 3->2}) +
        #        exp(\tilde{beta}_{4->3, 3->2}))
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.5149084
        #   - \tilde{\alpha}_{4->3, 3->2} =
        #     = exp(\tilde{beta}_{4->3, 3->2}) /
        #       (exp(\tilde{beta}_{2->1, 3->2}) +
        #        exp(\tilde{beta}_{4->3, 3->2}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.4850916
        #   - \tilde{\alpha}_{2->3, 3->4} =
        #     = \tilde{\alpha}_{4->5, 3->4} =
        #     = exp(\tilde{beta}_{2->3, 3->4}) /
        #       (exp(\tilde{beta}_{2->3, 3->4}) +
        #        exp(\tilde{beta}_{4->5, 3->4}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3))) =
        #     = 0.5
        #   - \tilde{\alpha}_{3->2, 4->3} =
        #     = \tilde{\alpha}_{5->4, 4->3} =
        #     = exp(\tilde{beta}_{3->2, 4->3}) /
        #       (exp(\tilde{beta}_{3->2, 4->3}) +
        #        exp(\tilde{beta}_{5->4, 4->3}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (2 * exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     = 0.5
        #   - \tilde{\alpha}_{3->4, 4->5} =
        #     = \tilde{\alpha}_{5->6, 4->5} =
        #     = exp(\tilde{beta}_{3->4, 4->5}) /
        #       (exp(\tilde{beta}_{3->4, 4->5}) +
        #        exp(\tilde{beta}_{5->6, 4->5}) +
        #        exp(\tilde{beta}_{5->1, 4->5}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.4021348
        #   - \tilde{\alpha}_{5->1, 4->5} =
        #     = exp(\tilde{beta}_{5->1, 4->5}) /
        #       (exp(\tilde{beta}_{3->4, 4->5}) +
        #        exp(\tilde{beta}_{5->6, 4->5}) +
        #        exp(\tilde{beta}_{5->1, 4->5}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1957304
        #   - \tilde{\alpha}_{1->0, 5->1} =
        #     = \tilde{\alpha}_{1->2, 5->1} =
        #     = \tilde{\alpha}_{4->5, 5->1} =
        #     = \tilde{\alpha}_{6->5, 5->1} =
        #     = exp(\tilde{beta}_{1->0, 5->1}) /
        #       (exp(\tilde{beta}_{1->0, 5->1}) +
        #        exp(\tilde{beta}_{1->2, 5->1}) +
        #        exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (4 * exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     = 0.25
        #   - \tilde{\alpha}_{1->5, 5->4} =
        #     = \tilde{\alpha}_{6->5, 5->4} =
        #     = exp(\tilde{beta}_{1->5, 5->4}) /
        #       (exp(\tilde{beta}_{1->5, 5->4}) +
        #        exp(\tilde{beta}_{4->3, 5->4}) +
        #        exp(\tilde{beta}_{6->5, 5->4}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.3398941
        #   - \tilde{\alpha}_{4->3, 5->4} =
        #     = exp(\tilde{beta}_{4->3, 5->4}) /
        #       (exp(\tilde{beta}_{1->5, 5->4}) +
        #        exp(\tilde{beta}_{4->3, 5->4}) +
        #        exp(\tilde{beta}_{6->5, 5->4}))
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8)) =
        #     ~= 0.3202119
        #   - \tilde{\alpha}_{1->5, 5->6} =
        #     = \tilde{\alpha}_{4->5, 5->6} =
        #     = exp(\tilde{beta}_{1->5, 5->6}) /
        #       (exp(\tilde{beta}_{1->5, 5->6}) +
        #        exp(\tilde{beta}_{4->5, 5->6}) +
        #        exp(\tilde{beta}_{6->7, 5->6}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.4128818
        #   - \tilde{\alpha}_{6->7, 5->6} =
        #     = exp(\tilde{beta}_{6->7, 5->6}) /
        #       (exp(\tilde{beta}_{1->5, 5->6}) +
        #        exp(\tilde{beta}_{4->5, 5->6}) +
        #        exp(\tilde{beta}_{6->7, 5->6}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #        (2 * exp(pi + 2 / sqrt(3)) +
        #         exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.1742364
        #   - \tilde{\alpha}_{5->1, 6->5} =
        #     = exp(\tilde{beta}_{5->1, 6->5}) /
        #       (exp(\tilde{beta}_{5->1, 6->5}) +
        #        exp(\tilde{beta}_{5->4, 6->5}) +
        #        exp(\tilde{beta}_{7->6, 6->5}))
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.1957304
        #   - \tilde{\alpha}_{5->4, 6->5} =
        #     = \tilde{\alpha}_{7->6, 6->5} =
        #     = exp(\tilde{beta}_{5->4, 6->5}) /
        #       (exp(\tilde{beta}_{5->1, 6->5}) +
        #        exp(\tilde{beta}_{5->4, 6->5}) +
        #        exp(\tilde{beta}_{7->6, 6->5}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3)) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.4021348
        #   - \tilde{\alpha}_{5->6, 6->7} =
        #     = \tilde{\alpha}_{7->0, 6->7} =
        #     = exp(\tilde{beta}_{5->6, 6->7}) /
        #       (exp(\tilde{beta}_{5->6, 6->7}) +
        #        exp(\tilde{beta}_{7->0, 6->7}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #       (2 * exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     = 0.5
        #   - \tilde{\alpha}_{0->1, 7->0} =
        #     = exp(\tilde{beta}_{0->1, 7->0}) /
        #       (exp(\tilde{beta}_{0->1, 7->0}) +
        #        exp(\tilde{beta}_{6->7, 7->0}))
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #        exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.7032346
        #   - \tilde{\alpha}_{6->7, 7->0} =
        #     = exp(\tilde{beta}_{6->7, 7->0}) /
        #       (exp(\tilde{beta}_{0->1, 7->0}) +
        #        exp(\tilde{beta}_{6->7, 7->0}))
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 0.5) /
        #        (exp(pi + 2 / sqrt(3)) +
        #         exp(3 * pi / 4 + 1 / sqrt(3) + 0.5)) =
        #     ~= 0.2967654
        #   - \tilde{\alpha}_{0->7, 7->6} =
        #     = \tilde{\alpha}_{6->5, 7->6} =
        #     = exp(\tilde{beta}_{0->7, 7->6}) /
        #       (exp(\tilde{beta}_{0->7, 7->6}) +
        #        exp(\tilde{beta}_{6->5, 7->6}))
        #     = exp(pi + 2 / sqrt(3)) / (2 * exp(pi + 2 / sqrt(3))) =
        #     = 0.5
        #
        # - The output features are then obtained as f_{i->j}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->i, i->j} * \tilde{f}_{m->i}) +
        #           sum_n(\tilde{\alpha}_{{j->n, i->j} * \tilde{f}_{j->n})).
        #    We thus have:
        #    - f_{0->1}^' = ReLU(
        #                   \tilde{\alpha}_{7->0, 0->1} * \tilde{f}_{7->0} +
        #                   \tilde{\alpha}_{1->2, 0->1} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{1->5, 0->1} * \tilde{f}_{1->5})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{0->7}^' = ReLU(
        #                   \tilde{\alpha}_{1->0, 0->7} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{7->6, 0->7} * \tilde{f}_{7->6})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{1->0}^' = ReLU(
        #                   \tilde{\alpha}_{0->7, 1->0} * \tilde{f}_{0->7} +
        #                   \tilde{\alpha}_{2->1, 1->0} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{5->1, 1->0} * \tilde{f}_{5->1})
        #                 = ReLU((0.4021348 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1957304 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                = [4.014422, 4.014422, 4.014422, 4.014422, 4.014422];
        #    - f_{1->2}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 1->2} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{5->1, 1->2} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{2->3, 1->2} * \tilde{f}_{2->3})
        #                 = ReLU((0.4021348 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1957304 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                = [4.014422, 4.014422, 4.014422, 4.014422, 4.014422];
        #    - f_{1->5}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 1->5} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{2->1, 1->5} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{5->4, 1->5} * \tilde{f}_{5->4} +
        #                   \tilde{\alpha}_{5->6, 1->5} * \tilde{f}_{5->6})
        #                 = ReLU((0.25 * 4) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{2->1}^' = ReLU(
        #                   \tilde{\alpha}_{1->0, 2->1} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{1->5, 2->1} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{3->2, 2->1} * \tilde{f}_{3->2})
        #                 = ReLU((1 / 3 * 3) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{2->3}^' = ReLU(
        #                   \tilde{\alpha}_{1->2, 2->3} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{3->4, 2->3} * \tilde{f}_{3->4})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{3->2}^' = ReLU(
        #                   \tilde{\alpha}_{2->1, 3->2} * \tilde{f}_{2->1} +
        #                   \tilde{\alpha}_{4->3, 3->2} * \tilde{f}_{4->3})
        #                 = ReLU(0.5149084 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.4850916 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                ~= [4.238421, 4.238421, 4.238421, 4.238421, 4.238421];
        #    - f_{3->4}^' = ReLU(
        #                   \tilde{\alpha}_{2->3, 3->4} * \tilde{f}_{2->3} +
        #                   \tilde{\alpha}_{4->5, 3->4} * \tilde{f}_{4->5})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{4->3}^' = ReLU(
        #                   \tilde{\alpha}_{3->2, 4->3} * \tilde{f}_{3->2} +
        #                   \tilde{\alpha}_{5->4, 4->3} * \tilde{f}_{5->4})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{4->5}^' = ReLU(
        #                   \tilde{\alpha}_{3->4, 4->5} * \tilde{f}_{3->4} +
        #                   \tilde{\alpha}_{5->1, 4->5} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{5->6, 4->5} * \tilde{f}_{5->6})
        #                 = ReLU((0.4021348 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1957304 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                ~= [4.014422, 4.014422, 4.014422, 4.014422, 4.014422];
        #    - f_{5->1}^' = ReLU(
        #                   \tilde{\alpha}_{1->0, 5->1} * \tilde{f}_{1->0} +
        #                   \tilde{\alpha}_{1->2, 5->1} * \tilde{f}_{1->2} +
        #                   \tilde{\alpha}_{4->5, 5->1} * \tilde{f}_{4->5} +
        #                   \tilde{\alpha}_{6->5, 5->1} * \tilde{f}_{6->5})
        #                 = ReLU((0.25 * 4) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{5->4}^' = ReLU(
        #                   \tilde{\alpha}_{1->5, 5->4} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{4->3, 5->4} * \tilde{f}_{4->3} +
        #                   \tilde{\alpha}_{6->5, 5->4} * \tilde{f}_{6->5})
        #                 = ReLU((0.3398941 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.3202119 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                ~= [4.258092, 4.258092, 4.258092, 4.258092, 4.258092];
        #    - f_{5->6}^' = ReLU(
        #                   \tilde{\alpha}_{1->5, 5->6} * \tilde{f}_{1->5} +
        #                   \tilde{\alpha}_{4->5, 5->6} * \tilde{f}_{4->5} +
        #                   \tilde{\alpha}_{6->7, 5->6} * \tilde{f}_{6->7})
        #                 = ReLU((0.4128818 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1742364 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1]) =
        #                ~= [3.995649, 3.995649, 3.995649, 3.995649, 3.995649];
        #    - f_{6->5}^' = ReLU(
        #                   \tilde{\alpha}_{5->1, 6->5} * \tilde{f}_{5->1} +
        #                   \tilde{\alpha}_{5->4, 6->5} * \tilde{f}_{5->4} +
        #                   \tilde{\alpha}_{7->6, 6->5} * \tilde{f}_{7->6})
        #                 = ReLU((0.4021348 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.1957304 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                ~= [4.014422, 4.014422, 4.014422, 4.014422, 4.014422];
        #    - f_{6->7}^' = ReLU(
        #                   \tilde{\alpha}_{5->6, 6->7} * \tilde{f}_{5->6} +
        #                   \tilde{\alpha}_{7->0, 6->7} * \tilde{f}_{7->0})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)];
        #    - f_{7->0}^' = ReLU(
        #                   \tilde{\alpha}_{0->1, 7->0} * \tilde{f}_{0->1} +
        #                   \tilde{\alpha}_{6->7, 7->0} * \tilde{f}_{6->7})
        #                 = ReLU(0.7032346 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.2967654 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1] =
        #                ~= [3.784225, 3.784225, 3.784225, 3.784225, 3.784225];
        #    - f_{7->6}^' = ReLU(
        #                   \tilde{\alpha}_{0->7, 7->6} * \tilde{f}_{0->7} +
        #                   \tilde{\alpha}_{6->5, 7->6} * \tilde{f}_{6->5})
        #                 = ReLU((0.5 * 2) * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)]) =
        #                = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)].
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node i->j the quantity
        #   \beta_{i->j} = att_{primal}^T * f_{i->j}^'. We thus have:
        #
        #   - \beta_{0->1} = att_{primal}^T * f_{0->1}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{0->7} = att_{primal}^T * f_{0->7}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{1->0} = att_{primal}^T * f_{1->0}^' =
        #                  = (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) =
        #                  = 4.014422;
        #   - \beta_{1->2} = att_{primal}^T * f_{1->2}^' =
        #                  = (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) =
        #                  = 4.014422;
        #   - \beta_{1->5} = att_{primal}^T * f_{1->5}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{2->1} = att_{primal}^T * f_{2->1}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{2->3} = att_{primal}^T * f_{2->3}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{3->2} = att_{primal}^T * f_{3->2}^' =
        #                  = (1 / 5 * 4.238421) + (1 / 5 * 4.238421) +
        #                    (1 / 5 * 4.238421) + (1 / 5 * 4.238421) +
        #                    (1 / 5 * 4.238421) =
        #                  = 4.238421;
        #   - \beta_{3->4} = att_{primal}^T * f_{3->4}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{4->3} = att_{primal}^T * f_{4->3}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{4->5} = att_{primal}^T * f_{4->5}^' =
        #                  = (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) =
        #                  = 4.014422;
        #   - \beta_{5->1} = att_{primal}^T * f_{5->1}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{5->4} = att_{primal}^T * f_{5->4}^' =
        #                  = (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) + (1 / 5 * 4.258092) +
        #                    (1 / 5 * 4.258092) =
        #                  = 4.258092;
        #   - \beta_{5->6} = att_{primal}^T * f_{5->6}^' =
        #                  = (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) + (1 / 5 * 3.995649) +
        #                    (1 / 5 * 3.995649) =
        #                  = 3.995649;
        #   - \beta_{6->5} = att_{primal}^T * f_{6->5}^' =
        #                  = (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) + (1 / 5 * 4.014422) +
        #                    (1 / 5 * 4.014422) =
        #                  = 4.014422;
        #   - \beta_{6->7} = att_{primal}^T * f_{6->7}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3);
        #   - \beta_{7->0} = att_{primal}^T * f_{7->0}^' =
        #                  = (1 / 5 * 3.784225) + (1 / 5 * 3.784225) +
        #                    (1 / 5 * 3.784225) + (1 / 5 * 3.784225) +
        #                    (1 / 5 * 3.784225) =
        #                  = 3.784225.
        #   - \beta_{7->6} = att_{primal}^T * f_{7->6}^' =
        #                  = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) +
        #                    (1 / 5 * (pi + 2 / sqrt(3))) =
        #                  = pi + 2 / sqrt(3).
        #
        #   Then LeakyReLU is applied (with no effect, since \beta_{i->j} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                     exp(\beta_{7->0})) =
        #                   = exp(4.014422) / (exp(4.014422) + exp(3.784225)) =
        #                   ~= 0.5572965;
        #   - \alpha_{7->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                     exp(\beta_{7->0})) =
        #                   = exp(3.784225) / (exp(4.014422) + exp(3.784225)) =
        #                   ~= 0.4427035;
        #   - \alpha_{0->1} = exp(\beta_{0->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(pi + 2 / sqrt(3))) =
        #                   = 1 / 3
        #   - \alpha_{2->1} = exp(\beta_{2->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(pi + 2 / sqrt(3))) =
        #                   = 1 / 3
        #   - \alpha_{5->1} = exp(\beta_{5->1}) / (exp(\beta_{0->1}) +
        #                     exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                     exp(pi + 2 / sqrt(3)) + exp(pi + 2 / sqrt(3))) =
        #                   = 1 / 3
        #   - \alpha_{1->2} = exp(\beta_{1->2}) / (exp(\beta_{1->2} +
        #                     exp(\beta_{3->2})) =
        #                   = exp(4.014422) / (exp(4.014422) + exp(4.238421)) =
        #                   ~= 0.4442332;
        #   - \alpha_{3->2} = exp(\beta_{3->2}) / (exp(\beta_{1->2} +
        #                     exp(\beta_{3->2})) =
        #                   = exp(4.238421) / (exp(4.014422) + exp(4.238421)) =
        #                   ~= 0.5557668;
        #   - \alpha_{2->3} = exp(\beta_{2->3}) / (exp(\beta_{2->3} +
        #                     exp(\beta_{4->3})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5
        #   - \alpha_{4->3} = exp(\beta_{4->3}) / (exp(\beta_{2->3} +
        #                     exp(\beta_{4->3})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5
        #   - \alpha_{3->4} = exp(\beta_{3->4}) / (exp(\beta_{3->4} +
        #                     exp(\beta_{5->4})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.258092)) =
        #                   = 0.5095491;
        #   - \alpha_{5->4} = exp(\beta_{5->4}) / (exp(\beta_{3->4} +
        #                     exp(\beta_{5->4})) =
        #                   = exp(4.258092) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.258092)) =
        #                   = 0.4904509;
        #   - \alpha_{1->5} = exp(\beta_{1->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.014422) + exp(4.014422)) =
        #                   = 0.3986062;
        #   - \alpha_{4->5} = exp(\beta_{4->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(4.014422) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.014422) + exp(4.014422)) =
        #                   = 0.3006969;
        #   - \alpha_{6->5} = exp(\beta_{6->5}) / (exp(\beta_{1->5} +
        #                     exp(\beta_{4->5}) + exp(\beta_{6->5})) =
        #                   = exp(4.014422) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(4.014422) + exp(4.014422)) =
        #                   = 0.3006969;
        #   - \alpha_{5->6} = exp(\beta_{5->6}) / (exp(\beta_{5->6} +
        #                     exp(\beta_{7->6})) =
        #                   = exp(3.995649) / (exp(3.995649) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.4254000;
        #   - \alpha_{7->6} = exp(\beta_{7->6}) / (exp(\beta_{5->6} +
        #                     exp(\beta_{7->6})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(3.995649) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5746000;
        #   - \alpha_{0->7} = exp(\beta_{0->7}) / (exp(\beta_{0->7} +
        #                     exp(\beta_{6->7})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5
        #   - \alpha_{6->7} = exp(\beta_{6->7}) / (exp(\beta_{0->7} +
        #                     exp(\beta_{6->7})) =
        #                   = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                      exp(pi + 2 / sqrt(3))) =
        #                   = 0.5
        #
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.5572965 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.4427035 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(1 / 3 * [6., 6., 6.] +
        #                   (1 / 3 + 1 / 3) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            ~= [25 / 12, 25 / 12, 25 / 12];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.4442332 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5557668 * [0.5, 0.5, 0.5]) =
        #            ~= [0.3334126, 0.3334126, 0.3334126];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(0.5 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(0.5095491 * [0.5, 0.5, 0.5] +
        #                   0.4904509 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [0.3160809, 0.3160809, 0.3160809];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU((0.3986062 + 0.3006969 + 0.3006969) * [1 / 8,
        #               1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(0.4254000 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.5746000 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(0.5 * [6., 6., 6.] +  0.5 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [49 / 16, 49 / 16, 49 / 16];
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))

        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 25 / 12, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3334126, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.3160809, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 49 / 16, 5)

        # Dual features.
        dual_node_to_feature = {
            (0, 1): np.pi + 2 / np.sqrt(3),
            (0, 7): np.pi + 2 / np.sqrt(3),
            (1, 0): 4.014422,
            (1, 2): 4.014422,
            (1, 5): np.pi + 2 / np.sqrt(3),
            (2, 1): np.pi + 2 / np.sqrt(3),
            (2, 3): np.pi + 2 / np.sqrt(3),
            (3, 2): 4.238421,
            (3, 4): np.pi + 2 / np.sqrt(3),
            (4, 3): np.pi + 2 / np.sqrt(3),
            (4, 5): 4.014422,
            (5, 1): np.pi + 2 / np.sqrt(3),
            (5, 4): 4.258092,
            (5, 6): 3.995649,
            (6, 5): 4.014422,
            (6, 7): np.pi + 2 / np.sqrt(3),
            (7, 0): 3.784225,
            (7, 6): np.pi + 2 / np.sqrt(3)
        }
        for dual_node, dual_node_feature in dual_node_to_feature.items():
            dual_node_idx = graph_creator.primal_edge_to_dual_node_idx[
                dual_node]
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), dual_node_feature, 5)

    def test_simple_mesh_config_C_features_not_from_dual_with_self_loops(self):
        # - Dual-graph configuration C.
        single_dual_nodes = False
        undirected_dual_edges = False
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=True)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last two features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratio 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{j->i} of the dual node j->i
        #   becomes \tilde{f}_{j->i} = f_{j->i} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->j)->(j->i) the quantity (|| indicates concatenation)
        #   \tilde{\beta}_{m->j, j->i} = att_{dual}^T * (\tilde{f}_{m->j} ||
        #   \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3).
        #   NOTE: We also have self-loops in the dual graph! Hence, one has:
        #   - \tilde{\beta}_{j->i, j->i} = att_{dual}^T * (\tilde{f}_{j->i} ||
        #     \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all i, j.
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 2 / sqrt(3) > 0). Then, compute the softmax over the (m->j)'s,
        #   neighboring nodes of (j->i), including also (j->i).
        #   Since all \tilde{\beta}_{j->i}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{m->j, j->i} are simply
        #   1 / #(neighboring nodes (m->j) + 1 (for self-loop)). Therefore:
        #   - \tilde{\alpha}_{m->0, 0->1} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{0->1, 0->1} = 1 / 2;
        #   - \tilde{\alpha}_{m->1, 1->0} = 1 / 3 for m in {2, 5};
        #   - \tilde{\alpha}_{1->0, 1->0} = 1 / 3;
        #   - \tilde{\alpha}_{m->0, 0->7} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{0->7, 0->7} = 1 / 2;
        #   - \tilde{\alpha}_{m->7, 7->0} = 1 / 2 for m in {6};
        #   - \tilde{\alpha}_{7->0, 7->0} = 1 / 2;
        #   - \tilde{\alpha}_{m->1, 1->2} = 1 / 3 for m in {0, 5};
        #   - \tilde{\alpha}_{1->2, 1->2} = 1 / 3;
        #   - \tilde{\alpha}_{m->2, 2->1} = 1 / 2 for m in {3};
        #   - \tilde{\alpha}_{2->1, 2->1} = 1 / 2;
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 3 for m in {0, 2};
        #   - \tilde{\alpha}_{1->5, 1->5} = 1 / 3;
        #   - \tilde{\alpha}_{m->5, 5->1} = 1 / 3 for m in {4, 6};
        #   - \tilde{\alpha}_{5->1, 5->1} = 1 / 3;
        #   - \tilde{\alpha}_{m->2, 2->3} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{2->3, 2->3} = 1 / 2;
        #   - \tilde{\alpha}_{m->3, 3->2} = 1 / 2 for m in {4};
        #   - \tilde{\alpha}_{3->2, 3->2} = 1 / 2;
        #   - \tilde{\alpha}_{m->3, 3->4} = 1 / 2 for m in {2};
        #   - \tilde{\alpha}_{3->4, 3->4} = 1 / 2;
        #   - \tilde{\alpha}_{m->4, 4->3} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{4->3, 4->3} = 1 / 2;
        #   - \tilde{\alpha}_{m->4, 4->5} = 1 / 2 for m in {3};
        #   - \tilde{\alpha}_{4->5, 4->5} = 1 / 2;
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 3 for m in {1, 6};
        #   - \tilde{\alpha}_{5->4, 5->4} = 1 / 3;
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 3 for m in {1, 4};
        #   - \tilde{\alpha}_{5->6, 5->6} = 1 / 3;
        #   - \tilde{\alpha}_{m->6, 6->5} = 1 / 2 for m in {7};
        #   - \tilde{\alpha}_{6->5, 6->5} = 1 / 2;
        #   - \tilde{\alpha}_{m->6, 6->7} = 1 / 2 for m in {5};
        #   - \tilde{\alpha}_{6->7, 6->7} = 1 / 2;
        #   - \tilde{\alpha}_{m->7, 7->6} = 1 / 2 for m in {0};
        #   - \tilde{\alpha}_{7->6, 7->6} = 1 / 2.
        #  - The output features are then obtained as f_{j->i}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->j, j->i} * \tilde{f}_{m->j}) +
        #           \tilde{\alpha}_{j->i, j->i} * \tilde{f}_{j->i}) =
        #    = ReLU(\tilde{f}_{n->j} * sum_m(\tilde{\alpha}_{m->j, j->i})) =
        #    = ReLU(\tilde{f}_{n->j}) =
        #    = \tilde{f}_{n->j},      with n->j any of the neighbors of j->i,
        #    where the third-last equality holds since the \tilde{f}_{m->j}'s -
        #    with m->j in the neighborhood of j->i - are all equal and are
        #    also equal to j->i, the second-last holds since the sum all the
        #    attention coefficients over the neighborhood of each dual node
        #    (dual node included) is 1 by construction, and the last one holds
        #    because \tilde{f}_{n->j} > 0 for all valid j, n.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node j->i the quantity
        #   \beta_{j->i} = att_{primal}^T * f_{j->i}^' =
        #   = (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 2 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node j->i.
        #   Since all \beta_{j->i}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{j->i} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{j->0} = 1 / 2     for j in {1, 7};
        #   - \alpha_{j->1} = 1 / 3     for j in {0, 2, 5};
        #   - \alpha_{j->2} = 1 / 2     for j in {1, 3};
        #   - \alpha_{j->3} = 1 / 2     for j in {2, 4};
        #   - \alpha_{j->4} = 1 / 2     for j in {3, 5};
        #   - \alpha_{j->5} = 1 / 3     for j in {1, 4, 6};
        #   - \alpha_{j->6} = 1 / 2     for j in {5, 7};
        #   - \alpha_{j->7} = 1 / 2     for j in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j->i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes 6->7, 5->1 and 4->3.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        dual_idx_51 = graph_creator.primal_edge_to_dual_node_idx[(5, 1)]
        dual_idx_43 = graph_creator.primal_edge_to_dual_node_idx[(4, 3)]
        dual_graph.x[dual_idx_67, :] = torch.Tensor([np.pi / 2, 1., 0., 0.])
        dual_graph.x[dual_idx_51, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0., 0.])
        dual_graph.x[dual_idx_43, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0., 0.])
        # - As previously, to simplify computations, we manually set the last
        #   two features of each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node j->i             Nodes m->j with edges (m->j)->(j->i)
        #   _____________________________________________________
        #       0->1              7->0
        #       1->0              2->1, 5->1
        #       0->7              1->0
        #       7->0              6->7
        #       1->2              0->1, 5->1
        #       2->1              3->2
        #       1->5              0->1, 2->1
        #       5->1              4->5, 6->5
        #       2->3              1->2
        #       3->2              4->3
        #       3->4              2->3
        #       4->3              5->4
        #       4->5              3->4
        #       5->4              1->5, 6->5
        #       5->6              1->5, 4->5
        #       6->5              7->6
        #       6->7              5->6
        #       7->6              0->7
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratio 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{j->i} of the dual node/primal edge
        #   j->i becomes \tilde{f}_{j->i} = f_{j->i} * W_{dual} =
        #   = [pi, 2 / sqrt(3),  0, 0] * [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)], for all j->i not in
        #   {6->7, 5->1, 4->3}.
        #   We have also:
        #   - \tilde{f}_{6->7} = [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #       pi / 2 + 1, pi / 2 + 1];
        #   - \tilde{f}_{5->1} = [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #       3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5];
        #   - \tilde{f}_{4->3} = [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #       5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25];
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->j)->(j->i) the quantity (|| indicates concatenation)
        #   \tilde{\beta}_{m->j, j->i} = att_{dual}^T * (\tilde{f}_{m->j} ||
        #   \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all m->j, j->i not in {6->7, 5->1, 4->3}.
        #   Likewise, we have also:
        #   - \tilde{beta}_{5->6, 6->7} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{4->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{6->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->4, 4->3} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   - \tilde{beta}_{6->7, 7->0} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (pi / 2 + 1) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{5->1, 1->0} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (3 * pi / 4 + 0.5) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->1, 1->2} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (3 * pi / 4 + 0.5) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{4->3, 3->2} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) + (1 / 10 * (pi + 2 / sqrt(3)))
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (5 * pi / 4 + 0.25) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8.
        #   NOTE: We also have self-loops in the dual graph! Hence, one has:
        #   - \tilde{\beta}_{j->i, j->i} = att_{dual}^T * (\tilde{f}_{j->i} ||
        #     \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all j->i not in {6->7, 5->1, 4->3}.
        #   - \tilde{beta}_{6->7, 6->7} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = pi / 2 + 1;
        #   - \tilde{beta}_{5->1, 5->1} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 3 * pi / 4 + 0.5;
        #   - \tilde{beta}_{4->3, 4->3} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 5 * pi / 4 + 0.25.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{m->j, j->i} > 0 and \tilde{beta}_{j->i, j->i} > 0 for
        #   all i, j, m). Then, compute the softmax over the neighboring nodes
        #   m->j. We have (cf. adjacency matrix + self-loops):
        #   - \tilde{\alpha}_{m->0, 0->1} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{0->1, 0->1} = 1 / 2;
        #   - \tilde{\alpha}_{2->1, 1->0} =
        #     = exp(\tilde{beta}_{2->1, 1->0}) /
        #       (exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}) +
        #        exp(\tilde{beta}_{1->0, 1->0})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{5->1, 1->0} =
        #     = exp(\tilde{beta}_{5->1, 1->0}) /
        #       (exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}) +
        #        exp(\tilde{beta}_{1->0, 1->0})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.195730369;
        #   - \tilde{\alpha}_{1->0, 1->0} =
        #     = exp(\tilde{beta}_{1->0, 1->0}) /
        #       (exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0}) +
        #        exp(\tilde{beta}_{1->0, 1->0})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{m->0, 0->7} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{0->7, 0->7} = 1 / 2;
        #   - \tilde{\alpha}_{6->7, 7->0} =
        #     = exp(\tilde{beta}_{6->7, 7->0}) /
        #       (exp(\tilde{beta}_{6->7, 7->0}) +
        #        exp(\tilde{beta}_{7->0, 7->0})) =
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) /
        #       (exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) +
        #        exp(pi + 2 / sqrt(3))) =
        #     ~= 0.296765439;
        #   - \tilde{\alpha}_{7->0, 7->0} =
        #     = exp(\tilde{beta}_{7->0, 7->0}) /
        #       (exp(\tilde{beta}_{6->7, 7->0}) +
        #        exp(\tilde{beta}_{7->0, 7->0})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) +
        #       exp(pi + 2 / sqrt(3))) =
        #     ~= 0.703234561;
        #   - \tilde{\alpha}_{0->1, 1->2} =
        #     = exp(\tilde{beta}_{0->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}) + exp(\tilde{beta}_{1->2, 1->2})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{5->1, 1->2} =
        #     = exp(\tilde{beta}_{5->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}) +
        #        exp(\tilde{beta}_{1->2, 1->2})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.195730369;
        #   - \tilde{\alpha}_{1->2, 1->2} =
        #     = exp(\tilde{beta}_{1->2, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2}) +
        #        exp(\tilde{beta}_{1->2, 1->2})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) + exp(pi + 2 / sqrt(3))) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{m->2, 2->1} = 1 / 2 for m in {3};
        #   - \tilde{\alpha}_{2->1, 2->1} = 1 / 2;
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 3 for m in {0, 2};
        #   - \tilde{\alpha}_{1->5, 1->5} = 1 / 3;
        #   - \tilde{\alpha}_{4->5, 5->1} =
        #     = exp(\tilde{beta}_{4->5, 5->1}) /
        #       (exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}) +
        #        exp(\tilde{beta}_{5->1, 5->1})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(3 * pi / 4 + 0.5)) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{6->5, 5->1} =
        #     = exp(\tilde{beta}_{6->5, 5->1}) /
        #       (exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}) +
        #        exp(\tilde{beta}_{5->1, 5->1})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(3 * pi / 4 + 0.5)) =
        #     ~= 0.402134816;
        #   - \tilde{\alpha}_{5->1, 5->1} =
        #     = exp(\tilde{beta}_{5->1, 5->1}) /
        #       (exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}) +
        #        exp(\tilde{beta}_{5->1, 5->1})) =
        #     = exp(3 * pi / 4 + 0.5) / (exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(3 * pi / 4 + 0.5)) =
        #     ~= 0.195730369;
        #   - \tilde{\alpha}_{m->2, 2->3} = 1 / 2 for m in {1};
        #   - \tilde{\alpha}_{2->3, 2->3} = 1 / 2;
        #   - \tilde{\alpha}_{4->3, 3->2} =
        #     = exp(\tilde{beta}_{4->3, 3->2}) /
        #       (exp(\tilde{beta}_{4->3, 3->2}) +
        #        exp(\tilde{beta}_{3->2, 3->2})) =
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #        exp(pi + 2 / sqrt(3))) =
        #     ~= 0.485091624;
        #   - \tilde{\alpha}_{3->2, 3->2} =
        #     = exp(\tilde{beta}_{3->2, 3->2}) /
        #       (exp(\tilde{beta}_{4->3, 3->2}) +
        #        exp(\tilde{beta}_{3->2, 3->2})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #       exp(pi + 2 / sqrt(3))) =
        #     ~= 0.514908376;
        #   - \tilde{\alpha}_{m->3, 3->4} = 1 / 2 for m in {2};
        #   - \tilde{\alpha}_{3->4, 3->4} = 1 / 2;
        #   - \tilde{\alpha}_{5->4, 4->3} =
        #     = exp(\tilde{beta}_{5->4, 4->3}) /
        #       (exp(\tilde{beta}_{5->4, 4->3}) +
        #        exp(\tilde{beta}_{4->3, 4->3})) =
        #     = exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) /
        #       (exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #        exp(5 * pi / 4 + 0.25)) =
        #     ~= 0.514908376;
        #   - \tilde{\alpha}_{4->3, 4->3} =
        #     = exp(\tilde{beta}_{4->3, 4->3}) /
        #       (exp(\tilde{beta}_{5->4, 4->3}) +
        #        exp(\tilde{beta}_{4->3, 4->3})) =
        #     = exp(5 * pi / 4 + 0.25) /
        #       (exp(9 * pi / 8 + 1 / sqrt(3) + 1 / 8) +
        #        exp(5 * pi / 4 + 0.25)) =
        #     ~= 0.485091624;
        #   - \tilde{\alpha}_{m->4, 4->5} = 1 / 2 for m in {3};
        #   - \tilde{\alpha}_{4->5, 4->5} = 1 / 2;
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 3 for m in {1, 6};
        #   - \tilde{\alpha}_{5->4, 5->4} = 1 / 3;
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 3 for m in {1, 4};
        #   - \tilde{\alpha}_{5->6, 5->6} = 1 / 3;
        #   - \tilde{\alpha}_{m->6, 6->5} = 1 / 2 for m in {7};
        #   - \tilde{\alpha}_{6->5, 6->5} = 1 / 2;
        #   - \tilde{\alpha}_{5->6, 6->7} =
        #     = exp(\tilde{beta}_{5->6, 6->7}) /
        #       (exp(\tilde{beta}_{5->6, 6->7}) +
        #        exp(\tilde{beta}_{6->7, 6->7})) =
        #     = exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) /
        #       (exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) + exp(pi / 2 + 1)) =
        #     ~= 0.703234561;
        #   - \tilde{\alpha}_{6->7, 6->7} =
        #     = exp(\tilde{beta}_{6->7, 6->7}) /
        #       (exp(\tilde{beta}_{5->6, 6->7}) +
        #        exp(\tilde{beta}_{6->7, 6->7})) =
        #     = exp(pi / 2 + 1) / (exp(3 * pi / 4 + 1 / sqrt(3) + 1 / 2) +
        #       exp(pi / 2 + 1)) =
        #     ~= 0.296765439;
        #   - \tilde{\alpha}_{m->7, 7->6} = 1 / 2  for m in {0}.
        #   - \tilde{\alpha}_{7->6, 7->6} = 1 / 2.
        # - The output features are then obtained as f_{j->i}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->j, j->i} * \tilde{f}_{m->j}) +
        #           \tilde{\alpha}_{j->i, j->i} * \tilde{f}_{j->i}) =
        #    = ReLU(\tilde{f}_{n->j} * sum_m(\tilde{\alpha}_{m->j, j->i})) =
        #    = ReLU(\tilde{f}_{n->j}) =
        #    = \tilde{f}_{n->j},      with n->j any of the neighbors of j->i,
        #    where the third-last equality holds if the \tilde{f}_{n->j}'s -
        #    with n->j in the neighborhood of j->i - are all equal and are also
        #    equal to j->i, the second-last holds since the sum all the
        #    attention coefficients over the neighborhood of each dual node
        #    (dual node included) is 1 by construction, and the last one holds
        #    because \tilde{f}_{n->j} > 0, for all valid j, n. The above does
        #    not hold for the "special" \tilde{f}_{m->j} and \tilde{f}_{j->i},
        #    i.e., for i, j such that (m->j) or (j->i) are in {(6->7), (5,->1),
        #    (4->3)}.
        #    We thus have:
        #    - f_{7->0}^' = ReLU(sum_m(\tilde{\alpha}_{m->7, 7->0} *
        #                   \tilde{f}_{m->7}) + \tilde{\alpha}_{7->0, 7->0} *
        #                   \tilde{f}_{7->0}) =
        #                 = ReLU(\tilde{\alpha}_{6->7, 7->0} *
        #                   \tilde{f}_{6->7} + \tilde{\alpha}_{7->0, 7->0} *
        #                   \tilde{f}_{7->0}) =
        #                 = ReLU(0.296765439 * [pi / 2 + 1, pi / 2 + 1,
        #                   pi / 2 + 1, pi / 2 + 1, pi / 2 + 1] + 0.703234561 *
        #                   [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)]) =
        #                ~= [3.78422536, 3.78422536, 3.78422536, 3.78422536,
        #                    3.78422536];
        #    - f_{1->0}^' = ReLU(sum_m(\tilde{\alpha}_{m->1, 1->0} *
        #                   \tilde{f}_{m->1}) + \tilde{\alpha}_{1->0, 1->0} *
        #                   \tilde{f}_{1->0}) =
        #                 = ReLU(\tilde{\alpha}_{2->1, 1->0} *
        #                   \tilde{f}_{2->1} + \tilde{\alpha}_{5->1, 1->0} *
        #                   \tilde{f}_{5->1} + \tilde{\alpha}_{1->0, 1->0} *
        #                   \tilde{f}_{1->0}) =
        #                 = ReLU(0.402134816 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.195730369 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5] + 0.402134816 *
        #                   [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)]) =
        #                 ~= [4.01442215, 4.01442215, 4.01442215, 4.01442215,
        #                     4.01442215];
        #    - f_{1->2}^' = ReLU(sum_m(\tilde{\alpha}_{m->1, 1->2} *
        #                   \tilde{f}_{m->1}) + \tilde{\alpha}_{1->2, 1->2} *
        #                   \tilde{f}_{1->2}) =
        #                 = ReLU(\tilde{\alpha}_{0->1, 1->2} *
        #                   \tilde{f}_{0->1} + \tilde{\alpha}_{5->1, 1->2} *
        #                   \tilde{f}_{5->1} + \tilde{\alpha}_{1->2, 1->2} *
        #                   \tilde{f}_{1->2}) =
        #                 = ReLU(0.402134816 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.195730369 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5] + 0.402134816 *
        #                   [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)]) =
        #                 ~= [4.01442215, 4.01442215, 4.01442215, 4.01442215,
        #                     4.01442215];
        #    - f_{3->2}^' = ReLU(sum_m(\tilde{\alpha}_{m->3, 3->2} *
        #                   \tilde{f}_{m->3}) + \tilde{\alpha}_{3->2, 3->2} *
        #                   \tilde{f}_{3->2}) =
        #                 = ReLU(\tilde{\alpha}_{4->3, 3->2} *
        #                   \tilde{f}_{4->3} + \tilde{\alpha}_{3->2, 3->2} *
        #                   \tilde{f}_{3->2}) =
        #                 = ReLU(0.485091624 * [5 * pi / 4 + 0.25,
        #                   5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                   5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25] +
        #                   0.514908376 * [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3)]) =
        #                 ~= [4.23842061, 4.23842061, 4.23842061, 4.23842061,
        #                     4.23842061];
        #    - f_{6->7}^' = ReLU(sum_m(\tilde{\alpha}_{m->6, 6->7} *
        #                   \tilde{f}_{m->6}) + \tilde{\alpha}_{6->7, 6->7} *
        #                   \tilde{f}_{6->7}) =
        #                 = ReLU(\tilde{\alpha}_{5->6, 6->7} *
        #                   \tilde{f}_{5->6} + \tilde{\alpha}_{6->7, 6->7} *
        #                   \tilde{f}_{6->7}) =
        #                 = ReLU(0.703234561 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.296765439 *
        #                   [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1]) =
        #                 ~= [3.78422536, 3.78422536, 3.78422536, 3.78422536,
        #                     3.78422536];
        #    - f_{5->1}^' = ReLU(sum_m(\tilde{\alpha}_{m->5, 5->1} *
        #                   \tilde{f}_{m->1}) + \tilde{\alpha}_{5->1, 5->1} *
        #                   \tilde{f}_{5->1}) =
        #                 = ReLU(\tilde{\alpha}_{4->5, 5->1} *
        #                   \tilde{f}_{4->5} + \tilde{\alpha}_{6->5, 5->1} *
        #                   \tilde{f}_{6->5} + \tilde{\alpha}_{5->1, 5->1} *
        #                   \tilde{f}_{5->1}) =
        #                 = ReLU(0.402134816 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.402134816 *
        #                   [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)] + 0.195730369 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                 ~= [4.01442215, 4.01442215, 4.01442215, 4.01442215,
        #                     4.01442215];
        #    - f_{4->3}^' = ReLU(sum_m(\tilde{\alpha}_{m->4, 4->3} *
        #                   \tilde{f}_{m->4}) + \tilde{\alpha}_{4->3, 4->3} *
        #                   \tilde{f}_{4->3}) =
        #                 = ReLU(\tilde{\alpha}_{5->4, 4->3} *
        #                   \tilde{f}_{5->4} + \tilde{\alpha}_{4->3, 4->3} *
        #                   \tilde{f}_{4->3}) =
        #                 = ReLU(0.514908376 * [pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.485091624 *
        #                   [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                 ~= [4.23842061, 4.23842061, 4.23842061, 4.23842061,
        #                     4.23842061];
        #     - f_{j->i}' = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)], for all other i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node j->i the quantity
        #   \beta_{j->i} = att_{primal}^T * f_{j->i}^' =
        #                = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                  (1 / 5 * (pi + 2 / sqrt(3))) +
        #                  (1 / 5 * (pi + 2 / sqrt(3))) +
        #                  (1 / 5 * (pi + 2 / sqrt(3))) +
        #                  (1 / 5 * (pi + 2 / sqrt(3))) =
        #                = pi + 2 / sqrt(3), for all (j->i) not in
        #   {(7->0), (1->0), (1->2), (3->2), (6->7), (5->1), (4->3)}. We also
        #   have:
        #   - \beta_{7->0} = att_{primal}^T * f_{7->0}^' =
        #                  = (1 / 5 * 3.78422536) + (1 / 5 * 3.78422536) +
        #                    (1 / 5 * 3.78422536) + (1 / 5 * 3.78422536) +
        #                    (1 / 5 * 3.78422536) =
        #                  = 3.78422536;
        #   - \beta_{1->0} = att_{primal}^T * f_{1->0}^' =
        #                  = (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) =
        #                  = 4.01442215;
        #   - \beta_{1->2} = att_{primal}^T * f_{1->2}^' =
        #                  = (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) =
        #                  = 4.01442215;
        #   - \beta_{3->2} = att_{primal}^T * f_{3->2}^' =
        #                  = (1 / 5 * 4.23842061) + (1 / 5 * 4.23842061) +
        #                    (1 / 5 * 4.23842061) + (1 / 5 * 4.23842061) +
        #                    (1 / 5 * 4.23842061) =
        #                  = 4.23842061.
        #   - \beta_{6->7} = att_{primal}^T * f_{6->7}^' =
        #                  = (1 / 5 * 3.78422536) + (1 / 5 * 3.78422536) +
        #                    (1 / 5 * 3.78422536) + (1 / 5 * 3.78422536) +
        #                    (1 / 5 * 3.78422536) =
        #                  = 3.78422536;
        #   - \beta_{5->1} = att_{primal}^T * f_{5->1}^' =
        #                  = (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) + (1 / 5 * 4.01442215) +
        #                    (1 / 5 * 4.01442215) =
        #                  = 4.01442215;
        #   - \beta_{4->3} = att_{primal}^T * f_{4->3}^' =
        #                  = (1 / 5 * 4.23842061) + (1 / 5 * 4.23842061) +
        #                    (1 / 5 * 4.23842061) + (1 / 5 * 4.23842061) +
        #                    (1 / 5 * 4.23842061) =
        #                  = 4.23842061.
        #   Then LeakyReLU is applied (with no effect, since \beta_{j->i} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge/dual node j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                   exp(\beta_{7->0})) =
        #                 = exp(4.01442215) / (exp(4.01442215) +
        #                    exp(3.78422536)) =
        #                 ~= 0.557296407;
        #   - \alpha_{7->0} = exp(\beta_{7->0}) / (exp(\beta_{1->0}) +
        #                   exp(\beta_{7->0})) =
        #                 = exp(3.78422536) / (exp(4.01442215) +
        #                    exp(3.78422536))
        #                 ~= 0.442703593;
        #   - \alpha_{0->1} = exp(\beta_{0->1}) / (exp(\beta_{0->1}) +
        #                   exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                 = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(pi + 2 / sqrt(3)) + exp(4.01442215)) =
        #                 ~= 0.363059303;
        #   - \alpha_{2->1} = exp(\beta_{2->1}) / (exp(\beta_{0->1}) +
        #                   exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                 = exp(4.01442215) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(4.01442215) + exp(4.01442215)) =
        #                 ~= 0.363059303;
        #   - \alpha_{5->1} = exp(\beta_{5->1}) / (exp(\beta_{0->1}) +
        #                   exp(\beta_{2->1}) + exp(\beta_{5->1})) =
        #                 = exp(4.01442215) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(pi + 2 / sqrt(3)) + exp(4.01442215)) =
        #                 ~= 0.273881394;
        #   - \alpha_{1->2} = exp(\beta_{1->2}) / (exp(\beta_{1->2} +
        #                   exp(\beta_{3->2}) =
        #                 = exp(4.01442215) / (exp(4.01442215) +
        #                    exp(4.23842061)) =
        #                 ~= 0.444233366;
        #   - \alpha_{3->2} = exp(\beta_{3->2}) / (exp(\beta_{1->2} +
        #                   exp(\beta_{3->2}) =
        #                 = exp(4.23842061) / (exp(4.01442215) +
        #                    exp(4.23842061)) =
        #                 ~= 0.555766634;
        #   - \alpha_{2->3} = exp(\beta_{2->3}) / (exp(\beta_{2->3} +
        #                   exp(\beta_{4->3}) =
        #                 = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(4.23842061)) =
        #                 ~= 0.514464109;
        #   - \alpha_{4->3} = exp(\beta_{4->3}) / (exp(\beta_{2->3} +
        #                   exp(\beta_{4->3}) =
        #                 = exp(4.23842061) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(4.23842061)) =
        #                 ~= 0.485535891;
        #   - \alpha_{j->4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{j->5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{j->6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{0->7} = exp(\beta_{0->7}) / (exp(\beta_{0->7} +
        #                   exp(\beta_{6->7}) =
        #                 = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(3.78422536)) =
        #                 ~= 0.625291097;
        #   - \alpha_{6->7} = exp(\beta_{6->7}) / (exp(\beta_{0->7} +
        #                   exp(\beta_{6->7}) =
        #                 = exp(3.78422536) / (exp(pi + 2 / sqrt(3)) +
        #                    exp(3.78422536)) =
        #                 ~= 0.374708903.
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.557296407 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.442703593 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(0.363059303 * [6., 6., 6.] +
        #                   (0.363059303 + 0.273881394) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            ~= [2.25797341, 2.25797341, 2.25797341];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.444233366 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.555766634 * [0.5, 0.5, 0.5]) =
        #            ~= [0.333412488, 0.333412488, 0.333412488];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(0.514464109 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.485535891 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(1 / 2 * [0.5, 0.5, 0.5] +
        #                   1 / 2 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [5 / 16, 5 / 16, 5 / 16];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU(3 / 3 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(1 / 2 * [1 / 8, 1 / 8, 1 / 8] +
        #                   1 / 2 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(0.625291097 * [6., 6., 6.] +
        #                   0.374708903 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [3.79858520, 3.79858520, 3.79858520];
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 2.25797341, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.333412488, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 5 / 16, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 3.79858520, 5)
        # Dual features.
        dual_idx_70 = graph_creator.primal_edge_to_dual_node_idx[(7, 0)]
        dual_idx_10 = graph_creator.primal_edge_to_dual_node_idx[(1, 0)]
        dual_idx_12 = graph_creator.primal_edge_to_dual_node_idx[(1, 2)]
        dual_idx_32 = graph_creator.primal_edge_to_dual_node_idx[(3, 2)]

        remaining_dual_indices = set(range(len(x_dual))) - set([
            dual_idx_70, dual_idx_10, dual_idx_12, dual_idx_32, dual_idx_67,
            dual_idx_51, dual_idx_43
        ])
        for dual_node_idx in remaining_dual_indices:
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)

        x = x_dual[dual_idx_70]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 3.78422536, 5)
        x = x_dual[dual_idx_10]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 4.01442215, 5)
        x = x_dual[dual_idx_12]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 4.01442215, 5)
        x = x_dual[dual_idx_32]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 4.23842061, 5)
        x = x_dual[dual_idx_67]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 3.78422536, 5)
        x = x_dual[dual_idx_51]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 4.01442215, 5)
        x = x_dual[dual_idx_43]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 4.23842061, 5)

    def test_simple_mesh_config_C_features_not_from_dual_no_self_loops(self):
        # - Dual-graph configuration C.
        single_dual_nodes = False
        undirected_dual_edges = False
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges,
                              add_self_loops_to_dual_graph=False)
        # Perform convolution.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - To simplify computations, we manually set the last two features of
        #   each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # Therefore, initial features are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratio 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{j->i} of the dual node/primal
        #   edge j->i becomes \tilde{f}_{j->i} = f_{j->i} * W_{dual} =
        #   = [pi, 2 / sqrt(3), 0, 0] * [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)].
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->j)->(j->i) the quantity (|| indicates concatenation)
        #   \tilde{\beta}_{mj, ji} = att_{dual}^T * (\tilde{f}_{m->j} ||
        #   \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3).
        #   NOTE: There are no self-loops in the dual graph.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   pi + 2 / sqrt(3) > 0). Then, compute the softmax over the (m->j)'s,
        #   neighboring nodes of (j->i).
        #   Since all \tilde{\beta}_{j->i}'s are equal by symmetry, the dual
        #   attention coefficients \tilde{\alpha}_{m->j, j->i} are simply
        #   1 / #(neighboring nodes (m->j)). Therefore:
        #   - \tilde{\alpha}_{m->0, 0->1} = 1     for m in {1};
        #   - \tilde{\alpha}_{m->1, 1->0} = 1 / 2 for m in {2, 5};
        #   - \tilde{\alpha}_{m->0, 0->7} = 1     for m in {1};
        #   - \tilde{\alpha}_{m->7, 7->0} = 1     for m in {6};
        #   - \tilde{\alpha}_{m->1, 1->2} = 1 / 2 for m in {0, 5};
        #   - \tilde{\alpha}_{m->2, 2->1} = 1     for m in {3};
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 2 for m in {0, 2};
        #   - \tilde{\alpha}_{m->5, 5->1} = 1 / 2 for m in {4, 6};
        #   - \tilde{\alpha}_{m->2, 2->3} = 1     for m in {1};
        #   - \tilde{\alpha}_{m->3, 3->2} = 1     for m in {4};
        #   - \tilde{\alpha}_{m->3, 3->4} = 1     for m in {2};
        #   - \tilde{\alpha}_{m->4, 4->3} = 1     for m in {5};
        #   - \tilde{\alpha}_{m->4, 4->5} = 1     for m in {3};
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 2 for m in {1, 6};
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 2 for m in {1, 4};
        #   - \tilde{\alpha}_{m->6, 6->5} = 1     for m in {7};
        #   - \tilde{\alpha}_{m->6, 6->7} = 1     for m in {5};
        #   - \tilde{\alpha}_{m->7, 7->6} = 1     for m in {0};
        #  - The output features are then obtained as f_{j->i}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->j, j->i} * \tilde{f}_{m->j})) =
        #    = ReLU(\tilde{f}_{n->j} * sum_m(\tilde{\alpha}_{m->j, j->i})) =
        #    = ReLU(\tilde{f}_{n->j}) =
        #    = \tilde{f}_{n->j},      with n->j any of the neighbors of j->i,
        #    where the third-last equality holds since the \tilde{f}_{n->j}'s -
        #    with n->j in the neighborhood of j->i - are all equal, the
        #    second-last holds since the sum all the attention coefficients over
        #    the neighborhood of each dual node is 1 by construction, and the
        #    last one holds because \tilde{f}_{n->j} > 0 for all valid j, n.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8].
        # - To compute the attention coefficients, first compute for each dual
        #   node j->i the quantity
        #   \beta_{j->i} = att_{primal}^T * f_{j->i}^' =
        #   = (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) + (1 / 5 * (pi + 2 / sqrt(3))) +
        #     (1 / 5 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3);
        #   then apply LeakyReLU (no effect, since pi + 2 / sqrt(3) > 0), and
        #   then compute the softmax over the neighboring nodes j, i.e, those
        #   nodes s.t. there exists a primal edge/dual node j->i.
        #   Since all \beta_{j->i}'s are equal by symmetry, the primal attention
        #   coefficients \alpha_{j->i} are simply 1 / #(neighboring nodes j).
        #   Therefore:
        #   - \alpha_{j->0} = 1 / 2     for j in {1, 7};
        #   - \alpha_{j->1} = 1 / 3     for j in {0, 2, 5};
        #   - \alpha_{j->2} = 1 / 2     for j in {1, 3};
        #   - \alpha_{j->3} = 1 / 2     for j in {2, 4};
        #   - \alpha_{j->4} = 1 / 2     for j in {3, 5};
        #   - \alpha_{j->5} = 1 / 3     for j in {1, 4, 6};
        #   - \alpha_{j->6} = 1 / 2     for j in {5, 7};
        #   - \alpha_{j->7} = 1 / 2     for j in {0, 6}.
        # - The output features are then obtained as f_i^' =
        #   = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)) =
        #   = ReLU(\tilde{f}_j * sum_j(\alpha_{j->i})) =
        #   = ReLU(\tilde{f}_j) =
        #   = \tilde{f}_j,
        #   where the third-last equality holds since the \tilde{f}_j's are
        #   all equal, the second-last one holds since the sum all the attention
        #   coefficients over the neighborhood of each primal node is 1 by
        #   construction, and the last one holds because \tilde{f}_j > 0 for all
        #   valid j.
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        for x in x_primal:
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        for x in x_dual:
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)

        # ************************
        # Repeat the convolution above, but arbitrarily modify the input
        # features of some primal-/dual-graph nodes.
        # - Set the weights in such a way that weight-multiplied features of
        #   both the primal and the dual graph have all elements equal to the
        #   input features.
        # - Initialize the attention parameter vector to uniformly weight all
        #   contributions.
        conv._primal_layer.weight.data = torch.ones(
            [in_channels_primal, out_channels_primal])
        conv._primal_layer.att.data = torch.ones(
            [1, num_heads, out_channels_dual]) / out_channels_dual
        conv._dual_layer.weight.data = torch.ones(
            [in_channels_dual, out_channels_dual])
        conv._dual_layer.att.data = torch.ones(
            [1, num_heads, 2 * out_channels_dual]) / (2 * out_channels_dual)
        # - Modify the input features of some nodes.
        #   - Change primal nodes 0 and 3.
        primal_graph.x[0] = 6.
        primal_graph.x[3] = 0.5
        #   - Change dual nodes 6->7, 5->1 and 4->3.
        dual_idx_67 = graph_creator.primal_edge_to_dual_node_idx[(6, 7)]
        dual_idx_51 = graph_creator.primal_edge_to_dual_node_idx[(5, 1)]
        dual_idx_43 = graph_creator.primal_edge_to_dual_node_idx[(4, 3)]
        dual_graph.x[dual_idx_67, :] = torch.Tensor([np.pi / 2, 1., 0., 0.])
        dual_graph.x[dual_idx_51, :] = torch.Tensor(
            [3 * np.pi / 4, 0.5, 0., 0.])
        dual_graph.x[dual_idx_43, :] = torch.Tensor(
            [5 * np.pi / 4, 0.25, 0., 0.])
        # - As previously, to simplify computations, we manually set the last
        #   two features of each dual node (i.e., the edge-to-previous-edge- and
        #   edge-to-subsequent-edge- ratios) to 0.
        dual_graph.x[:, 2:4] = 0
        # *** Adjacency matrix primal graph (incoming edges) ***
        #   Node i                  Nodes j with edges j->i
        #   _____________________________________________________
        #       0                   1, 7
        #       1                   0, 2, 5
        #       2                   1, 3
        #       3                   2, 4
        #       4                   3, 5
        #       5                   1, 4, 6
        #       6                   5, 7
        #       7                   0, 6
        #
        # *** Adjacency matrix dual graph (incoming edges) ***
        #   Node j->i             Nodes m->j with edges (m->j)->(j->i)
        #   _____________________________________________________
        #       0->1              7->0
        #       1->0              2->1, 5->1
        #       0->7              1->0
        #       7->0              6->7
        #       1->2              0->1, 5->1
        #       2->1              3->2
        #       1->5              0->1, 2->1
        #       5->1              4->5, 6->5
        #       2->3              1->2
        #       3->2              4->3
        #       3->4              2->3
        #       4->3              5->4
        #       4->5              3->4
        #       5->4              1->5, 6->5
        #       5->6              1->5, 4->5
        #       6->5              7->6
        #       6->7              5->6
        #       7->6              0->7
        #
        # Apart from the features modified above, the remaining initial features
        # are as follows:
        # - Primal graph: all nodes correspond to faces with equal areas,
        #   therefore they all have a single feature equal to 1 / #nodes =
        #   = 1 / 8.
        # - Dual graph: all nodes have associated dihedral angle pi and
        #   edge-height ratio 2 / sqrt(3).
        #
        # Convolution on dual graph:
        # - The node features are first multiplied by the weight matrix
        #   W_{dual}; thus the node feature f_{j->i} of the dual node/primal
        #   edge j->i becomes \tilde{f}_{j->i} = f_{j->i} * W_{dual} =
        #   = [pi, 2 / sqrt(3),  0, 0] * [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1],
        #      [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]] =
        #   = [pi + 2 / sqrt(3), pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #      pi + 2 / sqrt(3), pi + 2 / sqrt(3)], for all j->i not in
        #   {6->7, 5->1, 4->3}.
        #   We have also:
        #   - \tilde{f}_{6->7} = [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #       pi / 2 + 1, pi / 2 + 1];
        #   - \tilde{f}_{5->1} = [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #       3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5];
        #   - \tilde{f}_{4->3} = [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #       5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25];
        # - To compute the attention coefficients, first compute for each dual
        #   edge (m->j)->(j->i) the quantity (|| indicates concatenation)
        #   \tilde{\beta}_{m->j, j->i} = att_{dual}^T * (\tilde{f}_{m->j} ||
        #   \tilde{f}_{j->i}) =
        #   = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #     (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #   = pi + 2 / sqrt(3), for all m->j, j->i not in {6->7, 5->1, 4->3}.
        #   Likewise, we have also:
        #   - \tilde{beta}_{5->6, 6->7} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (pi / 2 + 1) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{4->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{6->5, 5->1} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (3 * pi / 4 + 0.5) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->4, 4->3} =
        #     = (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) =
        #     = 1 / 2 * (pi + 2 / sqrt(3)) + 1 / 2  * (5 * pi / 4 + 0.25) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8;
        #   - \tilde{beta}_{6->7, 7->0} =
        #     = (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi / 2 + 1)) +
        #       (1 / 10 * (pi / 2 + 1)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (pi / 2 + 1) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 3 * pi / 4 + 1 / sqrt(3) + 1 / 2;
        #   - \tilde{beta}_{5->1, 1->0} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (3 * pi / 4 + 0.5) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{5->1, 1->2} =
        #     = (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (3 * pi / 4 + 0.5)) +
        #       (1 / 10 * (3 * pi / 4 + 0.5)) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (3 * pi / 4 + 0.5) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 7 * pi / 8 + 1 / sqrt(3) + 1 / 4;
        #   - \tilde{beta}_{4->3, 3->2} =
        #     = (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) +
        #       (1 / 10 * (5 * pi / 4 + 0.25)) + (1 / 10 * (pi + 2 / sqrt(3)))
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) +
        #       (1 / 10 * (pi + 2 / sqrt(3))) + (1 / 10 * (pi + 2 / sqrt(3))) =
        #     = 1 / 2  * (5 * pi / 4 + 0.25) + 1 / 2 * (pi + 2 / sqrt(3)) =
        #     = 9 * pi / 8 + 1 / sqrt(3) + 1 / 8.
        #   NOTE: There are no self-loops in the dual graph.
        #
        #   Then LeakyReLU is applied (with no effect, since
        #   \tilde{beta}_{m->j, j->i} > 0 for all i, j, m). Then, compute the
        #   softmax over the neighboring nodes m->j. We have (cf. adjacency
        #   matrix):
        #   - \tilde{\alpha}_{m->0, 0->1} = 1     for m in {1};
        #   - \tilde{\alpha}_{2->1, 1->0} =
        #     = exp(\tilde{beta}_{2->1, 1->0}) /
        #       (exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.672617884;
        #   - \tilde{\alpha}_{5->1, 1->0} =
        #     = exp(\tilde{beta}_{5->1, 1->0}) /
        #       (exp(\tilde{beta}_{2->1, 1->0}) +
        #        exp(\tilde{beta}_{5->1, 1->0})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.327382116;
        #   - \tilde{\alpha}_{m->0, 0->7} = 1     for m in {1};
        #   - \tilde{\alpha}_{m->7, 7->0} = 1     for m in {6};
        #   - \tilde{\alpha}_{0->1, 1->2} =
        #     = exp(\tilde{beta}_{0->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2})) =
        #     = exp(pi + 2 / sqrt(3)) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.672617884;
        #   - \tilde{\alpha}_{5->1, 1->2} =
        #     = exp(\tilde{beta}_{5->1, 1->2}) /
        #       (exp(\tilde{beta}_{0->1, 1->2}) +
        #        exp(\tilde{beta}_{5->1, 1->2})) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) / (exp(pi + 2 / sqrt(3)) +
        #       exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     ~= 0.327382116;
        #   - \tilde{\alpha}_{m->2, 2->1} = 1     for m in {3};
        #   - \tilde{\alpha}_{m->1, 1->5} = 1 / 2 for m in {0, 2};
        #   - \tilde{\alpha}_{4->5, 5->1} =
        #     = exp(\tilde{beta}_{4->5, 5->1}) /
        #       (exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     = 1 / 2;
        #   - \tilde{\alpha}_{6->5, 5->1} =
        #     = exp(\tilde{beta}_{6->5, 5->1}) /
        #       (exp(\tilde{beta}_{4->5, 5->1}) +
        #        exp(\tilde{beta}_{6->5, 5->1}) =
        #     = exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) /
        #       (exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4) +
        #        exp(7 * pi / 8 + 1 / sqrt(3) + 1 / 4)) =
        #     = 1 / 2;
        #   - \tilde{\alpha}_{m->2, 2->3} = 1     for m in {1};
        #   - \tilde{\alpha}_{m->3, 3->2} = 1     for m in {4};
        #   - \tilde{\alpha}_{m->3, 3->4} = 1     for m in {2};
        #   - \tilde{\alpha}_{m->4, 4->3} = 1     for m in {5};
        #   - \tilde{\alpha}_{m->4, 4->5} = 1     for m in {3};
        #   - \tilde{\alpha}_{m->5, 5->4} = 1 / 2 for m in {1, 6};
        #   - \tilde{\alpha}_{m->5, 5->6} = 1 / 2 for m in {1, 4};
        #   - \tilde{\alpha}_{m->6, 6->5} = 1     for m in {7};
        #   - \tilde{\alpha}_{m->6, 6->7} = 1     for m in {5};
        #   - \tilde{\alpha}_{m->7, 7->6} = 1     for m in {0}..
        # - The output features are then obtained as f_{j->i}^' =
        #    = ReLU(sum_m(\tilde{\alpha}_{m->j, j->i} * \tilde{f}_{m->j})) =
        #    = ReLU(\tilde{f}_{n->j} * sum_m(\tilde{\alpha}_{m->j, j->i})) =
        #    = ReLU(\tilde{f}_{n->j}) =
        #    = \tilde{f}_{n->j},      with n->j any of the neighbors of j->i,
        #    where the third-last equality holds if the \tilde{f}_{n->j}'s -
        #    with n->j in the neighborhood of j->i - are all equal, the
        #    second-last holds since the sum all the attention coefficients over
        #    the neighborhood of each dual node is 1 by construction, and the
        #    last one holds because \tilde{f}_{n->j} > 0, for all valid j, n.
        #    The above does not hold for the "special" \tilde{f}_{m->j}, i.e.,
        #    for i, j such that (m->j) are in {(6->7), (5->1), (4->3)}.
        #    We thus have:
        #    - f_{7->0}^' = ReLU(sum_m(\tilde{\alpha}_{m->7, 7->0} *
        #                   \tilde{f}_{m->7})) =
        #                 = ReLU(\tilde{\alpha}_{6->7, 7->0} *
        #                   \tilde{f}_{6->7}) =
        #                 = ReLU(1 * [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1, pi / 2 + 1]) =
        #                 = [pi / 2 + 1, pi / 2 + 1, pi / 2 + 1, pi / 2 + 1,
        #                    pi / 2 + 1];
        #    - f_{1->0}^' = ReLU(sum_m(\tilde{\alpha}_{m->1, 1->0} *
        #                   \tilde{f}_{m->1})) =
        #                 = ReLU(\tilde{\alpha}_{2->1, 1->0} *
        #                   \tilde{f}_{2->1} + \tilde{\alpha}_{5->1, 1->0} *
        #                   \tilde{f}_{5->1}) =
        #                 = ReLU(0.672617884 * [pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                   pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.327382116 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5]) =
        #                 ~= [3.82483063, 3.82483063, 3.82483063, 3.82483063,
        #                     3.82483063];
        #    - f_{1->2}^' = ReLU(sum_m(\tilde{\alpha}_{m->1, 1->2} *
        #                   \tilde{f}_{m->1})) =
        #                 = ReLU(\tilde{\alpha}_{0->1, 1->2} *
        #                    \tilde{f}_{0->1} + \tilde{\alpha}_{5->1, 1->2} *
        #                   \tilde{f}_{5->1}) =
        #                 = ReLU(0.672617884 * [pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3)] + 0.327382116 *
        #                   [3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5, 3 * pi / 4 + 0.5,
        #                    3 * pi / 4 + 0.5]) =
        #                 ~= [3.82483063, 3.82483063, 3.82483063, 3.82483063,
        #                     3.82483063];
        #    - f_{3->2}^' = ReLU(sum_m(\tilde{\alpha}_{m->3, 3->2} *
        #                   \tilde{f}_{m->3}) =
        #                 = ReLU(\tilde{\alpha}_{4->3, 3->2} *
        #                   \tilde{f}_{4->3}) =
        #                 = ReLU(1 * [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25]) =
        #                 = [5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25, 5 * pi / 4 + 0.25,
        #                    5 * pi / 4 + 0.25];
        #    - f_{j->i}^' = [pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3), pi + 2 / sqrt(3),
        #                    pi + 2 / sqrt(3)], for all
        #       other i, j.
        #
        # Convolution on primal graph:
        # - The node features are first multiplied by the weight matrix
        #   W_primal; thus the node feature f_i of the primal node i becomes
        #   \tilde{f}_i = f_i * W_{primal} =
        #   = [1 / 8] * [[1, 1, 1]] = [1 / 8, 1 / 8, 1 / 8], for all i not in
        #   {0, 3}.
        #   We also have:
        #   - \tilde{f}_0 = f_0 * W_primal = [6., 6., 6.];
        #   - \tilde{f}_3 = f_3 * W_primal = [0.5, 0.5, 0.5];
        # - To compute the attention coefficients, first compute for each dual
        #   node j->i the quantity
        #   \beta_{j->i} = att_{primal}^T * f_{j->i}^' =
        #              = (1 / 5 * (pi + 2 / sqrt(3))) +
        #                (1 / 5 * (pi + 2 / sqrt(3))) +
        #                (1 / 5 * (pi + 2 / sqrt(3))) +
        #                (1 / 5 * (pi + 2 / sqrt(3))) +
        #                (1 / 5 * (pi + 2 / sqrt(3))) =
        #              = pi + 2 / sqrt(3), for all (j->i) not in
        #   {(7->0), (1->0), (1->2), (3->2)}. We also have:
        #   - \beta_{7->0} = att_{primal}^T * f_{7->0}^' =
        #                = (1 / 5 * (pi / 2 + 1)) + (1 / 5 * (pi / 2 + 1)) +
        #                  (1 / 5 * (pi / 2 + 1)) + (1 / 5 * (pi / 2 + 1)) +
        #                  (1 / 5 * (pi / 2 + 1)) =
        #                = pi / 2 + 1;
        #   - \beta_{1->0} = att_{primal}^T * f_{1->0}^' =
        #                = (1 / 5 * 3.82483063) + (1 / 5 * 3.82483063) +
        #                  (1 / 5 * 3.82483063) + (1 / 5 * 3.82483063) +
        #                  (1 / 5 * 3.82483063) =
        #                = 3.82483063;
        #   - \beta_{1->2} = att_{primal}^T * f_{1->2}^' =
        #                = (1 / 5 * 3.82483063) + (1 / 5 * 3.82483063) +
        #                  (1 / 5 * 3.82483063) + (1 / 5 * 3.82483063) +
        #                  (1 / 5 * 3.82483063) =
        #                = 3.82483063;
        #   - \beta_{3->2} = att_{primal}^T * f_{3->2}^' =
        #                = (1 / 5 * (5 * pi / 4 + 0.25)) +
        #                  (1 / 5 * (5 * pi / 4 + 0.25)) +
        #                  (1 / 5 * (5 * pi / 4 + 0.25)) +
        #                  (1 / 5 * (5 * pi / 4 + 0.25)) +
        #                  (1 / 5 * (5 * pi / 4 + 0.25)) =
        #                = 5 * pi / 4 + 0.25.
        #   Then LeakyReLU is applied (with no effect, since \beta_{j->i} > 0
        #   for all i, j). Then, compute the softmax over the neighboring nodes
        #   j, i.e, those nodes s.t. there exists a primal edge/dual node j->i.
        #   We have:
        #   - \alpha_{1->0} = exp(\beta_{1->0}) / (exp(\beta_{1->0}) +
        #                   exp(\beta_{7->0})) =
        #                 = exp(3.82483063) / (exp(3.82483063) +
        #                    exp(pi / 2 + 1)) =
        #                 ~= 0.777997437;
        #   - \alpha_{7->0} = exp(\beta_{7->0}) / (exp(\beta_{1->0}) +
        #                   exp(\beta_{7->0})) =
        #                 = exp(pi / 2 + 1) / (exp(3.82483063) +
        #                    exp(pi / 2 + 1)) =
        #                 ~= 0.222002563;
        #   - \alpha_{j->1} = 1 / 3     for m in {0, 2, 5};
        #   - \alpha_{1->2} = exp(\beta_{1->2}) / (exp(\beta_{1->2} +
        #                   exp(\beta_{3->2}) =
        #                 = exp(3.82483063) / (exp(3.82483063) +
        #                    exp(5 * pi / 4 + 0.25)) =
        #                 ~= 0.412858680;
        #   - \alpha_{3->2} = exp(\beta_{3->2}) / (exp(\beta_{1->2} +
        #                   exp(\beta_{3->2}) =
        #                 = exp(5 * pi / 4 + 0.25) / (exp(3.82483063) +
        #                    exp(5 * pi / 4 + 0.25)) =
        #                 ~= 0.587141320;
        #   - \alpha_{j->3} = 1 / 2     for m in {2, 4};
        #   - \alpha_{j->4} = 1 / 2     for m in {3, 5};
        #   - \alpha_{j->5} = 1 / 3     for m in {1, 4, 6};
        #   - \alpha_{j->6} = 1 / 2     for m in {5, 7};
        #   - \alpha_{j->7} = 1 / 2     for m in {0, 6}.
        #  - The output features are then obtained as f_i^' =
        #    = ReLU(sum_j(\alpha_{j->i} * \tilde{f}_j)).
        #    We thus have:
        #    - f_0^' = ReLU(sum_j(\alpha_{j->0} * \tilde{f}_j)) =
        #            = ReLU(\alpha_{1->0} * \tilde{f}_1 +
        #                   \alpha_{7->0} * \tilde{f}_7)) =
        #            = ReLU(0.777997437 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.222002563 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_1^' = ReLU(sum_j(\alpha_{j->1} * \tilde{f}_j))
        #            = ReLU(\alpha_{0->1} * \tilde{f}_0 +
        #                   \alpha_{2->1} * \tilde{f}_2 +
        #                   \alpha_{5->1} * \tilde{f}_5) =
        #            = ReLU(1 / 3 * [6., 6., 6.] + (1 / 3 + 1 / 3) *
        #                   [1 / 8, 1 / 8, 1 / 8]) =
        #            = [25 / 12, 25 / 12, 25 / 12];
        #    - f_2^' = ReLU(sum_j(\alpha_{j->2} * \tilde{f}_j))
        #            = ReLU(\alpha_{1->2} * \tilde{f}_1 +
        #                   \alpha_{3->2} * \tilde{f}_3) =
        #            = ReLU(0.412858680 * [1 / 8, 1 / 8, 1 / 8] +
        #                   0.587141320 * [0.5, 0.5, 0.5]) =
        #            ~= [0.345177995, 0.345177995, 0.345177995];
        #    - f_3^' = ReLU(sum_j(\alpha_{j->3} * \tilde{f}_j))
        #            = ReLU(\alpha_{2->3} * \tilde{f}_2 +
        #                   \alpha_{4->3} * \tilde{f}_4) =
        #            = ReLU(1 / 2 * [1 / 8, 1 / 8, 1 / 8] +
        #                   1 / 2 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_4^' = ReLU(sum_j(\alpha_{j->4} * \tilde{f}_j))
        #            = ReLU(\alpha_{3->4} * \tilde{f}_3 +
        #                   \alpha_{5->4} * \tilde{f}_5) =
        #            = ReLU(1 / 2 * [0.5, 0.5, 0.5] +
        #                   1 / 2 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [5 / 16, 5 / 16, 5 / 16];
        #    - f_5^' = ReLU(sum_j(\alpha_{j->5} * \tilde{f}_j))
        #            = ReLU(\alpha_{1->5} * \tilde{f}_1 +
        #                   \alpha_{4->5} * \tilde{f}_4 +
        #                   \alpha_{6->5} * \tilde{f}_6) =
        #            = ReLU(3 / 3 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_6^' = ReLU(sum_j(\alpha_{j->6} * \tilde{f}_j))
        #            = ReLU(\alpha_{5->6} * \tilde{f}_5 +
        #                   \alpha_{7->6} * \tilde{f}_7) =
        #            = ReLU(1 / 2 * [1 / 8, 1 / 8, 1 / 8] +
        #                   1 / 2 * [1 / 8, 1 / 8, 1 / 8]) =
        #            = [1 / 8, 1 / 8, 1 / 8];
        #    - f_7^' = ReLU(sum_j(\alpha_{j->7} * \tilde{f}_j))
        #            = ReLU(\alpha_{0->7} * \tilde{f}_0 +
        #                   \alpha_{6->7} * \tilde{f}_6) =
        #            = ReLU(1 / 2 * [6., 6., 6.] + 1 / 2 *
        #               [1 / 8, 1 / 8, 1 / 8]) =
        #            = [49 / 16, 49 / 16, 49 / 16];
        #
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))
        # Primal features.
        for primal_node in [0, 3, 5, 6]:
            x = x_primal[primal_node]
            for idx in range(3):
                self.assertAlmostEqual(x[idx].item(), 1 / 8)
        x = x_primal[1]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 25 / 12, 5)
        x = x_primal[2]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 0.345177995, 5)
        x = x_primal[4]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 5 / 16, 5)
        x = x_primal[7]
        for idx in range(3):
            self.assertAlmostEqual(x[idx].item(), 49 / 16, 5)
        # Dual features.
        dual_idx_70 = graph_creator.primal_edge_to_dual_node_idx[(7, 0)]
        dual_idx_10 = graph_creator.primal_edge_to_dual_node_idx[(1, 0)]
        dual_idx_12 = graph_creator.primal_edge_to_dual_node_idx[(1, 2)]
        dual_idx_32 = graph_creator.primal_edge_to_dual_node_idx[(3, 2)]

        remaining_dual_indices = set(range(len(x_dual))) - set(
            [dual_idx_70, dual_idx_10, dual_idx_12, dual_idx_32])
        for dual_node_idx in remaining_dual_indices:
            x = x_dual[dual_node_idx]
            for idx in range(5):
                self.assertAlmostEqual(x[idx].item(), np.pi + 2 / np.sqrt(3), 5)
        x = x_dual[dual_idx_70]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), np.pi / 2 + 1, 5)
        x = x_dual[dual_idx_10]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 3.82483063, 5)
        x = x_dual[dual_idx_12]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 3.82483063, 5)
        x = x_dual[dual_idx_32]
        for idx in range(5):
            self.assertAlmostEqual(x[idx].item(), 5 * np.pi / 4 + 0.25, 5)


class TestDualPrimalResConv(unittest.TestCase):

    def test_simple_mesh_directed_primal_features_not_from_dual(self):
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 1
        concat_primal = True
        concat_dual = True
        negative_slope_primal = 0.2
        negative_slope_dual = 0.2
        dropout_primal = 0
        dropout_dual = 0
        bias_primal = False
        bias_dual = False
        conv = DualPrimalResConv(in_channels_primal=in_channels_primal,
                                 in_channels_dual=in_channels_dual,
                                 out_channels_primal=out_channels_primal,
                                 out_channels_dual=out_channels_dual,
                                 heads=num_heads,
                                 concat_primal=concat_primal,
                                 concat_dual=concat_dual,
                                 negative_slope_primal=negative_slope_primal,
                                 negative_slope_dual=negative_slope_dual,
                                 dropout_primal=dropout_primal,
                                 dropout_dual=dropout_dual,
                                 bias_primal=bias_primal,
                                 bias_dual=bias_dual,
                                 single_dual_nodes=single_dual_nodes,
                                 undirected_dual_edges=undirected_dual_edges,
                                 add_self_loops_to_dual_graph=True)
        # Perform convolution.
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        # Simply check that the convolution outputs tensors with the right
        # shape.
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape, (len(dual_graph.x), out_channels_dual))

    def test_multihead_attention_both_concat(self):
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 4
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges)
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        # Note: here the arguments 'concat_primal' and 'concat_dual' are True (
        # by default), so the number of output channels in the primal graph and
        # in the dual graph will be multiplied by the number of attention heads.
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal * num_heads))
        self.assertEqual(x_dual.shape,
                         (len(dual_graph.x), out_channels_dual * num_heads))

    def test_multihead_attention_concat_primal_avg_dual(self):
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 4
        self.assertRaises(ValueError,
                          DualPrimalConv,
                          in_channels_primal=in_channels_primal,
                          in_channels_dual=in_channels_dual,
                          out_channels_primal=out_channels_primal,
                          out_channels_dual=out_channels_dual,
                          heads=num_heads,
                          concat_primal=True,
                          concat_dual=False,
                          single_dual_nodes=False,
                          undirected_dual_edges=False,
                          add_self_loops_to_dual_graph=True)

    def test_multihead_attention_avg_primal_concat_dual(self):
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Create a single dual-primal convolutional layer.
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 4
        conv = DualPrimalConv(in_channels_primal=in_channels_primal,
                              in_channels_dual=in_channels_dual,
                              out_channels_primal=out_channels_primal,
                              out_channels_dual=out_channels_dual,
                              heads=num_heads,
                              concat_primal=False,
                              concat_dual=True,
                              add_self_loops_to_dual_graph=True,
                              single_dual_nodes=single_dual_nodes,
                              undirected_dual_edges=undirected_dual_edges)
        x_primal, x_dual = conv(x_primal=primal_graph.x,
                                x_dual=dual_graph.x,
                                edge_index_primal=primal_graph.edge_index,
                                edge_index_dual=dual_graph.edge_index,
                                primal_edge_to_dual_node_idx=graph_creator.
                                primal_edge_to_dual_node_idx)
        # Note: here the argument 'concat_primal' is False, so the number of
        # output channels in the primal graph will not be multiplied by the
        # number of attention heads. On the other hand, the argument
        # 'concat_dual' is True, so the number of output channels in the dual
        # graph will be multiplied by the number of attention heads.
        self.assertEqual(x_primal.shape,
                         (len(primal_graph.x), out_channels_primal))
        self.assertEqual(x_dual.shape,
                         (len(dual_graph.x), out_channels_dual * num_heads))

    def test_multihead_attention_both_avg(self):
        # Create a single dual-primal convolutional layer.
        single_dual_nodes = False
        undirected_dual_edges = True
        in_channels_primal = 1
        in_channels_dual = 4
        out_channels_primal = 3
        out_channels_dual = 5
        num_heads = 4
        self.assertRaises(ValueError,
                          DualPrimalConv,
                          in_channels_primal=in_channels_primal,
                          in_channels_dual=in_channels_dual,
                          out_channels_primal=out_channels_primal,
                          out_channels_dual=out_channels_dual,
                          heads=num_heads,
                          concat_primal=False,
                          concat_dual=False,
                          add_self_loops_to_dual_graph=True,
                          single_dual_nodes=single_dual_nodes,
                          undirected_dual_edges=undirected_dual_edges)
