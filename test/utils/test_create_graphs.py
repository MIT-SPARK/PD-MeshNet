import numpy as np
import os.path as osp
import pymesh
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import unittest

from pd_mesh_net.utils import create_graphs, create_dual_primal_batch

current_dir = osp.dirname(__file__)


class TestGraphCreator(unittest.TestCase):

    def test_dual_graph_configuration_A(self):
        # Single dual nodes, undirected dual edges.
        graph_creator = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                   single_dual_nodes=True,
                                                   undirected_dual_edges=True)
        # Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Tests on the dual graph.
        petdni = graph_creator.primal_edge_to_dual_node_idx
        # - Check the existence of nodes in the graph.
        for node in [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6),
                     (6, 7), (0, 7)]:
            self.assertTrue(node in petdni)
            self.assertTrue(node[::-1] in petdni)
            self.assertEqual(petdni[node], petdni[node[::-1]])

        # - Check edges in the graph.
        self.assertEqual(dual_graph.num_edges, 24)
        dual_edges = dual_graph.edge_index.t().tolist()
        for node_1, node_2 in [[(0, 1), (1, 2)], [(0, 1), (1, 5)],
                               [(0, 1), (0, 7)], [(1, 2), (1, 5)],
                               [(1, 2), (2, 3)], [(2, 3), (3, 4)],
                               [(3, 4), (4, 5)], [(4, 5), (1, 5)],
                               [(4, 5), (5, 6)], [(5, 6), (1, 5)],
                               [(5, 6), (6, 7)], [(6, 7), (0, 7)]]:
            self.assertTrue([petdni[node_1], petdni[node_2]] in dual_edges)
            self.assertTrue([petdni[node_2], petdni[node_1]] in dual_edges)
        # Tests on the primal graph.
        self.assertEqual(primal_graph.num_edges, 18)
        primal_edges = primal_graph.edge_index.t().tolist()
        for edge in [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [4, 5], [5, 6],
                     [6, 7], [0, 7]]:
            self.assertTrue(edge in primal_edges)
            self.assertTrue(edge[::-1] in primal_edges)

    def test_dual_graph_configuration_B(self):
        # Double dual nodes, undirected dual edges.
        graph_creator = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                   single_dual_nodes=False,
                                                   undirected_dual_edges=True)
        # Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Tests on the dual graph.
        petdni = graph_creator.primal_edge_to_dual_node_idx
        # - Check the existence of nodes in the graph.
        for node in [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6),
                     (6, 7), (0, 7)]:
            self.assertTrue(node in petdni)
            self.assertTrue(node[::-1] in petdni)
            self.assertNotEqual(petdni[node], petdni[node[::-1]])
        # - Check edges in the graph.
        self.assertEqual(dual_graph.num_edges, 48)
        dual_edges = dual_graph.edge_index.t().tolist()
        for node_1, node_2 in [[(0, 1), (1, 2)], [(0, 1), (1, 5)],
                               [(0, 7), (7, 6)], [(1, 0), (0, 7)],
                               [(1, 2), (2, 3)], [(1, 5), (5, 4)],
                               [(1, 5), (5, 6)], [(2, 1), (1, 5)],
                               [(2, 3), (3, 4)], [(3, 4), (4, 5)],
                               [(4, 5), (5, 6)], [(5, 6), (6, 7)]]:
            self.assertTrue([petdni[node_1], petdni[node_2]] in dual_edges)
            self.assertTrue([petdni[node_2], petdni[node_1]] in dual_edges)
            self.assertTrue(
                [petdni[node_2[::-1]], petdni[node_1[::-1]]] in dual_edges)
            self.assertTrue(
                [petdni[node_1[::-1]], petdni[node_2[::-1]]] in dual_edges)
        # Tests on the primal graph.
        self.assertEqual(primal_graph.num_edges, 18)
        primal_edges = primal_graph.edge_index.t().tolist()
        for edge in [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [4, 5], [5, 6],
                     [6, 7], [0, 7]]:
            self.assertTrue(edge in primal_edges)
            self.assertTrue(edge[::-1] in primal_edges)

    def test_dual_graph_configuration_C(self):
        # Double dual nodes, undirected dual edges.
        graph_creator = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                   single_dual_nodes=False,
                                                   undirected_dual_edges=False)
        # Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Tests on the dual graph.
        petdni = graph_creator.primal_edge_to_dual_node_idx
        # - Check the existence of nodes in the graph.
        for node in [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6),
                     (6, 7), (0, 7)]:
            self.assertTrue(node in petdni)
            self.assertTrue(node[::-1] in petdni)
            self.assertNotEqual(petdni[node], petdni[node[::-1]])
        # - Check edges in the graph.
        self.assertEqual(dual_graph.num_edges, 24)
        dual_edges = dual_graph.edge_index.t().tolist()
        for node_1, node_2 in [[(0, 1), (1, 2)], [(0, 1), (1, 5)],
                               [(0, 7), (7, 6)], [(1, 0), (0, 7)],
                               [(1, 2), (2, 3)], [(1, 5), (5, 4)],
                               [(1, 5), (5, 6)], [(2, 1), (1, 5)],
                               [(2, 3), (3, 4)], [(3, 4), (4, 5)],
                               [(4, 5), (5, 6)], [(5, 6), (6, 7)]]:
            self.assertTrue([petdni[node_1], petdni[node_2]] in dual_edges)
            self.assertTrue(
                [petdni[node_2[::-1]], petdni[node_1[::-1]]] in dual_edges)
        # Tests on the primal graph.
        self.assertEqual(primal_graph.num_edges, 18)
        primal_edges = primal_graph.edge_index.t().tolist()
        for edge in [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [4, 5], [5, 6],
                     [6, 7], [0, 7]]:
            self.assertTrue(edge in primal_edges)
            self.assertTrue(edge[::-1] in primal_edges)

    def test_graph_config_A_primal_features_not_from_dual(self):
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration A + primal features not from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=True,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        original_mesh = pymesh.load_mesh(mesh_filename)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        #   - In the current mode, dual nodes are assigned as features:
        #     1. The dihedral angle between the corresponding two faces in the
        #        original triangular mesh. Since the input mesh is planar, this
        #        value should be pi radians for each node;
        #     2.,3. The edge-height ratios - where the edge is the one
        #        corresponding to the dual node in the original triangular mesh
        #        - of the two faces associated with the dual node in the
        #        original triangular mesh, sorted in increasing order. The faces
        #        with index 0, 1, 2, 4, 5 and 6 are approximately equilateral
        #        triangles, and have therefore edge-height ratio equal to
        #        2 / sqrt(3); it is easy to show that by construction also the
        #        faces 3 and 7 have the same edge-height ratio when considering
        #        the edges shared with other faces in the mesh;
        #     4.,5. The edge-to-previous-edge ratios of the two face associated
        #        with the dual node in the original triangular mesh, sorted so
        #        as to match the sorting of the edge-height ratios (cf. docs of
        #        GraphCreator). The faces with index 0, 1, 2, 4, 5 and 6 are
        #        approximately equilateral triangles, and have therefore
        #        edge-to-previous edge ratio equal to 1; by construction the
        #        edge-to-previous-edge ratios (cf. docs of GraphCreator for
        #        notation) r_{32}_{bef} and r_{76}_{bef} are 1 / sqrt(3),
        #        whereas r_{34}_{bef} and r_{70}_{bef} are 1.
        #     6.,7. The edge-to-subsequent-edge ratios. The faces with index 0,
        #        1, 2, 4, 5 and 6 are approximately equilateral triangles, and
        #        have therefore edge-to-subsequent edge ratio equal to 1; by
        #        construction the edge-to-subsequent-edge ratios r_{34}_{aft}
        #        and r_{70}_{aft} are 1 / sqrt(3), whereas r_{32}_{aft} and
        #        r_{76}_{aft} are 1.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertEqual(primal_edge_to_dual_node_idx[(2, 3)],
                         primal_edge_to_dual_node_idx[(3, 2)])
        dual_node_idx_edge_23 = primal_edge_to_dual_node_idx[(2, 3)]
        self.assertEqual(primal_edge_to_dual_node_idx[(6, 7)],
                         primal_edge_to_dual_node_idx[(7, 6)])
        dual_node_idx_edge_67 = primal_edge_to_dual_node_idx[(6, 7)]
        self.assertEqual(primal_edge_to_dual_node_idx[(3, 4)],
                         primal_edge_to_dual_node_idx[(4, 3)])
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        self.assertEqual(primal_edge_to_dual_node_idx[(0, 7)],
                         primal_edge_to_dual_node_idx[(7, 0)])
        dual_node_idx_edge_07 = primal_edge_to_dual_node_idx[(0, 7)]
        num_dual_nodes = maybe_num_nodes(dual_graph.edge_index)
        self.assertEqual(num_dual_nodes, len(primal_edge_to_dual_node_idx) // 2)
        for dual_node in range(num_dual_nodes):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 3)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_23, dual_node_idx_edge_67]):
                edge_previous_edge_ratios = sorted([
                    dual_features[dual_node, 3].item(), dual_features[dual_node,
                                                                      4].item()
                ])
                self.assertAlmostEqual(edge_previous_edge_ratios[0],
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(edge_previous_edge_ratios[1], 1., 3)
                self.assertAlmostEqual(dual_features[dual_node, 5].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 6].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_07, dual_node_idx_edge_34]):
                edge_subsequent_edge_ratios = sorted([
                    dual_features[dual_node, 5].item(), dual_features[dual_node,
                                                                      6].item()
                ])
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 4].item(), 1.,
                                       3)
                self.assertAlmostEqual(edge_subsequent_edge_ratios[0],
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(edge_subsequent_edge_ratios[1], 1., 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 4].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 5].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 6].item(), 1.,
                                       3)

        #   Tests on the primal graph.
        #   - In the current mode, primal nodes are assigned as feature the
        #     ratio between the area of the corresponding face in the original
        #     triangular mesh and the sum of the areas of all the faces in the
        #     mesh. Let us therefore perform the check by computing the areas.
        primal_features = primal_graph.x
        original_mesh.add_attribute("face_area")
        face_areas = original_mesh.get_attribute("face_area")
        sum_face_areas = np.sum(face_areas)
        for primal_node in range(len(primal_features)):
            face_area = face_areas[primal_node]
            self.assertAlmostEqual(primal_features[primal_node].item(),
                                   face_area / sum_face_areas)

    def test_graph_config_A_primal_features_from_dual(self):
        self.__test_graph_config_A_primal_features_from_dual()

    def __test_graph_config_A_primal_features_from_dual(self):
        dual_node_shift = 0
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration A + primal features from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=True,
            undirected_dual_edges=True,
            primal_features_from_dual_features=True)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        #   - In the current mode, dual nodes are assigned as features the same
        #     features as in the test
        #     test_graph_config_A_primal_features_not_from_dual.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertEqual(primal_edge_to_dual_node_idx[(2, 3)],
                         primal_edge_to_dual_node_idx[(3, 2)])
        dual_node_idx_edge_23 = primal_edge_to_dual_node_idx[(2, 3)]
        self.assertEqual(primal_edge_to_dual_node_idx[(6, 7)],
                         primal_edge_to_dual_node_idx[(7, 6)])
        dual_node_idx_edge_67 = primal_edge_to_dual_node_idx[(6, 7)]
        self.assertEqual(primal_edge_to_dual_node_idx[(3, 4)],
                         primal_edge_to_dual_node_idx[(4, 3)])
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        self.assertEqual(primal_edge_to_dual_node_idx[(0, 7)],
                         primal_edge_to_dual_node_idx[(7, 0)])
        dual_node_idx_edge_07 = primal_edge_to_dual_node_idx[(0, 7)]
        num_dual_nodes = maybe_num_nodes(dual_graph.edge_index)
        self.assertEqual(num_dual_nodes, len(primal_edge_to_dual_node_idx) // 2)
        for dual_node in range(dual_node_shift, num_dual_nodes):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 3)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_23, dual_node_idx_edge_67]):
                edge_previous_edge_ratios = sorted([
                    dual_features[dual_node, 3].item(), dual_features[dual_node,
                                                                      4].item()
                ])
                self.assertAlmostEqual(edge_previous_edge_ratios[0],
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(edge_previous_edge_ratios[1], 1., 3)
                self.assertAlmostEqual(dual_features[dual_node, 5].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 6].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_07, dual_node_idx_edge_34]):
                edge_subsequent_edge_ratios = sorted([
                    dual_features[dual_node, 5].item(), dual_features[dual_node,
                                                                      6].item()
                ])
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 4].item(), 1.,
                                       3)
                self.assertAlmostEqual(edge_subsequent_edge_ratios[0],
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(edge_subsequent_edge_ratios[1], 1., 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 4].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 5].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 6].item(), 1.,
                                       3)

        #   Tests on the primal graph.
        #   - In the current mode, primal nodes i are assigned as features:
        #     1. The average of the dihedral angles over their neighboring edges
        #        (i.e., dual nodes). By symmetry considerations, the dihedral
        #        angles are all expected to be approximately pi radians;
        #     2. The average of the edge-height ratios k_{im} over their
        #        neighboring faces m. By symmetry considerations, the
        #        edge-height ratios are all expected to be approximately
        #        2 / sqrt(3).
        #     3. The average of the edge-to-previous-edge ratios r_{im}_{bef}
        #        over their neighboring faces m. By symmetry considerations, for
        #        primal nodes 0, 1, 2, 4, 5 and 6, this average is expected to
        #        be 1.; for primal nodes 3 and 7, this average is expected to be
        #        (1. + 1. / sqrt(3)) / 2 = 1 / 2 + 1 / (2 * sqrt(3)).
        #     4. The average of the edge-to-subsequent-edge ratios r_{im}_{aft}
        #        over their neighboring faces m. By symmetry considerations, for
        #        primal nodes 0, 1, 2, 4, 5 and 6, this average is expected to
        #        be 1.; for primal nodes 3 and 7, this average is expected to be
        #        (1. + 1. / sqrt(3)) / 2 = 1 / 2 + 1 / (2 * sqrt(3)).
        primal_features = primal_graph.x
        for primal_node in range(len(primal_features)):
            self.assertAlmostEqual(primal_features[primal_node, 0].item(),
                                   np.pi, 3)
            self.assertAlmostEqual(primal_features[primal_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (primal_node in [3, 7]):
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
            else:
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       1., 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       1., 3)

    def test_graph_config_B_primal_features_not_from_dual(self):
        self.__test_graph_config_B_primal_features_not_from_dual()

    def __test_graph_config_B_primal_features_not_from_dual(self):
        dual_node_shift = 0
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration B + primal features not from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        original_mesh = pymesh.load_mesh(mesh_filename)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        #   - In the current mode, dual node i->j is assigned as features:
        #     1. The dihedral angle between faces i and j in the original
        #        triangular mesh. Since the input mesh is planar, this value
        #        should be pi radians for each node;
        #     2. The edge-height ratio - where the edge is the one corresponding
        #        to the dual node in the original triangular mesh - of face i.
        #        The faces with index 0, 1, 2, 4, 5 and 6 are approximately
        #        equilateral triangles, and have therefore all edge-height
        #        ratios equal to 2 / sqrt(3); it is easy to show that by
        #        construction also the faces 3 and 7 have the same edge-height
        #        ratio when considering the edges shared with other faces in the
        #        mesh;
        #     3. The edge-to-previous-edge ratio. The faces with index 0, 1, 2,
        #         4, 5 and 6 are approximately equilateral triangles, and have
        #         therefore all edge-to-previous edge ratios equal to 1; by
        #         construction the edge-to-previous-edge ratios associated to
        #         dual nodes 3->2 and 7->6 are 1 / sqrt(3), whereas those
        #         associated to dual nodes 3->4 and 7->0 are 1.
        #     4. The edge-to-subsequent-edge ratio. The faces with index 0, 1,
        #        2, 4, 5 and 6 are approximately equilateral triangles, and have
        #        therefore all edge-to-subsequent edge ratios equal to 1; by
        #        construction the edge-to-subsequent-edge ratios associated to
        #        dual nodes 3->4 and 7->0 are 1 / sqrt(3), whereas those
        #        associated to dual nodes 3->2 and 7->6 are 1.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        dual_node_idx_edge_32 = primal_edge_to_dual_node_idx[(3, 2)]
        dual_node_idx_edge_76 = primal_edge_to_dual_node_idx[(7, 6)]
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        dual_node_idx_edge_70 = primal_edge_to_dual_node_idx[(7, 0)]
        for dual_node in range(dual_node_shift,
                               len(primal_edge_to_dual_node_idx)):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 5)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_32, dual_node_idx_edge_76]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_34, dual_node_idx_edge_70]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(),
                                       1. / np.sqrt(3), 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)

        #   Tests on the primal graph.
        #   - In the current mode, primal nodes are assigned as feature the
        #     ratio between the area of the corresponding face in the original
        #     triangular mesh and the sum of the areas of all the faces in the
        #     mesh. Let us therefore perform the check by computing the areas.
        primal_features = primal_graph.x
        original_mesh.add_attribute("face_area")
        face_areas = original_mesh.get_attribute("face_area")
        sum_face_areas = np.sum(face_areas)
        for primal_node in range(len(primal_features)):
            face_area = face_areas[primal_node]
            self.assertAlmostEqual(primal_features[primal_node].item(),
                                   face_area / sum_face_areas)

        # * 90-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_90_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            np.pi / 2)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        #   - Letting e be the edge between face 0 and 1 and h be the
        #     corresponding height in face 0, it holds that e / h = 1 / sqrt(3).
        #     Since face 0 is isosceles, the two (equal) edges other than e have
        #     length:
        #     l = sqrt((e / 2)^2 + h^2) = sqrt((e / 2)^2 + (sqrt(3) * e)^2) =
        #       = e * sqrt(13) / 2. Therefore, the two edge-to-previous-edge-
        #     and edge-to-subsequent-edge- ratios are both e / l = 2 / sqrt(13).
        #     Since face 1 is equilateral, its edge-to-previous-edge- and
        #     edge-to-subsequent-edge- ratios are both 1.
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 2].item(),
            2 / np.sqrt(13), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 3].item(),
            2 / np.sqrt(13), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            np.pi / 2)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 2].item(), 1.,
            3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 3].item(), 1.,
            3)
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 135-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_135_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            3 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            3 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 225-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_225_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            5 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            5 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 315-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_315_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            7 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            7 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)

    def test_graph_config_B_primal_features_from_dual(self):
        self.__test_graph_config_B_primal_features_from_dual()

    def __test_graph_config_B_primal_features_from_dual(self):
        dual_node_shift = 0
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration B + primal features from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=True,
            primal_features_from_dual_features=True)
        # This case is equal to dual-graph configuration A for the primal nodes,
        # and to case with primal features not from dual of dual-graph
        # configuration B.
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # - Dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        dual_node_idx_edge_32 = primal_edge_to_dual_node_idx[(3, 2)]
        dual_node_idx_edge_76 = primal_edge_to_dual_node_idx[(7, 6)]
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        dual_node_idx_edge_70 = primal_edge_to_dual_node_idx[(7, 0)]
        for dual_node in range(dual_node_shift,
                               len(primal_edge_to_dual_node_idx)):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 5)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_32, dual_node_idx_edge_76]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_34, dual_node_idx_edge_70]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(),
                                       1. / np.sqrt(3), 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
        # - Primal graph.
        primal_features = primal_graph.x
        for primal_node in range(len(primal_features)):
            self.assertAlmostEqual(primal_features[primal_node, 0].item(),
                                   np.pi, 3)
            self.assertAlmostEqual(primal_features[primal_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (primal_node in [3, 7]):
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
            else:
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       1., 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       1., 3)

    def test_graph_config_C_primal_features_not_from_dual(self):
        self.__test_graph_config_C_primal_features_not_from_dual()

    def __test_graph_config_C_primal_features_not_from_dual(self):
        dual_node_shift = 0
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration C + primal features not from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=False)
        original_mesh = pymesh.load_mesh(mesh_filename)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # Both primal and dual features in this case should be equal to those of
        # the case of dual-graph configuration C.
        # - Dual-graph features.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        dual_node_idx_edge_32 = primal_edge_to_dual_node_idx[(3, 2)]
        dual_node_idx_edge_76 = primal_edge_to_dual_node_idx[(7, 6)]
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        dual_node_idx_edge_70 = primal_edge_to_dual_node_idx[(7, 0)]
        for dual_node in range(dual_node_shift,
                               len(primal_edge_to_dual_node_idx)):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 5)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_32, dual_node_idx_edge_76]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_34, dual_node_idx_edge_70]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(),
                                       1. / np.sqrt(3), 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
        # - Primal-graph features.
        primal_features = primal_graph.x
        original_mesh.add_attribute("face_area")
        face_areas = original_mesh.get_attribute("face_area")
        sum_face_areas = np.sum(face_areas)
        for primal_node in range(len(primal_features)):
            face_area = face_areas[primal_node]
            self.assertAlmostEqual(primal_features[primal_node].item(),
                                   face_area / sum_face_areas)

        # * 90-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_90_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            np.pi / 2)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        #   - Letting e be the edge between face 0 and 1 and h be the
        #     corresponding height in face 0, it holds that e / h = 1 / sqrt(3).
        #     Since face 0 is isosceles, the two (equal) edges other than e have
        #     length:
        #     l = sqrt((e / 2)^2 + h^2) = sqrt((e / 2)^2 + (sqrt(3) * e)^2) =
        #       = e * sqrt(13) / 2. Therefore, the two edge-to-previous-edge-
        #     and edge-to-subsequent-edge- ratios are both e / l = 2 / sqrt(13).
        #     Since face 1 is equilateral, its edge-to-previous-edge- and
        #     edge-to-subsequent-edge- ratios are both 1.
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 2].item(),
            2 / np.sqrt(13), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 3].item(),
            2 / np.sqrt(13), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            np.pi / 2)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 2].item(), 1.,
            3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 3].item(), 1.,
            3)
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 135-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_135_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            3 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            3 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 225-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_225_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            5 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            5 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)
        # * 315-degree mesh.
        mesh_filename = osp.join(current_dir,
                                 '../common_data/two_faces_315_deg.ply')
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=False)
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        #   Tests on the dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 0].item(),
            7 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(0, 1)], 1].item(),
            1 / np.sqrt(3), 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 0].item(),
            7 * np.pi / 4, 3)
        self.assertAlmostEqual(
            dual_features[primal_edge_to_dual_node_idx[(1, 0)], 1].item(),
            2 / np.sqrt(3), 3)
        #   - Edge-to-previous-edge- and edge-to-subsequent-edge- ratios are
        #     equal to the case with 90 degrees are therefore not repeated.
        #   Tests on the primal graph.
        primal_features = primal_graph.x
        self.assertAlmostEqual(primal_features[0].item(), 2 / 3, 4)
        self.assertAlmostEqual(primal_features[1].item(), 1 / 3, 4)

    def test_graph_config_C_primal_features_from_dual(self):
        self.__test_graph_config_C_primal_features_from_dual()

    def __test_graph_config_C_primal_features_from_dual(self):
        dual_node_shift = 0
        # * Simple mesh.
        mesh_filename = osp.join(current_dir, '../common_data/simple_mesh.ply')
        # Dual-graph configuration C + primal features not from dual features.
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=mesh_filename,
            single_dual_nodes=False,
            undirected_dual_edges=False,
            primal_features_from_dual_features=True)
        # This case is equal to dual-graph configuration A for the primal nodes,
        # and to case with primal features not from dual of dual-graph
        # configuration C.
        #   Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        # - Dual graph.
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        dual_features = dual_graph.x
        dual_node_idx_edge_32 = primal_edge_to_dual_node_idx[(3, 2)]
        dual_node_idx_edge_76 = primal_edge_to_dual_node_idx[(7, 6)]
        dual_node_idx_edge_34 = primal_edge_to_dual_node_idx[(3, 4)]
        dual_node_idx_edge_70 = primal_edge_to_dual_node_idx[(7, 0)]
        for dual_node in range(dual_node_shift,
                               len(primal_edge_to_dual_node_idx)):
            self.assertAlmostEqual(dual_features[dual_node, 0].item(), np.pi, 5)
            self.assertAlmostEqual(dual_features[dual_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (dual_node in [dual_node_idx_edge_32, dual_node_idx_edge_76]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(),
                                       1. / np.sqrt(3), 3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
            elif (dual_node in [dual_node_idx_edge_34, dual_node_idx_edge_70]):
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(),
                                       1. / np.sqrt(3), 3)
            else:
                self.assertAlmostEqual(dual_features[dual_node, 2].item(), 1.,
                                       3)
                self.assertAlmostEqual(dual_features[dual_node, 3].item(), 1.,
                                       3)
        # - Primal graph.
        primal_features = primal_graph.x
        for primal_node in range(len(primal_features)):
            self.assertAlmostEqual(primal_features[primal_node, 0].item(),
                                   np.pi, 3)
            self.assertAlmostEqual(primal_features[primal_node, 1].item(),
                                   2 / np.sqrt(3), 3)
            if (primal_node in [3, 7]):
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       ((1. + 1. / np.sqrt(3)) / 2.), 3)
            else:
                self.assertAlmostEqual(primal_features[primal_node, 2].item(),
                                       1., 3)
                self.assertAlmostEqual(primal_features[primal_node, 3].item(),
                                       1., 3)


class TestBatchCreator(unittest.TestCase):

    def test_configuration_B(self):
        self.__test_configuration_B()

    def __test_configuration_B(self):
        graph_creator = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                   single_dual_nodes=False,
                                                   undirected_dual_edges=True)
        # Obtain primal and dual graph.
        primal_graph, dual_graph = graph_creator.create_graphs()
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        # Create a simple batch made of three equal versions of the primal-dual
        # graphs.
        num_replicas = 3
        primal_graph_list = [primal_graph] * num_replicas
        dual_graph_list = [dual_graph] * num_replicas
        primal_edge_to_dual_node_idx_list = [primal_edge_to_dual_node_idx
                                            ] * num_replicas
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch) = create_dual_primal_batch(
             primal_graph_list, dual_graph_list,
             primal_edge_to_dual_node_idx_list)

        num_primal_nodes = 8
        num_primal_edges = 18
        num_dual_edges = 48
        self.assertEqual(num_primal_nodes,
                         maybe_num_nodes(primal_graph.edge_index))
        self.assertEqual(num_primal_edges, primal_graph.num_edges)

        self.assertEqual(num_primal_edges,
                         maybe_num_nodes(dual_graph.edge_index))
        self.assertEqual(num_dual_edges, dual_graph.num_edges)

        # Tests on the dual-graph batch.
        # - Check the existence of nodes in the graph.
        for batch_sample_idx in range(num_replicas):
            for edge in [(0, 1), (0, 7), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5),
                         (5, 6), (6, 7)]:
                self.assertTrue((edge[0] + num_primal_nodes * batch_sample_idx,
                                 edge[1] + num_primal_nodes * batch_sample_idx
                                ) in primal_edge_to_dual_node_idx_batch)
                self.assertTrue((edge[1] + num_primal_nodes * batch_sample_idx,
                                 edge[0] + num_primal_nodes * batch_sample_idx
                                ) in primal_edge_to_dual_node_idx_batch)
        # - Check edges in the graph.
        self.assertEqual(dual_graph_batch.num_edges,
                         num_dual_edges * num_replicas)
        dual_edges = dual_graph_batch.edge_index.t().tolist()
        petdni_batch = primal_edge_to_dual_node_idx_batch
        for batch_sample_idx in range(num_replicas):
            for edge in [[(0, 1), (1, 2)], [(0, 1), (1, 5)], [(0, 7), (7, 6)],
                         [(1, 0), (0, 7)], [(1, 2), (2, 3)], [(1, 5), (5, 4)],
                         [(1, 5), (5, 6)], [(2, 1), (1, 5)], [(2, 3), (3, 4)],
                         [(3, 4), (4, 5)], [(4, 5), (5, 6)], [(5, 6), (6, 7)]]:
                node_1 = (edge[0][0] + num_primal_nodes * batch_sample_idx,
                          edge[0][1] + num_primal_nodes * batch_sample_idx)
                node_2 = (edge[1][0] + num_primal_nodes * batch_sample_idx,
                          edge[1][1] + num_primal_nodes * batch_sample_idx)
                self.assertTrue(
                    [petdni_batch[node_1], petdni_batch[node_2]] in dual_edges)
                self.assertTrue(
                    [petdni_batch[node_2], petdni_batch[node_1]] in dual_edges)
                self.assertTrue(
                    [petdni_batch[node_2[::-1]], petdni_batch[node_1[::-1]]
                    ] in dual_edges)
                self.assertTrue(
                    [petdni_batch[node_1[::-1]], petdni_batch[node_2[::-1]]
                    ] in dual_edges)

        # Tests on the primal graph.
        self.assertEqual(primal_graph_batch.num_edges,
                         num_primal_edges * num_replicas)
        primal_edges = primal_graph_batch.edge_index.t().tolist()
        for batch_sample_idx in range(num_replicas):
            for edge in [(0, 1), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6),
                         (6, 7), (0, 7)]:
                self.assertTrue([
                    edge[0] + num_primal_nodes * batch_sample_idx, edge[1] +
                    num_primal_nodes * batch_sample_idx
                ] in primal_edges)
                self.assertTrue([
                    edge[1] + num_primal_nodes * batch_sample_idx, edge[0] +
                    num_primal_nodes * batch_sample_idx
                ] in primal_edges)

    def test_configurations_B_and_A(self):
        self.__test_configurations_B_and_A()

    def __test_configurations_B_and_A(self):
        graph_creator_A = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                     single_dual_nodes=True,
                                                     undirected_dual_edges=True)
        graph_creator_B = create_graphs.GraphCreator(mesh_filename=osp.join(
            current_dir, '../common_data/simple_mesh.ply'),
                                                     single_dual_nodes=False,
                                                     undirected_dual_edges=True)
        # Obtain primal and dual graph.
        (primal_graph_A, dual_graph_A) = graph_creator_A.create_graphs()
        petdni_A = graph_creator_A.primal_edge_to_dual_node_idx
        (primal_graph_B, dual_graph_B) = graph_creator_B.create_graphs()
        petdni_B = graph_creator_B.primal_edge_to_dual_node_idx
        # Create a simple batch made of the primal-dual graphs of configuration
        # A followed by the primal-dual graphs of configuration B, and the same
        # combination repeated.
        # - First, adapt the number of node-feature channels of the undirected
        #   graphs so as to match the one of the directed graphs.
        num_features_primal_A = primal_graph_A.x.shape[-1]
        num_features_dual_A = dual_graph_A.x.shape[-1]
        num_features_primal_B = primal_graph_B.x.shape[-1]
        num_features_dual_B = dual_graph_B.x.shape[-1]
        dual_graph_B.x = torch.nn.ConstantPad1d(
            (0, num_features_dual_A - num_features_dual_B), 0)(dual_graph_B.x)
        num_replicas = 2
        primal_graph_list = [primal_graph_A, primal_graph_B] * num_replicas
        dual_graph_list = [dual_graph_A, dual_graph_B] * num_replicas
        primal_edge_to_dual_node_idx_list = [petdni_A, petdni_B] * num_replicas
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch) = create_dual_primal_batch(
             primal_graph_list, dual_graph_list,
             primal_edge_to_dual_node_idx_list)

        num_primal_nodes_A = 8
        num_primal_nodes_B = 8
        num_primal_edges_A = 18
        num_primal_edges_B = 18
        num_dual_nodes_A = 9
        num_dual_nodes_B = 18
        num_dual_edges_A = 24
        num_dual_edges_B = 48
        self.assertEqual(maybe_num_nodes(primal_graph_A.edge_index),
                         num_primal_nodes_A)
        self.assertEqual(maybe_num_nodes(primal_graph_B.edge_index),
                         num_primal_nodes_B)
        self.assertEqual(primal_graph_A.num_edges, num_primal_edges_A)
        self.assertEqual(primal_graph_B.num_edges, num_primal_edges_B)
        self.assertEqual(maybe_num_nodes(dual_graph_A.edge_index),
                         num_dual_nodes_A)
        self.assertEqual(maybe_num_nodes(dual_graph_B.edge_index),
                         num_dual_nodes_B)
        self.assertEqual(dual_graph_A.num_edges, num_dual_edges_A)
        self.assertEqual(dual_graph_B.num_edges, num_dual_edges_B)

        # Tests on the dual-graph batch.
        # - Check the existence of nodes in the graph.
        for batch_sample_idx in range(num_replicas):
            # - Dual-graph configuration A.
            for edge in [(0, 1), (0, 7), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5),
                         (5, 6), (6, 7)]:
                new_node_1 = edge[0] + (num_primal_nodes_A +
                                        num_primal_nodes_B) * batch_sample_idx
                new_node_2 = edge[1] + (num_primal_nodes_A +
                                        num_primal_nodes_B) * batch_sample_idx
                self.assertTrue(
                    (new_node_1,
                     new_node_2) in primal_edge_to_dual_node_idx_batch)
                self.assertTrue(
                    (new_node_2,
                     new_node_1) in primal_edge_to_dual_node_idx_batch)
                self.assertEqual(
                    primal_edge_to_dual_node_idx_batch[(new_node_1,
                                                        new_node_2)],
                    primal_edge_to_dual_node_idx_batch[(new_node_2,
                                                        new_node_1)])
            # - Dual-graph configuration B.
            for edge in [(0, 1), (0, 7), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5),
                         (5, 6), (6, 7)]:
                new_node_1 = edge[0] + (num_primal_nodes_A + num_primal_nodes_B
                                       ) * batch_sample_idx + num_primal_nodes_A
                new_node_2 = edge[1] + (num_primal_nodes_A + num_primal_nodes_B
                                       ) * batch_sample_idx + num_primal_nodes_A
                self.assertTrue(
                    (new_node_1,
                     new_node_2) in primal_edge_to_dual_node_idx_batch)
                self.assertTrue(
                    (new_node_2,
                     new_node_1) in primal_edge_to_dual_node_idx_batch)
                self.assertNotEqual(
                    primal_edge_to_dual_node_idx_batch[(new_node_1,
                                                        new_node_2)],
                    primal_edge_to_dual_node_idx_batch[(new_node_2,
                                                        new_node_1)])
        # - Check edges in the graph.
        self.assertEqual(dual_graph_batch.num_edges,
                         (num_dual_edges_A + num_dual_edges_B) * num_replicas)
        dual_edges = dual_graph_batch.edge_index.t().tolist()
        petdni_batch = primal_edge_to_dual_node_idx_batch
        for batch_sample_idx in range(num_replicas):
            # - Dual-graph configuration A.
            for edge in [[(0, 1), (1, 2)], [(0, 1), (1, 5)], [(0, 7), (6, 7)],
                         [(0, 1), (0, 7)], [(1, 2), (2, 3)], [(1, 5), (4, 5)],
                         [(1, 5), (5, 6)], [(1, 2), (1, 5)], [(2, 3), (3, 4)],
                         [(3, 4), (4, 5)], [(4, 5), (5, 6)], [(5, 6), (6, 7)]]:
                new_node_1 = (edge[0][0] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx, edge[0][1] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx)
                new_node_2 = (edge[1][0] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx, edge[1][1] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx)
                self.assertTrue(
                    [petdni_batch[new_node_1], petdni_batch[new_node_2]
                    ] in dual_edges)
                self.assertTrue(
                    [petdni_batch[new_node_2], petdni_batch[new_node_1]
                    ] in dual_edges)
            # - Dual-graph configuration B.
            for edge in [[(0, 1), (1, 2)], [(0, 1), (1, 5)], [(0, 7), (7, 6)],
                         [(1, 0), (0, 7)], [(1, 2), (2, 3)], [(1, 5), (5, 4)],
                         [(1, 5), (5, 6)], [(2, 1), (1, 5)], [(2, 3), (3, 4)],
                         [(3, 4), (4, 5)], [(4, 5), (5, 6)], [(5, 6), (6, 7)]]:
                new_node_1 = (edge[0][0] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx + num_primal_nodes_A,
                              edge[0][1] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx + num_primal_nodes_A)
                new_node_2 = (edge[1][0] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx + num_primal_nodes_A,
                              edge[1][1] +
                              (num_primal_nodes_B + num_primal_nodes_A) *
                              batch_sample_idx + num_primal_nodes_A)
                self.assertTrue(
                    [petdni_batch[new_node_1], petdni_batch[new_node_2]
                    ] in dual_edges)
                self.assertTrue(
                    [petdni_batch[new_node_2], petdni_batch[new_node_1]
                    ] in dual_edges)
                self.assertTrue([
                    petdni_batch[new_node_2[::-1]], petdni_batch[
                        new_node_1[::-1]]
                ] in dual_edges)
                self.assertTrue([
                    petdni_batch[new_node_1[::-1]], petdni_batch[
                        new_node_2[::-1]]
                ] in dual_edges)

        # Tests on the primal graph.
        self.assertEqual(primal_graph_batch.num_edges,
                         (num_primal_edges_A + num_primal_edges_B) *
                         num_replicas)
        primal_edges = primal_graph_batch.edge_index.t().tolist()
        for batch_sample_idx in range(num_replicas):
            # - Dual-graph configuration A.
            for edge in [(0, 1), (0, 7), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5),
                         (5, 6), (6, 7)]:
                new_edge = [
                    edge[0] + (num_primal_nodes_B + num_primal_nodes_A) *
                    batch_sample_idx, edge[1] +
                    (num_primal_nodes_B + num_primal_nodes_A) * batch_sample_idx
                ]
                self.assertTrue(new_edge in primal_edges)
                self.assertTrue(new_edge[::-1] in primal_edges)
            # - Dual-graph configuration B.
            for edge in [(0, 1), (0, 7), (1, 2), (1, 5), (2, 3), (3, 4), (4, 5),
                         (5, 6), (6, 7)]:
                new_edge = [
                    edge[0] + (num_primal_nodes_B + num_primal_nodes_A) *
                    batch_sample_idx + num_primal_nodes_A, edge[1] +
                    (num_primal_nodes_B + num_primal_nodes_A) * batch_sample_idx
                    + num_primal_nodes_A
                ]
                self.assertTrue(new_edge in primal_edges)
