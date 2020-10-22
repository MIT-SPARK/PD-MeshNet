import numpy as np
import pymesh
import torch
import torch_geometric
import warnings

from pd_mesh_net.utils import (dihedral_angle_and_local_indices_edges,
                               preprocess_mesh)


class GraphCreator:
    r"""Creates the primal- ("simplex mesh", or what is commonly known as the
    "dual graph" of the triangulation seen as a planar graph) and dual- ("medial
    graph" of the triangulation) graph from a given input mesh.

    Args:
        mesh (pymesh.Mesh.Mesh, optional): If not None, input mesh. Note: if
            this option is used, it will be assumed that the mesh has already
            been preprocessed (i.e., with removal of duplicated vertices,
            duplicated faces, zero-area faces and optionally non-manifold
            edges). If you wish to perform preprocessing, use function
            `preprocess_mesh` from `pd_mesh_net.utils`. (default: :obj:`None`)
        mesh_filename (str, optional): If not None, filename of the input mesh.
            Note: if this option is used, it will be assumed that the mesh has
            not been preprocessed yet, so preprocessing will be applied (i.e.,
            with removal of duplicated vertices, duplicated faces, zero-area
            faces and optionally non-manifold edges, cf. argument
            `prevent_nonmanifold_edges`). (default: :obj:`None`)
        single_dual_nodes (bool, optional): If True, for each undirected primal
            edge `{i, j}` (i.e., for each pair of opposite directed primal edges
            `i->j` and `j->i`) a single dual node `{i, j}` is created. If False,
            for each primal edge `{i, j}`, a pair of dual nodes `i->j` and
            `j->i` are created, corresponding to the primal edges `i->j` and
            `j->i` respectively. (default: :obj:`True`)
        undirected_dual_edges (bool, optional): If True, dual edges are created
            as pairs of opposite directed edges (i.e., they can be seen an
            undirected edge); otherwise, dual edges are created as directed,
            i.e., each dual edge has no opposite edge. In particular:
            - If single dual nodes are created (cf. argument
              `single_dual_nodes`), this argument can only be True. Each dual
              node `{i, j}` is connected via undirected edges to the dual nodes
              of the form `{i, m}` and `{j, n}`, with `m` neighboring face of
              `i` different from `j` and `n` neighboring face of `j` different
              from `i`. That is to say, the following directed dual edges are
              created: `{i, m}->{i, j}`, `{i, j}->{i, m}`, `{j, n}->{i, j}`,
              `{i, j}->{j, n}`. We refer to this configuration of the dual graph
              as configuration A.
            - If double dual nodes are created (cf. argument
              `single_dual_nodes`) and this argument is True, each dual node
              `i->j` is connected via undirected edges to the dual nodes of the
              form `m->i` and `j->n`, with `m` neighboring face of `i` different
              from `j` and `n` neighboring face of `j` different from `i`. That
              is to say, the following directed dual edges are created:
              `(m->i)->(i->j)`, `(i->j)->(m->i), `(i->j)->(j->n)`,
              `(j->n)->(i->j)`. We refer to this configuration of the dual graph
              as configuration B.
            - If double dual nodes are created (cf. argument
              `single_dual_nodes`) and this argument is False, each dual node
              `i->j` is connected via an incoming directed edge to the dual
              nodes of the form `m->i` and via an outgoing directed edge to the
              dual nodes of the form `j->n`, with `m` neighboring face of `i`
              different from `j` and `n` neighboring face of `j` different from
              `i`. That is to say, the following directed dual edges are
              created: `(m->i)->(i->j)`, `(i->j)->(j->n)`. We refer to this
              configuration of the dual graph as configuration C.
            (default: :obj:`True`)
        prevent_nonmanifold_edges (bool, optional): If True, the faces of the
            input mesh are parsed one at a time and only considered if adding
            them to the set of previous faces does not cause any edges to become
            non-manifold. (default: :obj:`True`)
        primal_features_from_dual_features (bool, optional): Whether or not the
            features of the nodes in the primal graph should be computed from
            the features of the nodes in the dual graph. In particular, drawing
            inspiration from MeshCNN (https://github.com/ranahanocka/MeshCNN/):
            - If True:
                - In dual-graph configurations B and C (cf. argument
                  `undirected_dual_edges`), the dual-graph node `i->j` is
                  assigned as features:
                  1. The dihedral angle between faces `i` and `j` in the
                     original triangular mesh (referred to as
                     :math:`\alpha_{ij}`);
                  2. The ratio (referred to as :math:`k_{ij}`) between the
                     length of the edge between faces `i` and `j` (referred to
                     as :math:`e_{ij}`) and the corresponding height in face i
                     in the original triangular mesh;
                  3. The ratio (referred to as :math:`r_{ij}_{bef}`) between
                     :math:`e_{ij}` and the length of the one, among the two
                     remaining half-edges in face `i`, that has as end point the
                     starting point of the half-edge between faces `i` and `j`.
                     For instance, if face `i` in the original triangular mesh
                     has vertices :math:`v_1`, :math:`v_2` and :math:`v_3`,
                     half-edges :math:`v_1 \rightarrow v_2`,
                     :math:`v_2 \rightarrow v_3` and
                     :math:`v_3 \rightarrow v_1`, and if the half-edge between
                     face `i` and face `j` is :math:`v_2 \rightarrow v_3`, then
                     :math:`r_{ij}_{bef} = \frac{\|v_2 \rightarrow v_3\|}{
                         \|v_1 \rightarrow v_2\|}`, where :math:`\|\bullet\|`
                     denotes the length of an half-edge;
                  4. The ratio (referred to as :math:`r_{ij}_{aft}`) between
                     :math:`e_{ij}` and the length of the one, among the two
                     remaining half-edges in face `i`, that has as starting
                     point the end point of the half-edge between faces `i` and
                     `j`. For instance, if face `i` in the original triangular
                     mesh has vertices :math:`v_1`, :math:`v_2` and :math:`v_3`,
                     half-edges :math:`v_1 \rightarrow v_2`,
                     :math:`v_2 \rightarrow v_3` and
                     :math:`v_3 \rightarrow v_1`, and if the half-edge between
                     face `i` and face `j` is :math:`v_2 \rightarrow v_3`, then
                     :math:`r_{ij}_{aft} = \frac{\|v_2 \rightarrow v_3\|}{
                         \|v_3 \rightarrow v_1\|}`, where :math:`\|\bullet\|`
                     denotes the length of an half-edge;
                  The primal-graph node `i` is then assigned as features:
                  1. The average of the angles :math:`\alpha_{im}` over its
                     neighbors `m`;
                  2. The average of the ratios :math:`k_{im}` over its neighbors
                     `m`;
                  3. The average of the ratios :math:`r_{im}_{bef}` over its
                     neighbors `m`;
                  4. The average of the ratios :math:`r_{im}_{aft}` over its
                     neighbors `m`.
                - In dual-graph configuration A (cf. argument
                  `undirected_dual_edges`), the dual-graph node `{i, j}` is
                  assigned as features:
                  1. The dihedral angle between faces `i` and `j` in the
                     original triangular mesh (referred to as
                     :math:`\alpha_{ij}`);
                  2.,3. The two ratios - sorted in increasing order - between
                     the edge between faces `i` and `j` and the corresponding
                     height in face `i` and `j` respectively in the original
                     triangular mesh (referred to as :math:`k_{ij}` and as
                     :math:`k_{ji}`);
                  4.,5. The two ratios :math:`r_{ij}_{bef}` and
                     :math:`r_{ji}_{bef}` defined as above, sorted in the same
                     order as 2.,3., i.e., if :math:`k_{ij} < k_{ji}` then
                     4. = :math:`r_{ij}_{bef}` and 5. = :math:`r_{ji}_{bef}`,
                     otherwise 4. = :math:`r_{ji}_{bef}` and
                     5. = :math:`r_{ij}_{bef}`;
                  6.,7. The two ratios :math:`r_{ij}_{aft}` and
                     :math:`r_{ji}_{aft}` defined as above, sorted in the same
                     order as 2.,3., i.e., if :math:`k_{ij} < k_{ji}` then
                     4. = :math:`r_{ij}_{aft}` and 5. = :math:`r_{ji}_{aft}`,
                     otherwise 4. = :math:`r_{ji}_{aft}` and
                     5. = :math:`r_{ij}_{aft}`.
                  Primal-graph nodes are then assigned as features the same
                  features as in the case of dual-graph configurations B and C.
            - If False: the dual-graph nodes are assigned the same features as
                the case above. As for the primal-graph nodes, node `i` in the
                primal graph, corresponding to face `i` in the original
                triangular mesh, is assigned as features the ratio between the
                area of face `i` and the average of the areas of all the faces
                in the mesh.
            (default: :obj:`False`)

    Attributes:
        dual_graph (torch_geometric.data.data.Data): Set to None at
            initialization; when the method :obj:`create_dual_graph` is called,
            stores the dual graph computed from the input mesh.
        primal_graph (torch_geometric.data.data.Data): Set to None at
            initialization; when the method :obj:`create_primal_graph` is
            called, stores the primal graph computed from the input mesh.
        primal_edge_to_dual_node_idx (dict): Dictionary that associates a tuple,
            encoding an edge e in the primal graph, to the index of the node in
            the dual graph that corresponds to the edge e.
        dual_node_idx_to_primal_edge (dict): Dictionary that associates the
            index of a node in the dual graph to a tuple encoding the
            corresponding edge in the primal graph.
    """

    def __init__(self,
                 mesh=None,
                 mesh_filename=None,
                 single_dual_nodes=True,
                 undirected_dual_edges=True,
                 prevent_nonmanifold_edges=True,
                 primal_features_from_dual_features=False):
        if (single_dual_nodes):
            assert (undirected_dual_edges), (
                "The configuration with single dual nodes and directed dual "
                "edges is not valid. Please choose a different configuration.")
        self.__dual_graph = None
        self.__primal_graph = None
        self.__single_dual_nodes = single_dual_nodes
        self.__undirected_dual_edges = undirected_dual_edges
        self.__prevent_nonmanifold_edges = prevent_nonmanifold_edges
        (self.__primal_features_from_dual_features
        ) = primal_features_from_dual_features
        # The following dictionaries associate each edge in the primal graph to
        # the index of the corresponding node in the dual graph, and viceversa.
        # If dual nodes are single (cf. argument `single_dual_nodes`), a primal
        # edge i->j gets indexed by the increasingly-ordered pair of the indices
        # of their endpoints, i.e., the associated dual node is {i, j} if i < j
        # and {j, i} otherwise.
        self.__primal_edge_to_dual_node_idx = dict()
        self.__dual_node_idx_to_primal_edge = dict()
        if (mesh is None):
            assert (mesh_filename is not None)
            # Load input mesh.
            self.__input_mesh = pymesh.load_mesh(mesh_filename)
        else:
            assert (mesh_filename is None)
            self.__input_mesh = mesh
        self.__input_mesh.enable_connectivity()

        if (mesh is None):
            # Remove duplicated vertices, duplicated faces, degenerate faces
            # and, if required, faces with non-manifold edges.
            self.__input_mesh = preprocess_mesh(
                input_mesh=self.__input_mesh,
                prevent_nonmanifold_edges=self.__prevent_nonmanifold_edges)

    @property
    def dual_graph(self):
        return self.__dual_graph

    @property
    def primal_graph(self):
        return self.__primal_graph

    @property
    def primal_edge_to_dual_node_idx(self):
        if (self.__dual_graph is None):
            warning_msg = (
                'Graphs have not been built yet. Please call the method '
                'create_graphs() first.')
            warnings.warn(warning_msg, Warning)
            return None
        else:
            return self.__primal_edge_to_dual_node_idx

    @property
    def dual_node_idx_to_primal_edge(self):
        if (self.__dual_graph is None):
            warning_msg = (
                'Graphs have not been built yet. Please call the method '
                'create_graphs() first.')
            warnings.warn(warning_msg, Warning)
            return None
        else:
            return self.__dual_node_idx_to_primal_edge

    def __add_node_to_dual_graph(self, corresponding_primal_edge):
        r"""Adds either one node or two nodes (depending on the settings of the
        GraphCreator instance) to the dual graph, corresponding to the given
        input primal edge. Features corresponding to the newly added node(s) are
        also computed (cf. argument `primal_features_from_dual_features` in docs
        of the class). If the node/nodes is/are already in the graph, no action
        is performed.

        Args:
            corresponding_primal_edge (numpy.array of shape :obj:`[2,]` or tuple
                of length 2): Edge in the primal graph - expressed as the
                indices of its two endpoints - from which one corresponding node
                (or two, if double dual nodes) is created and added to the dual
                graph.

        Returns:
            None.
        """
        assert (len(corresponding_primal_edge) == 2)

        if (not self.__single_dual_nodes):
            primal_edge = tuple(corresponding_primal_edge)
            if (primal_edge in self.__primal_edge_to_dual_node_idx):
                # Node already in the graph.
                return
            # Original direction.
            curr_dual_node_idx = len(self.__primal_edge_to_dual_node_idx)
            self.__primal_edge_to_dual_node_idx[
                primal_edge] = curr_dual_node_idx
            self.__dual_node_idx_to_primal_edge[
                curr_dual_node_idx] = primal_edge
            # Reverse direction.
            curr_dual_node_idx = len(self.__primal_edge_to_dual_node_idx)
            reverse_primal_edge = primal_edge[::-1]
            self.__primal_edge_to_dual_node_idx[
                reverse_primal_edge] = curr_dual_node_idx
            self.__dual_node_idx_to_primal_edge[
                curr_dual_node_idx] = reverse_primal_edge
            # Compute the features of the dual nodes just added.
            ((edge_height_ratio_1, edge_height_ratio_2),
             (edge_previous_edge_ratio_1, edge_previous_edge_ratio_2),
             (edge_subsequent_edge_ratio_1,
              edge_subsequent_edge_ratio_2), dihedral_angle) = (
                  self.__edgelength_height_ratio_dihedral_angle_edgeedge_ratio(
                      primal_edge[0], primal_edge[1], True))
            self.__dual_features[curr_dual_node_idx - 1] = [
                dihedral_angle, edge_height_ratio_1, edge_previous_edge_ratio_1,
                edge_subsequent_edge_ratio_1
            ]
            self.__dual_features[curr_dual_node_idx] = [
                dihedral_angle, edge_height_ratio_2, edge_previous_edge_ratio_2,
                edge_subsequent_edge_ratio_2
            ]
            self.__num_dual_features_assigned += 2
        else:
            primal_edge = tuple(corresponding_primal_edge)
            if (primal_edge in self.__primal_edge_to_dual_node_idx):
                # Node already in the graph.
                return
            # Unique direction, with the primal-graph edge being indexed by the
            # increasingly-sorted pair of indices of its endpoints.
            # - Since primal edges are always directed and thus get added to the
            #   dictionary primal_edge_to_dual_node_idx in pairs of opposite
            #   edges, the number of elements in the dictionary must be even.
            assert (len(self.__primal_edge_to_dual_node_idx) % 2 == 0)
            curr_dual_node_idx = len(self.__primal_edge_to_dual_node_idx) // 2
            self.__primal_edge_to_dual_node_idx[
                primal_edge] = curr_dual_node_idx
            # Reverse direction.
            reverse_primal_edge = primal_edge[::-1]
            self.__primal_edge_to_dual_node_idx[
                reverse_primal_edge] = curr_dual_node_idx
            # Dual-node to undirected primal-edge.
            self.__dual_node_idx_to_primal_edge[curr_dual_node_idx] = tuple(
                sorted(primal_edge))
            # Compute the features of the dual node just added.
            primal_edge = tuple(sorted(primal_edge))
            ((edge_height_ratio_1, edge_height_ratio_2),
             (edge_previous_edge_ratio_1, edge_previous_edge_ratio_2),
             (edge_subsequent_edge_ratio_1,
              edge_subsequent_edge_ratio_2), dihedral_angle) = (
                  self.__edgelength_height_ratio_dihedral_angle_edgeedge_ratio(
                      primal_edge[0], primal_edge[1], True))
            if (edge_height_ratio_1 < edge_height_ratio_2):
                self.__dual_features[curr_dual_node_idx] = [
                    dihedral_angle, edge_height_ratio_1, edge_height_ratio_2,
                    edge_previous_edge_ratio_1, edge_previous_edge_ratio_2,
                    edge_subsequent_edge_ratio_1, edge_subsequent_edge_ratio_2
                ]
            else:
                self.__dual_features[curr_dual_node_idx] = [
                    dihedral_angle, edge_height_ratio_2, edge_height_ratio_1,
                    edge_previous_edge_ratio_2, edge_previous_edge_ratio_1,
                    edge_subsequent_edge_ratio_2, edge_subsequent_edge_ratio_1
                ]

            self.__num_dual_features_assigned += 1

    def __edgelength_height_ratio_dihedral_angle_edgeedge_ratio(
        self, face_idx_1, face_idx_2, use_both_faces=False):
        r"""Given the indices of two faces that share an edge, computes the
        following features:
        - The ratio :math:`k_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}`
          between :math:`l_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}` - i.e.,
          the length of the edge shared by the faces with indices
          :math:`\textrm{face\_idx}_1` and :math:`\textrm{face\_idx}_2` in the
          original triangular mesh - and :math:`h_{\textrm{face\_idx}_1,
          :math:`\textrm{face\_idx}_2`}` - i.e., the corresponding height in the
          face with index :math:`\textrm{face\_idx}_1` in the original mesh. If
          `use_both_faces` is :obj:`True`, also the ratio
          :math:`k_{\textrm{face\_idx}_3, \textrm{face\_idx}_1}` between
          :math:`l_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}` and
          :math:`h_{\textrm{face\_idx}_2, :math:`\textrm{face\_idx}_1`}` - i.e.,
          the corresponding height in the face with index
          :math:`\textrm{face\_idx}_2` in the original mesh - is computed;
        - The dihedral angle
          :math:`\alpha_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}` between
          the two faces.
        - The edge-to-previous-edge ratio
          :math:`r_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}_{bef}` as
          defined in the docs of the class. If `use_both_faces` is :obj:`True`,
          also the ratio
          :math:`r_{\textrm{face\_idx}_2, \textrm{face\_idx}_1}_{bef}` is
          computed.
        - The edge-to-subsequent-edge ratio
          :math:`r_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}_{aft}` as
          defined in the docs of the class. If `use_both_faces` is :obj:`True`,
          also the ratio
          :math:`r_{\textrm{face\_idx}_2, \textrm{face\_idx}_1}_{aft}` is
          computed.

        Args:
            face_idx_1, face_idx_2 (int): Indices in the original mesh of the
                two faces that define the ratio(s) described above.
            use_both_faces (bool, optional): If :obj:`False`, only
                :math:`h_{\textrm{face\_idx}_1, :math:`\textrm{face\_idx}_2`}`
                is considered for the ratio. Otherwise, also
                :math:`h_{\textrm{face\_idx}_2, :math:`\textrm{face\_idx}_1`}`
                is considered.

        Returns:
            A tuple, in which:
            - The first element is either a single float or a tuple of float,
              representing the edge-length/height ratio(s) described above. If
              `use_both_faces` is :obj:`True`, both the two ratios
              described above are returned, with the first element in the tuple
              corresponding to the ratio computed w.r.t.
              :math:`h_{\textrm{face\_idx}_1, :math:`\textrm{face\_idx}_2`}` and
              the second element in the tuple corresponding to the ratio
              computed w.r.t.
              :math:`h_{\textrm{face\_idx}_2, :math:`\textrm{face\_idx}_1`}`.
            - The second element is either a single float or a tuple of float,
              representing the edge-to-previous-edge ratio(s) described above.
              If `use_both_faces` is :obj:`True`, both the two ratios
              described above are returned, with the first element in the tuple
              corresponding to
              :math:`r_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}_{bef}` and
              the second element in the tuple corresponding to
              :math:`r_{\textrm{face\_idx}_2, \textrm{face\_idx}_1}_{bef}`.
            - The third element is either a single float or a tuple of float,
              representing the edge-to-subsequent-edge ratio(s) described above.
              If `use_both_faces` is :obj:`True`, both the two ratios
              described above are returned, with the first element in the tuple
              corresponding to
              :math:`r_{\textrm{face\_idx}_1, \textrm{face\_idx}_2}_{aft}` and
              the second element in the tuple corresponding to
              :math:`r_{\textrm{face\_idx}_2, \textrm{face\_idx}_1}_{aft}`.
            - The fourth element is a float representing the dihedral angle
              between the two faces.
        """
        # Obtain dihedral angle and local information about the two faces.
        (dihedral_angle,
         local_indices_common_edge) = dihedral_angle_and_local_indices_edges(
             mesh=self.__input_mesh,
             face_indices=[face_idx_1, face_idx_2],
             face_normals=self.__face_normals[(face_idx_1, face_idx_2), :])

        if (dihedral_angle is None or local_indices_common_edge is None):
            # The two faces share a half-edge with the same direction, i.e.,
            # the edge that they share is non-manifold. The function is called
            # again with the second face flipped. Note: this is arbitrary (the
            # first face could be flipped instead), but there is no way to
            # define a dihedral angle between two faces that share the same
            # half-edge, so a decision needs to be taken.
            face_normals = self.__face_normals[(face_idx_1, face_idx_2), :]
            face_1 = self.__input_mesh.faces[face_idx_1].tolist()
            face_2 = self.__input_mesh.faces[face_idx_2][::-1].tolist()
            (dihedral_angle, local_indices_common_edge
            ) = dihedral_angle_and_local_indices_edges(
                mesh=self.__input_mesh,
                faces=[face_1, face_2],
                face_normals=face_normals)
            assert (dihedral_angle is not None and
                    local_indices_common_edge is not None)
        else:
            face_1 = self.__input_mesh.faces[face_idx_1].tolist()
            face_2 = self.__input_mesh.faces[face_idx_2].tolist()
        # - Half-edge in face 1.
        common_edge_seen_from_face_1 = self.__input_mesh.vertices[face_1[
            (local_indices_common_edge[0] + 1) %
            3]] - self.__input_mesh.vertices[face_1[
                local_indices_common_edge[0]]]
        edge_length = np.linalg.norm(common_edge_seen_from_face_1)
        # - Previous half-edge in face 1.
        previous_edge_in_face_1 = self.__input_mesh.vertices[face_1[
            local_indices_common_edge[0]]] - self.__input_mesh.vertices[face_1[
                local_indices_common_edge[0] - 1]]
        previous_edge_in_face_1_length = np.linalg.norm(previous_edge_in_face_1)
        # - Subsequent half-edge in face 1.
        subsequent_edge_in_face_1 = self.__input_mesh.vertices[face_1[
            (local_indices_common_edge[0] + 2) %
            3]] - self.__input_mesh.vertices[face_1[
                (local_indices_common_edge[0] + 1) % 3]]
        subsequent_edge_in_face_1_length = np.linalg.norm(
            subsequent_edge_in_face_1)

        # Since we have the face areas, and since the area A of a triangle can
        # be found as A = e * h / 2, where e is the edge length and h is the
        # length of the corresponding height, the edge/height ratio e/h can be
        # found as e^2 / (2 * A).
        edge_height_ratio_1 = edge_length**2 / (
            2 * self.__face_areas[face_idx_1].item())

        # Find the edge- previous-edge/subsequent-edge ratios.
        (edge_previous_edge_ratio_1
        ) = edge_length / previous_edge_in_face_1_length
        (edge_subsequent_edge_ratio_1
        ) = edge_length / subsequent_edge_in_face_1_length

        if (use_both_faces):
            edge_height_ratio_2 = edge_length**2 / (
                2 * self.__face_areas[face_idx_2].item())
            # - Previous half-edge in face 2.
            previous_edge_in_face_2 = self.__input_mesh.vertices[face_2[
                (local_indices_common_edge[1] - 1) %
                3]] - self.__input_mesh.vertices[face_2[
                    (local_indices_common_edge[1] - 2) % 3]]
            previous_edge_in_face_2_length = np.linalg.norm(
                previous_edge_in_face_2)
            # - Subsequent half-edge in face 2.
            subsequent_edge_in_face_2 = self.__input_mesh.vertices[face_2[
                (local_indices_common_edge[1] + 1) %
                3]] - self.__input_mesh.vertices[face_2[
                    local_indices_common_edge[1]]]
            subsequent_edge_in_face_2_length = np.linalg.norm(
                subsequent_edge_in_face_2)

            # Find the edge- previous-edge/subsequent-edge ratios.
            (edge_previous_edge_ratio_2
            ) = edge_length / previous_edge_in_face_2_length
            (edge_subsequent_edge_ratio_2
            ) = edge_length / subsequent_edge_in_face_2_length

            return ((edge_height_ratio_1, edge_height_ratio_2),
                    (edge_previous_edge_ratio_1, edge_previous_edge_ratio_2),
                    (edge_subsequent_edge_ratio_1,
                     edge_subsequent_edge_ratio_2), dihedral_angle)
        else:
            return (edge_height_ratio_1, edge_previous_edge_ratio_1,
                    edge_subsequent_edge_ratio_1, dihedral_angle)

    def __compute_primal_features(self, sum_face_areas=None):
        r"""Computes the features on the nodes of the primal graph. 

        Args:
            sum_face_areas (float, optional): Sum of the areas of the faces in
                the original triangular mesh. Required to be non-None if
                `self.__primal_features_from_dual_features` is :obj:`False`.
                (default: :obj:`None`)

        Returns:
            None.

        Raises:
            - Raises an error if `self.__primal_features_from_dual_features` is
              :obj:`True` and the features of the nodes of the dual graph have
              not been computed yet.
            - Raises an error if sum_face_areas is None and
              `self.__primal_features_from_dual_features` is :obj:`False`.
        """
        assert (not (self.__primal_features_from_dual_features and
                     (self.__num_dual_features_assigned == 0)))
        assert (not (sum_face_areas is None and
                     not self.__primal_features_from_dual_features))

        for primal_node in range(len(self.__input_mesh.faces)):
            if (self.__primal_features_from_dual_features):
                # - Find neighbors of the primal node.
                (neighboring_faces
                ) = self.__input_mesh.get_face_adjacent_faces(primal_node)
                # - Initialize features.
                dihedral_angles = None
                edge_height_ratios = None
                # Primal node i is assigned as features:
                # 1. The average of the diehedral angles \alpha_{im} over its
                #    neighbors m;
                # 2. The average of the edge-height ratios k_{im} over its
                #    neighbors m;
                # 3. The average of the edge-to-previous-edge ratios
                #    r_{im}_{bef} over its neighbors m;
                # 4. The average of the edge-to-subsequent-edge ratios
                #    r_{im}_{aft} over its neighbors m.
                # - Find the indices in the dual graph of the dual nodes
                #   corresponding to the edges i->m in the primal graph.
                if (self.__single_dual_nodes):
                    compute_features = (
                        self.
                        __edgelength_height_ratio_dihedral_angle_edgeedge_ratio)
                    dihedral_angles = np.zeros([len(neighboring_faces,)])
                    edge_height_ratios = np.zeros([len(neighboring_faces,)])
                    edge_previous_edge_ratios = np.zeros(
                        [len(neighboring_faces,)])
                    edge_subsequent_edge_ratios = np.zeros(
                        [len(neighboring_faces,)])
                    for neighbor_idx, neighbor in enumerate(neighboring_faces):
                        (edge_height_ratio, edge_previous_edge_ratio,
                         edge_subsequent_edge_ratio,
                         dihedral_angle) = compute_features(
                             face_idx_1=primal_node,
                             face_idx_2=neighbor,
                             use_both_faces=False)
                        dihedral_angles[neighbor_idx] = dihedral_angle
                        edge_height_ratios[neighbor_idx] = edge_height_ratio
                        edge_previous_edge_ratios[
                            neighbor_idx] = edge_previous_edge_ratio
                        edge_subsequent_edge_ratios[
                            neighbor_idx] = edge_subsequent_edge_ratio
                else:
                    corresponding_dual_nodes_indices = [
                        self.__primal_edge_to_dual_node_idx[(primal_node,
                                                             neighbor)]
                        for neighbor in neighboring_faces
                    ]
                    assert (
                        self.__dual_features[corresponding_dual_nodes_indices].
                        shape[1] == 4)
                    dihedral_angles = self.__dual_features[
                        corresponding_dual_nodes_indices, 0]
                    edge_height_ratios = self.__dual_features[
                        corresponding_dual_nodes_indices, 1]
                    edge_previous_edge_ratios = self.__dual_features[
                        corresponding_dual_nodes_indices, 2]
                    edge_subsequent_edge_ratios = self.__dual_features[
                        corresponding_dual_nodes_indices, 3]

                # - Assign feature to primal node.
                self.__primal_features[primal_node, :] = [
                    np.mean(dihedral_angles),
                    np.mean(edge_height_ratios),
                    np.mean(edge_previous_edge_ratios),
                    np.mean(edge_subsequent_edge_ratios)
                ]
            else:
                # Primal nodes are assigned as feature simply the ratio between
                # the corresponding face in the original triangular mesh and the
                # sum of the areas of all the faces.
                self.__primal_features[primal_node] = self.__face_areas[
                    primal_node] / sum_face_areas

    def __match_neighboring_faces_to_face_edges(self, face_idx):
        r"""Given the index of a face, returns the indices of the neighboring
        faces in the order specified by the half-edges in the face. In
        particular, letting A be the input face, represented in the input mesh
        (cf. class attribute `__input_mesh`) as `[a, b, c]` - thus with
        consecutive half-edges a->b, b->c and c->a - and letting B, C and D be
        the faces sharing with A respectively edge a--b, edge b--c and edge
        c--a, the function returns the indices of faces B, C and D in this
        order. If any of the edges a--b, b--c and c--a is a boundary edge of the
        mesh, the corresponding output face index will be :obj:`None`.

        Args:
            face_idx (int): Index of the face of which to return the indices of
                the neighboring faces in the order defined above.

        Returns:
            matched_neighboring_faces (list of int/None): The indices of the
                neighboring faces of the input face, as defined above.
        """
        # Find the neighboring faces of the input face.
        neighboring_faces = self.__input_mesh.get_face_adjacent_faces(face_idx)
        face = self.__input_mesh.faces[face_idx]
        assert (len(face) == 3)
        matched_neighboring_faces = []
        faces_to_assign_to_edges = set(neighboring_faces)
        for i in range(3):
            edge = (face[i], face[(i + 1) % 3])
            found_face = False
            # - Search for the opposite half-edge in the neighboring face.
            for neighboring_face_idx in faces_to_assign_to_edges.copy():
                if (found_face):
                    break
                neighboring_face = self.__input_mesh.faces[neighboring_face_idx]
                for j in range(3):
                    if (neighboring_face[j] == face[i] and
                            neighboring_face[(j - 1) % 3] == face[(i + 1) % 3]):
                        # - Found face with opposite half-edge.
                        matched_neighboring_faces.append(neighboring_face_idx)
                        found_face = True
                        faces_to_assign_to_edges.remove(neighboring_face_idx)
                        break
            if (not found_face):
                # The edge is boundary.
                matched_neighboring_faces.append(None)

        return matched_neighboring_faces

    def __create_dual_edges_ordered(self, face_idx):
        r"""Creates the edges in the dual graph (medial graph) that connect the
        dual node(s) associated to the primal edges in the input face to its
        (their) neighbors in the dual graph. As opposed to
        :obj:`__create_dual_edges_unordered`, for each dual node a strict order
        is followed. Letting A and B be two adjacent faces of the mesh (primal
        nodes), M, N, B be the adjacent faces of A in counter-clockwise order
        and J, K, A be the adjacent faces of B in counter-clockwise order, let
        us consider the cases of the three possible dual-graph configurations:
        - Dual-graph configuration A (single dual nodes, undirected dual edges):
          Dual edges incoming to the dual node {A, B} are added to edge list in
          the following order: {A, M}->{A, B}, {A, N}->{A, B}, {B, J}->{A, B},
          {B, K}->{A, B}.
        - Dual-graph configuration B (double dual nodes, undirected dual edges):
          Dual edges incoming to the dual node A->B are added to edge list in
          the following order: (M->A)->(A->B), (N->A)->(A->B), (B->J)->(A->B),
          (B->K)->(A->B). Dual edges incoming to the dual node B->A are added to
          edge list in the following order: (J->B)->(B->A), (K->B)->(B->A),
          (A->M)->(B->A), (A->N)->(B->A).
        - Dual-graph configuration C (double dual nodes, directed dual edges):
          Dual edges incoming to the dual node A->B are added to edge list in
          the following order: (M->A)->(A->B), (N->A)->(A->B). Dual edges
          incoming to the dual node B->A are added to edge list in the following
          order: (J->B)->(B->A), (K->B)->(B->A).
        Note: in all the cases, each dual node other than the 'special' one
        required to represent boundary edges of the mesh - has also a self-loop
        added at the end of the list of its incoming edges, but directly in the
        :obj:`create_graphs` method.

        Args:
            face_idx (int): The dual nodes associated to the edges in the face
                with this index will be connected to other nodes in the dual
                graph, via new dual edges.

        Returns:
            None.
        """
        # For each edge in the face, find the neighboring face with which the
        # edge is shared.
        local_idx_edge_to_neighboring_face_idx = (
            self.__match_neighboring_faces_to_face_edges(face_idx))

        # Find the neighboring faces of the two faces A and B in the
        # original triangular mesh that correspond to the current
        # primal edge;
        if (self.__single_dual_nodes):
            # Dual-graph configuration A.
            # - Take all primal edges associated to the current face.
            for (neighbor_idx_in_neighborhood, neighboring_face_idx
                ) in enumerate(local_idx_edge_to_neighboring_face_idx):
                if (neighboring_face_idx is None):
                    # The edge being considered is boundary, hence there is
                    # no dual node associated to it.
                    continue

                primal_edge = (face_idx, neighboring_face_idx)
                # For two neighboring faces with indices i < j, PyMesh only
                # creates the edge i->j. Therefore, create the opposite
                # directed primal edge j->i, i.e., make every primal edge
                # undirected.
                if (primal_edge[0] < primal_edge[1]):
                    self.__primal_edges = np.vstack(
                        [self.__primal_edges, primal_edge[::-1]])
                # For each edge in the primal graph, add a corresponding
                # node in the dual graph and compute the associated dual
                # features.
                current_dual_node = tuple(sorted(primal_edge))
                self.__add_node_to_dual_graph(current_dual_node)
                # Connect via opposite directed edges the newly-added
                # dual node {A, B} to the dual nodes {A, M} and {A, N},
                # where M and N are neighboring faces of A, if existent, and
                # M, N != B.
                # If the nodes {A, M} and {A, N} are not in the dual
                # graph yet, add them.
                for i in range(1, 3):
                    other_neighbor_idx_in_neighborhood = (
                        neighbor_idx_in_neighborhood + i) % 3
                    other_neighboring_face_idx = (
                        local_idx_edge_to_neighboring_face_idx[
                            other_neighbor_idx_in_neighborhood])
                    if (other_neighboring_face_idx is None):
                        # Boundary edge => Special dual node.
                        other_dual_node = None
                    else:
                        other_primal_edge = (face_idx,
                                             other_neighboring_face_idx)
                        other_dual_node = tuple(sorted(other_primal_edge))
                        self.__add_node_to_dual_graph(other_dual_node)

                    # Add the edges {A, M}->{A, B}/{A, N}->{A, B} to the
                    # dual graph. Note: the opposite directed dual edges
                    # {A, B}->{A, M} and {A, B}->{A, N} will be added when
                    # iterating on the primal edges {A, M} and {A, N}, if
                    # these are not boundary.
                    self.__dual_edges[:, self.__num_dual_edges] = [
                        self.__primal_edge_to_dual_node_idx[other_dual_node],
                        self.__primal_edge_to_dual_node_idx[current_dual_node]
                    ]
                    self.__num_dual_edges += 1
        else:
            if (self.__undirected_dual_edges):
                # Dual-graph configuration B.
                # - Take all primal edges associated to the current face.
                for (neighbor_idx_in_neighborhood, neighboring_face_idx
                    ) in enumerate(local_idx_edge_to_neighboring_face_idx):
                    if (neighboring_face_idx is None):
                        # The edge being considered is boundary, hence there is
                        # no dual node associated to it.
                        continue
                    primal_edge = (face_idx, neighboring_face_idx)
                    if (primal_edge[0] > primal_edge[1]):
                        continue
                    # For two neighboring faces with indices i < j, PyMesh only
                    # creates the edge i->j. Therefore, create the opposite
                    # directed primal edge j->i, i.e., make every primal edge
                    # undirected.
                    self.__primal_edges = np.vstack(
                        [self.__primal_edges, primal_edge[::-1]])
                    # For each edge in the primal graph, add a corresponding
                    # node in the dual graph and compute the associated dual
                    # features.
                    current_dual_node = tuple(sorted(primal_edge))
                    self.__add_node_to_dual_graph(current_dual_node)
                    # Connect the newly-added dual nodes A->B and B->A to the
                    # dual nodes M->A/A->M, N->A/A->N, B->J/J->B, B->K/K->B,
                    # where M and N are neighboring faces of A (if existent), J
                    # and K are neighboring faces of B (if existent), M, N != B
                    # and J, K != A.
                    # If the nodes M->A/A->M, N->A/A->N, B->J/J->B and B->K/K->B
                    # are not in the dual graph yet, add them.
                    for i in range(1, 3):
                        other_neighbor_idx_in_neighborhood = (
                            neighbor_idx_in_neighborhood + i) % 3
                        other_neighboring_face_idx = (
                            local_idx_edge_to_neighboring_face_idx[
                                other_neighbor_idx_in_neighborhood])
                        if (other_neighboring_face_idx is None):
                            # Boundary edge => Special dual node.
                            other_dual_node = None
                        else:
                            other_dual_node = (other_neighboring_face_idx,
                                               face_idx)
                            self.__add_node_to_dual_graph(other_dual_node)

                        # Add the edges (M->A)->(A->B)/(N->A)->(A->B) to the
                        # dual graph.
                        self.__dual_edges[:, self.__num_dual_edges] = [
                            self.
                            __primal_edge_to_dual_node_idx[other_dual_node],
                            self.
                            __primal_edge_to_dual_node_idx[current_dual_node]
                        ]
                        # Add the edges (A->M)->(B->A)/(A->N)->(B->A) to the
                        # dual graph.
                        if (other_dual_node is not None):
                            reverse_other_dual_node = other_dual_node[::-1]
                        else:
                            reverse_other_dual_node = None
                        self.__dual_edges[:, self.__num_dual_edges + 1] = [
                            self.__primal_edge_to_dual_node_idx[
                                reverse_other_dual_node],
                            self.__primal_edge_to_dual_node_idx[
                                current_dual_node[::-1]]
                        ]
                        self.__num_dual_edges += 2

                    # If B is actually a face (i.e., the edge between faces A
                    # and B is not boundary), add the edges to A->B/B->A from
                    # the other neighboring faces of B.
                    if (neighboring_face_idx is not None):
                        idx_neighboring_faces_of_neighbor = (
                            self.__match_neighboring_faces_to_face_edges(
                                face_idx=neighboring_face_idx))
                        # - Find the index of A in the neighbors of the
                        #   neighbors, and take the other two faces.
                        try:
                            idx_face_in_neighbors_of_neighbors = (
                                idx_neighboring_faces_of_neighbor.index(
                                    face_idx))
                        except ValueError:
                            raise ValueError(
                                f"Should have found face no. {face_idx} in the "
                                "the neighboring faces of face "
                                f"{neighboring_face_idx}, but could not find "
                                "it.")
                        for j in range(1, 3):
                            idx_neighbor_of_neighbor = (
                                idx_neighboring_faces_of_neighbor[
                                    (idx_face_in_neighbors_of_neighbors + j) %
                                    3])
                            if (idx_neighbor_of_neighbor is None):
                                # Boundary edge => Special dual node.
                                other_dual_node = None
                            else:
                                other_dual_node = (neighboring_face_idx,
                                                   idx_neighbor_of_neighbor)
                                self.__add_node_to_dual_graph(
                                    sorted(tuple(other_dual_node)))

                            # Add the edges (B->J)->(A->B)/(B->K)->(A->B) to the
                            # dual graph.
                            self.__dual_edges[:, self.__num_dual_edges] = [
                                self.
                                __primal_edge_to_dual_node_idx[other_dual_node],
                                self.__primal_edge_to_dual_node_idx[
                                    current_dual_node]
                            ]
                            # Add the edges (J->B)->(B->A)/(K->B)->(B->A) to the
                            # dual graph.
                            if (other_dual_node is not None):
                                reverse_other_dual_node = (
                                    other_dual_node[::-1])
                            else:
                                reverse_other_dual_node = None
                            (self.__dual_edges[:, self.__num_dual_edges + 1]
                            ) = [
                                self.__primal_edge_to_dual_node_idx[
                                    reverse_other_dual_node],
                                self.__primal_edge_to_dual_node_idx[
                                    current_dual_node[::-1]]
                            ]
                            self.__num_dual_edges += 2
            else:
                # Dual-graph configuration C.
                # - Take all primal edges associated to the current face.
                for (neighbor_idx_in_neighborhood, neighboring_face_idx
                    ) in enumerate(local_idx_edge_to_neighboring_face_idx):
                    if (neighboring_face_idx is None):
                        # The edge being considered is boundary, hence there is
                        # no dual node associated to it.
                        continue
                    primal_edge = (face_idx, neighboring_face_idx)
                    current_dual_node = primal_edge
                    # For two neighboring faces with indices i < j, PyMesh only
                    # creates the edge i->j. Therefore, create the opposite
                    # directed primal edge j->i, i.e., make every primal edge
                    # undirected.
                    if (primal_edge[0] < primal_edge[1]):
                        self.__primal_edges = np.vstack(
                            [self.__primal_edges, primal_edge[::-1]])
                        # For each edge in the primal graph, add a corresponding
                        # node in the dual graph and compute the associated dual
                        # features. Since 'opposite' nodes are added at the same
                        # time, only call the __add method if
                        # primal_edge[0] < primal_edge[1].
                        self.__add_node_to_dual_graph(current_dual_node)
                    # Connect the newly-added dual nodes A->B and B->A to the
                    # dual nodes M->A/A->M, N->A/A->N, where M and N are
                    # neighboring faces of A (if existent), and M, N != B.
                    # If the nodes M->A/A->M and N->A/A->N are not in the dual
                    # graph yet, add them.
                    for i in range(1, 3):
                        other_neighbor_idx_in_neighborhood = (
                            neighbor_idx_in_neighborhood + i) % 3
                        other_neighboring_face_idx = (
                            local_idx_edge_to_neighboring_face_idx[
                                other_neighbor_idx_in_neighborhood])
                        if (other_neighboring_face_idx is None):
                            # Boundary edge => Special dual node.
                            other_dual_node = None
                        else:
                            other_dual_node = (other_neighboring_face_idx,
                                               face_idx)
                            self.__add_node_to_dual_graph(other_dual_node)

                        # Add the edges (M->A)->(A->B)/(N->A)->(A->B) to the
                        # dual graph.
                        self.__dual_edges[:, self.__num_dual_edges] = [
                            self.
                            __primal_edge_to_dual_node_idx[other_dual_node],
                            self.
                            __primal_edge_to_dual_node_idx[current_dual_node]
                        ]
                        self.__num_dual_edges += 1

    def __create_dual_edges_unordered(self, primal_edge):
        r"""Creates the edges in the dual graph (medial graph) that connect the
        dual node(s) associated to the input primal edge to its (their)
        neighbors in the dual graph.

        Args:
            primal_edge (numpy array of shape :obj:`[2,]`): Primal edge the
                associated dual node(s) of which should be connected to other
                nodes in the dual graph, via new dual edges.

        Returns:
            None.
        """
        if (self.__undirected_dual_edges):
            # Find the neighboring faces of the two faces i and j in the
            # original triangular mesh that correspond to the current
            # primal edge;
            if (self.__single_dual_nodes):
                # Dual-graph configuration A.
                # Connect via opposite directed edges the newly-added
                # dual node {i, j} to the dual nodes {i, m} and {j, n},
                # where m and n are neighboring faces of i and j
                # respectively, and m != j and n != i.
                # If the nodes {i, m} and {j, n} are not in the dual
                # graph yet, add them.
                current_dual_node = tuple(primal_edge)
                assert (current_dual_node[0] < current_dual_node[1])
                for primal_node_idx, primal_node in enumerate(primal_edge):
                    other_primal_node_in_primal_edge = primal_edge[
                        (primal_node_idx + 1) % 2]
                    # Find neighboring faces.
                    (neighboring_faces
                    ) = self.__input_mesh.get_face_adjacent_faces(primal_node)
                    for neighboring_face in neighboring_faces:
                        if (neighboring_face !=
                                other_primal_node_in_primal_edge):
                            # Neighboring nodes {i, m} and {j, n}.
                            other_dual_node = tuple(
                                sorted((neighboring_face, primal_node)))
                            self.__add_node_to_dual_graph(other_dual_node)
                            # Add the edges {i, m}->{i, j} and
                            # {j, n}->{i, j} to the dual graph. Note:
                            # the opposite directed dual edges
                            # {i, j}->{i, m} and {i, j}->{j, n} will be
                            # added when iterating on the primal edges
                            # {i, m} and {j, n}.
                            self.__dual_edges[:, self.__num_dual_edges] = [
                                self.
                                __primal_edge_to_dual_node_idx[other_dual_node],
                                self.__primal_edge_to_dual_node_idx[
                                    current_dual_node]
                            ]
                            self.__num_dual_edges += 1
            else:
                # Dual-graph configuration B.
                # Find neighboring faces.
                neighboring_faces = []
                for primal_node in primal_edge:
                    neighboring_faces.append(
                        self.__input_mesh.get_face_adjacent_faces(primal_node))
                assert (len(neighboring_faces) == 2)
                # Connect via opposite directed edges the newly-added
                # dual nodes i->j and j->i, respectively to the dual
                # nodes m->i and j->n and to the dual nodes n->j and
                # i->m, where m and n are neighboring faces of i and j
                # respectively, and m != j and n != i.
                # If the nodes m->i, i->m, n->j and j->n are not in the
                # the dual graph yet, add them.
                # - Neighboring nodes m->i/i->m.
                primal_node = primal_edge[0]
                other_primal_node_in_primal_edge = primal_edge[1]
                current_dual_node = (primal_node,
                                     other_primal_node_in_primal_edge)
                for neighboring_face in neighboring_faces[0]:
                    if (neighboring_face != other_primal_node_in_primal_edge):
                        other_dual_node = (neighboring_face, primal_node)
                        self.__add_node_to_dual_graph(other_dual_node)
                        # Add the edges (m->i)->(i->j) and
                        # (i->m)->(j->i) to the dual graph. Note: the
                        # dual edges (i->j)->(m->i) and (j->i)->(i->m)
                        # will be added when iterating on the primal
                        # edge i->m.
                        self.__dual_edges[:, self.__num_dual_edges] = [
                            self.
                            __primal_edge_to_dual_node_idx[other_dual_node],
                            self.
                            __primal_edge_to_dual_node_idx[current_dual_node]
                        ]
                        self.__dual_edges[:, self.__num_dual_edges + 1] = [
                            self.__primal_edge_to_dual_node_idx[
                                other_dual_node[::-1]],
                            self.__primal_edge_to_dual_node_idx[
                                current_dual_node[::-1]]
                        ]
                        self.__num_dual_edges += 2
                # - Neighboring nodes j->n/n->j.
                primal_node = primal_edge[1]
                other_primal_node_in_primal_edge = primal_edge[0]
                # Note: current_dual_node has not changed: it is still
                # i->j from before.
                for neighboring_face in neighboring_faces[1]:
                    if (neighboring_face != other_primal_node_in_primal_edge):
                        other_dual_node = (primal_node, neighboring_face)
                        self.__add_node_to_dual_graph(other_dual_node)
                        # Add the edges (j->n)->(i->j) and
                        # (n->j)->(j->i) to the dual graph. Note: the
                        # dual edges (i->j)->(j->n) and (j->i)->(n->j)
                        # will be added when iterating on the primal
                        # edge j->n.
                        self.__dual_edges[:, self.__num_dual_edges] = [
                            self.
                            __primal_edge_to_dual_node_idx[other_dual_node],
                            self.
                            __primal_edge_to_dual_node_idx[current_dual_node]
                        ]
                        self.__dual_edges[:, self.__num_dual_edges + 1] = [
                            self.__primal_edge_to_dual_node_idx[
                                other_dual_node[::-1]],
                            self.__primal_edge_to_dual_node_idx[
                                current_dual_node[::-1]]
                        ]
                        self.__num_dual_edges += 2
        else:
            # Directed dual-edges. Dual-graph configuration C.
            assert (not self.__single_dual_nodes)
            # Find the neighboring faces of the two faces i and j in the
            # original triangular mesh that correspond to the current
            # primal edge;
            neighboring_faces = []
            for primal_node in primal_edge:
                neighboring_faces.append(
                    self.__input_mesh.get_face_adjacent_faces(primal_node))
            assert (len(neighboring_faces) == 2)
            # Connect via a single incoming directed edge the
            # newly-added dual nodes i->j and j->i, respectively to the
            # dual nodes m->i and to the dual nodes n->j, where m and n
            # are neighboring faces of i and j respectively, and m != j
            # and n != i.
            # If the nodes m->i and n->j are not in the dual graph yet,
            # add them.
            # - Neighboring nodes m->i.
            primal_node = primal_edge[0]
            other_primal_node_in_primal_edge = primal_edge[1]
            current_dual_node = (primal_node, other_primal_node_in_primal_edge)
            for neighboring_face in neighboring_faces[0]:
                if (neighboring_face != other_primal_node_in_primal_edge):
                    other_dual_node = (neighboring_face, primal_node)
                    self.__add_node_to_dual_graph(other_dual_node)
                    # Add the edges (m->i)->(i->j) to the dual graph.
                    self.__dual_edges[:, self.__num_dual_edges] = [
                        self.__primal_edge_to_dual_node_idx[other_dual_node],
                        self.__primal_edge_to_dual_node_idx[current_dual_node]
                    ]
                    self.__num_dual_edges += 1
            # - Neighboring nodes n->j.
            primal_node = primal_edge[1]
            other_primal_node_in_primal_edge = primal_edge[0]
            # Note: current_dual_node is now j->i.
            current_dual_node = (primal_node, other_primal_node_in_primal_edge)
            for neighboring_face in neighboring_faces[1]:
                if (neighboring_face != other_primal_node_in_primal_edge):
                    other_dual_node = (neighboring_face, primal_node)
                    self.__add_node_to_dual_graph(other_dual_node)
                    # Add the edges (n->j)->(j->i) to the dual graph.
                    self.__dual_edges[:, self.__num_dual_edges] = [
                        self.__primal_edge_to_dual_node_idx[other_dual_node],
                        self.__primal_edge_to_dual_node_idx[current_dual_node]
                    ]
                    self.__num_dual_edges += 1

    def create_graphs(self):
        r"""Creates the primal graph (i.e., the simplex mesh) and the dual graph
        (i.e., the medial graph) out of the original mesh. Features are added to
        the nodes of both the primal and the dual graph (cf. docs of class).

        Args:
            None.

        Returns:
            primal_graph (torch_geometric.data.Data): Primal graph.
            dual_graph (torch_geometric.data.Data): Dual graph.
        """
        if (self.__dual_graph is None and self.__primal_graph is None):
            # Create the primal graph first.
            _, self.__primal_edges = pymesh.mesh_to_dual_graph(
                self.__input_mesh)
            # Obtain face areas and normals.
            self.__input_mesh.add_attribute("face_area")
            self.__input_mesh.add_attribute("face_normal")
            self.__face_areas = self.__input_mesh.get_face_attribute(
                "face_area")
            self.__face_normals = self.__input_mesh.get_face_attribute(
                "face_normal")

            # If the edges of the mesh are manifold, we can predict the maximum
            # number of edges in the dual graph. Otherwise we set originally a
            # value twice as big as the maximum in the manifold case. Note: in
            # both cases, only the edges actually used are inserted in the dual
            # graph.
            if (self.__prevent_nonmanifold_edges):
                multiplicative_factor = 1
            else:
                multiplicative_factor = 2
            # - Case single dual nodes (configuration A): since each node in the
            #   dual graph has 4 incident nodes at most (and in particular, is
            #   4-regular if the original triangular mesh is watertight), and
            #   each undirected edge in the dual graph is shared by two dual
            #   nodes, the number of (undirected edges) in the dual graph is at
            #   most twice the number of nodes in the dual graph, hence of the
            #   number of undirected edges in the primal graph. Therefore, the
            #   number of directed dual edges is twice as much (it is double the
            #   number of undirected dual edges), hence four times the number of
            #   undirected primal edges.
            # - Case double dual nodes, undirected dual edges (configuration B):
            #   the same considerations as above hold, but since primal edges
            #   have not been duplicated yet (will be done further down in the
            #   code), a further 2 multiplicative factor is required to account
            #   for the missing duplication.
            # - Case double dual nodes, directed dual edges (configuration C):
            #   the same considerations as configuration B hold, but for each
            #   undirected dual node of configuration B only a single directed
            #   edge is used in configuration C, hence the 2 multiplicative
            #   factor above is not necessary.
            if (self.__single_dual_nodes):
                self.__dual_edges = np.empty(
                    [2, 4 * multiplicative_factor * len(self.__primal_edges)],
                    dtype=np.int)
            else:
                if (self.__undirected_dual_edges):
                    self.__dual_edges = np.empty([
                        2, 8 * multiplicative_factor * len(self.__primal_edges)
                    ],
                                                 dtype=np.int)
                else:
                    self.__dual_edges = np.empty([
                        2, 4 * multiplicative_factor * len(self.__primal_edges)
                    ],
                                                 dtype=np.int)
            self.__num_dual_edges = 0
            # Initialize the node features of both the primal and the dual
            # graph.
            dim_dual_features = None
            dim_primal_features = None
            if (self.__primal_features_from_dual_features):
                dim_primal_features = 4
            else:
                dim_primal_features = 1

            if (self.__single_dual_nodes):
                num_dual_nodes = len(self.__primal_edges)
                dim_dual_features = 7
            else:
                num_dual_nodes = 2 * len(self.__primal_edges)
                dim_dual_features = 4

            self.__dual_features = np.empty([num_dual_nodes, dim_dual_features])
            self.__primal_features = np.empty(
                [self.__input_mesh.num_faces, dim_primal_features])

            self.__num_dual_features_assigned = 0
            # Compute the sum of face areas needed if primal features should be
            # computed independently of the dual features.
            sum_face_areas = None
            if (not self.__primal_features_from_dual_features):
                sum_face_areas = np.sum(self.__face_areas)

            for idx in range(len(self.__primal_edges)):
                primal_edge = self.__primal_edges[idx]
                # For each edge in the primal graph, add a corresponding node in
                # in the dual graph and compute the associated dual features.
                self.__add_node_to_dual_graph(primal_edge)
                # Create edges in the dual graph.
                self.__create_dual_edges_unordered(primal_edge=primal_edge)
            # For two neighboring faces with indices i < j, PyMesh only creates
            # the edge i->j. Therefore, create the opposite directed primal edge
            # edge j->i, i.e., make every primal edge undirected.
            self.__primal_edges = np.vstack(
                [self.__primal_edges, self.__primal_edges[:, ::-1]])

            # Compute primal features (cf. argument
            # `primal_features_from_dual_features` in docs of the class).
            self.__compute_primal_features(sum_face_areas)

            # Only keep the existing dual edges and dual features.
            self.__dual_edges = self.__dual_edges[:, :self.__num_dual_edges]
            (self.__dual_features
            ) = self.__dual_features[:self.__num_dual_features_assigned]

            self.__dual_graph = torch_geometric.data.Data(
                x=torch.Tensor(self.__dual_features),
                edge_index=torch.tensor(self.__dual_edges))
            self.__primal_graph = torch_geometric.data.Data(
                x=torch.Tensor(self.__primal_features),
                edge_index=torch.tensor(np.transpose(self.__primal_edges)))

            del self.__dual_features
            del self.__primal_features

        assert (self.__dual_graph is not None and
                self.__primal_graph is not None)

        primal_graph = self.__primal_graph
        dual_graph = self.__dual_graph

        return primal_graph, dual_graph


def create_dual_primal_batch(primal_graphs_list,
                             dual_graphs_list,
                             primal_edge_to_dual_node_idx_list,
                             primal_mean=None,
                             primal_std=None,
                             dual_mean=None,
                             dual_std=None):
    r"""Given a set of associated primal-dual graphs, transforms them into a
    batch that can be processed in a parallelized way by a network.

    Args:
        primal_graphs_list (list of torch_geometric.data.data.Data): List of
            primal graphs to insert in the graph.
        dual_graphs_list (list of torch_geometric.data.data.Data): List of dual
            graphs to insert in the graph.
        primal_edge_to_dual_node_idx_list (list of dict): List of the
            dictionaries that associate a tuple, encoding an edge e in a primal
            graph, to the index of the node in the dual graph that corresponds
            to the edge e.
        primal_mean, primal_std, dual_mean, dual_std (numpy array, optional):
            Mean and standard deviation of the primal-graph- and dual-graph-
            node features, respectively. If not None, the node features of each
            graph in the batch are 'standardized' (i.e., the mean over all nodes
            in the dataset becomes 0., and the standard deviation becomes 1.).
            (default: :obj:`None`)      

    Returns:
        primal_graph_batch (torch_geometric.data.batch.Batch): Batch version of
            the input primal graphs.
        dual_graph_batch (torch_geometric.data.batch.Batch): Batch version of
            the input dual graphs.
        primal_edge_to_dual_node_idx_batch (dict): Dictionary representing the
            associations between primal-graph edges and dual-graph nodes as a
            single dictionary, in which the indices of the nodes reflect the new
            indices encoded in the adjacency information from
            :obj:`primal_graph_batch.edge_index` and
            :obj:`dual_graph_batch.edge_index`.
    """
    assert (isinstance(primal_graphs_list, list) and
            isinstance(dual_graphs_list, list) and
            isinstance(primal_edge_to_dual_node_idx_list, list))
    assert (len(primal_graphs_list) == len(dual_graphs_list) ==
            len(primal_edge_to_dual_node_idx_list))
    batch_size = len(primal_graphs_list)
    # Create batched versions of the primal and the dual graphs.
    primal_graph_batch = torch_geometric.data.Batch.from_data_list(
        primal_graphs_list)
    dual_graph_batch = torch_geometric.data.Batch.from_data_list(
        dual_graphs_list)
    # Create a single 'primal-edge-to-dual-node-idx' dictionary.
    primal_edge_to_dual_node_idx_batch = primal_edge_to_dual_node_idx_list[
        0].copy()
    num_primal_nodes_before_current_batch_sample = primal_graphs_list[
        0].num_nodes
    num_dual_nodes_before_current_batch_sample = dual_graphs_list[0].num_nodes
    # Store the face areas (needed to be stored unnormalized only if one wants
    # to compute area-weighted accuracy in mesh segmentation tasks).
    if (primal_graph_batch.x.shape[1] == 1):
        primal_graph_batch.face_areas = primal_graph_batch.x.view(-1)
        warning_msg = (
            'Creating the attribute `face_areas` of the primal-graph batch '
            'assuming that the primal features represent the areas of the '
            'faces in the mesh.')
    else:
        primal_graph_batch.face_areas = None
        warning_msg = (
            "Will set the attribute `face_areas` of the primal-graph batch "
            "to None: it was assumed that the primal features represent "
            "the areas of the faces in the mesh, but the primal features "
            "have a number of channels larger than 1.")
    warnings.warn(warning_msg, Warning)

    # Optionally standardize the node features.
    if (not np.all(
        [arg is None for arg in [primal_mean, primal_std, dual_mean, dual_std]
        ])):
        assert (np.all([
            arg is not None
            for arg in [primal_mean, primal_std, dual_mean, dual_std]
        ])), (
            "To perform standardization of the node features, all the "
            "following four values must be provided: mean of the primal-graph "
            "node features, standard deviation of the primal-graph node "
            "features, mean of the dual-graph node features, standard "
            "deviation of the dual-graph node features.")
        # Check that the dimensionality of the mean/std vectors is compatible
        # with that of the node features in the graphs.
        assert (primal_mean.shape == (primal_graph_batch.x.shape[1],))
        assert (primal_std.shape == (primal_graph_batch.x.shape[1],))
        assert (dual_mean.shape == (dual_graph_batch.x.shape[1],))
        assert (dual_std.shape == (dual_graph_batch.x.shape[1],))
        # Perform the standardization.
        primal_graph_batch.x = (primal_graph_batch.x - primal_mean) / primal_std
        dual_graph_batch.x = (dual_graph_batch.x - dual_mean) / dual_std

    for batch_sample_idx in range(1, batch_size):
        # Update the node indices in the 'primal-edge-dual-idx' dictionary of
        # the current batch to match those in the batched version of the graphs.
        for primal_edge, dual_idx in primal_edge_to_dual_node_idx_list[
                batch_sample_idx].items():
            if (primal_edge is None):
                new_primal_edge = None
            else:
                new_primal_edge = (primal_edge[0] +
                                   num_primal_nodes_before_current_batch_sample,
                                   primal_edge[1] +
                                   num_primal_nodes_before_current_batch_sample)
            new_dual_idx = dual_idx + num_dual_nodes_before_current_batch_sample
            # Save in the 'batch' dictionary.
            if (new_primal_edge is not None):
                assert (
                    new_primal_edge not in primal_edge_to_dual_node_idx_batch)
            primal_edge_to_dual_node_idx_batch[new_primal_edge] = new_dual_idx

        num_primal_nodes_before_current_batch_sample += primal_graphs_list[
            batch_sample_idx].num_nodes
        num_dual_nodes_before_current_batch_sample += dual_graphs_list[
            batch_sample_idx].num_nodes

    return (primal_graph_batch, dual_graph_batch,
            primal_edge_to_dual_node_idx_batch)
