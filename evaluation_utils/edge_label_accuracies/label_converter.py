import numpy as np
import os
import pymesh
import sys

from models.layers.mesh import Mesh

seg_colors = np.array([[255., 0., 0.], [0., 255., 0.], [0., 0., 255.],
                       [128., 128., 0.], [0., 255., 255.], [255., 0., 255.],
                       [255., 255., 0.], [0., 255., 128.]])


class EdgeLabelToFaceLabelConverter:
    """Converts the edge labels associated to the given input mesh to labels
    associated to the mesh faces.
    """

    def __init__(self, mesh, edge_labels, num_classes, print_warnings=False):
        self.__print_warnings = print_warnings
        # Allow to call recursive function the needed number of times.
        sys.setrecursionlimit(5000)
        self.__num_classes = num_classes
        # Read the mesh data and per-edge class labels.
        # Read vertices.
        self.__vertices = mesh.vs
        self.__num_vertices = self.__vertices.shape[0]
        # Edges.
        if (hasattr(mesh, 'original_edges')):
            self.__edges = np.array(mesh.original_edges).squeeze()
        else:
            self.__edges = mesh.edges
        self.__gemm_edges = mesh.gemm_edges
        self.__num_edges = self.__edges.shape[0]
        self.__edge_labels = edge_labels[:self.__num_edges]
        self.__found_boundary_edge = False
        # - Make edges directed.
        self.__edges = np.vstack([self.__edges, self.__edges[:, ::-1]])
        self.__gemm_edges = np.vstack([self.__gemm_edges, self.__gemm_edges])
        self.__edge_labels = np.hstack([self.__edge_labels, self.__edge_labels])

        self.__halfedge_to_halfedge_idx = {}
        self.__halfedge_to_face_idx = {}
        self.__face_to_face_idx = {}
        for halfedge_idx, halfedge in enumerate(self.__edges):
            self.__halfedge_to_halfedge_idx[(halfedge[0],
                                             halfedge[1])] = halfedge_idx

        # Create faces.
        self.__was_halfedge_used = np.zeros([self.__num_edges * 2],
                                            dtype=np.bool)
        self.__faces = np.empty([2 * self.__num_edges, 3])
        # - Stores the class label of each face, obtained by taking the most
        #   frequent class label of those assigned to the face edges.
        self.__class_label_per_face = [None] * len(self.__faces)
        halfedges_to_find = []
        for halfedge in self.__edges:
            self.__halfedge_to_face_idx[tuple(halfedge.tolist())] = None
            halfedges_to_find.append(halfedge.tolist())
        # - Start with any half-edge.
        self.__num_faces = 0
        self.__is_face_ambiguously_labelled = [False] * len(self.__faces)
        num_connected_components = 0
        while (len(halfedges_to_find) > 0):
            old_halfedges_to_find = halfedges_to_find.copy()
            num_connected_components += 1
            first_halfedge = halfedges_to_find[0]
            self.__form_faces_with_halfedge(first_halfedge[0],
                                            first_halfedge[1])
            # Check that all edges were added to a face.
            all_faces_found = True
            halfedges_to_find = []
            for halfedge in self.__edges:
                if (self.__halfedge_to_face_idx[tuple(halfedge.tolist())] is
                        None):
                    all_faces_found = False
                    halfedges_to_find.append(halfedge.tolist())
            if (halfedges_to_find == old_halfedges_to_find):
                # The fact that no new half-edges get added to the mesh
                # may be caused by half-edges that cannot be assigned to any
                # face because the associated edge is boundary.
                assert (self.__found_boundary_edge)
                break
            if (not all_faces_found and print_warnings):
                print("Not all faces were found.")

        if (num_connected_components == 1):
            if (not self.__found_boundary_edge):
                # The mesh is watertight, Euler's formula must hold.
                genus = (self.__num_edges - self.__num_vertices -
                         self.__num_faces) // 2 + 1
                if (genus != 0 and print_warnings):
                    print(f"\033[93mDoes mesh {mesh.filename} have genus "
                          f"{genus}?\033[0m")
        else:
            if (print_warnings):
                print(f"Mesh {mesh.filename} has {num_connected_components} "
                      "connected components.")
        self.__faces = self.__faces[:self.__num_faces]
        self.__class_label_per_face = np.array(
            self.__class_label_per_face[:self.__num_faces])
        self.__is_face_ambiguously_labelled = np.array(
            self.__is_face_ambiguously_labelled[:self.__num_faces])
        # Transform the hard labels on the faces into soft labels on the edges.
        self.__edges_soft_labels_from_faces = np.zeros(
            [self.__num_edges, self.__num_classes], dtype=np.float)

        for edge_idx in range(self.__num_edges):
            # - Find the faces associated to each edge (i.e., to each half-edge
            #   and its opposite half-edge).
            halfedges = [
                tuple(self.__edges[edge_idx].tolist()),
                tuple(self.__edges[edge_idx + self.__num_edges].tolist())
            ]
            assert (halfedges[0] == halfedges[1][::-1])
            num_faces_found_for_edge = 0
            for halfedge in halfedges:
                face_idx = self.__halfedge_to_face_idx[halfedge]
                if (face_idx is not None):

                    face_label = int(self.__class_label_per_face[face_idx])

                    # - Assign the face label to the edge.
                    self.__edges_soft_labels_from_faces[edge_idx,
                                                        face_label] += 1
                    num_faces_found_for_edge += 1
            assert (num_faces_found_for_edge > 0)
        # - Make the labels assigned from the faces soft.
        (self.__edges_soft_labels_from_faces
        ) /= self.__edges_soft_labels_from_faces.sum(axis=1).reshape(-1, 1)

        # Find the area of each face in the mesh.
        self.__mesh = pymesh.form_mesh(self.__vertices, self.__faces)
        self.__mesh.add_attribute("face_area")
        self.__face_areas = self.__mesh.get_attribute("face_area")

        self.__mesh.add_attribute('face_blue')
        self.__mesh.add_attribute('face_green')
        self.__mesh.add_attribute('face_red')

    @property
    def face_labels(self):
        return self.__class_label_per_face

    @property
    def face_areas(self):
        return self.__face_areas

    @property
    def edges_soft_labels_from_faces(self):
        return self.__edges_soft_labels_from_faces

    @property
    def is_face_ambiguously_labelled(self):
        return self.__is_face_ambiguously_labelled

    def __form_faces_with_halfedge(self, vertex_a, vertex_b, verbose=False):
        r"""Recursive function. Constructs the face with the input half-edge,
        adds it to the mesh, and reiterates on the neighboring faces.

        Args:
            vertex_a, vertex_b (int): Indices of the two mesh vertices a and b
                forming the half-edge a->b, that belongs to the face that should
                be added to the mesh.
            verbose (bool, optional): If True, shows verbose prints.
        
        Returns:
            None.
        """
        # Form faces a->b->c->a.
        # There are up to two faces per each edge (one for each half-edge,
        # unless the edge is boundary).
        for halfedge_idx in range(2):
            idx_halfedge_ab = self.__halfedge_to_halfedge_idx[(vertex_a,
                                                               vertex_b)]
            edge_ab = set(self.__edges[idx_halfedge_ab])
            idx_halfedge_bc = self.__gemm_edges[idx_halfedge_ab,
                                                2 * halfedge_idx]
            if (idx_halfedge_bc == -1):
                # Boundary edge. -> No face.
                self.__found_boundary_edge = True
                if (verbose):
                    print("\tBoundary edge.")
                continue
            edge_bc = set(self.__edges[idx_halfedge_bc])
            vertex_b = edge_ab & edge_bc
            assert (len(vertex_b) == 1)
            vertex_a = list(edge_ab - vertex_b)[0]
            vertex_c = list(edge_bc - vertex_b)[0]
            vertex_b = list(vertex_b)[0]
            if (verbose):
                print(f"* Processing edge {vertex_a}->{vertex_b}")
            idx_halfedge_ab = self.__halfedge_to_halfedge_idx[(vertex_a,
                                                               vertex_b)]
            idx_halfedge_bc = self.__halfedge_to_halfedge_idx[(vertex_b,
                                                               vertex_c)]
            if (self.__was_halfedge_used[idx_halfedge_ab]):
                if (verbose):
                    face_containing_ab = self.__halfedge_to_face_idx[(vertex_a,
                                                                      vertex_b)]
                    print(
                        f"Edge a->b ({vertex_a}->{vertex_b}) was already used "
                        f"in face {face_containing_ab} ("
                        f"{self.__faces[face_containing_ab]}).")
                continue

            assert (not self.__was_halfedge_used[idx_halfedge_bc])
            idx_halfedge_ca = self.__gemm_edges[idx_halfedge_ab,
                                                2 * halfedge_idx + 1]
            edge_ca = set(self.__edges[idx_halfedge_ca])
            assert (vertex_a in edge_ca and vertex_c in edge_ca)
            idx_halfedge_ca = self.__halfedge_to_halfedge_idx[(vertex_c,
                                                               vertex_a)]

            were_same_halfedges_used_already = (
                self.__was_halfedge_used[idx_halfedge_ab] or
                self.__was_halfedge_used[idx_halfedge_bc] or
                self.__was_halfedge_used[idx_halfedge_ca])
            # Check that the opposite face was not inserted.
            was_opposite_face_inserted = (
                vertex_a, vertex_c, vertex_b) in self.__face_to_face_idx or (
                    vertex_c, vertex_b,
                    vertex_a) in self.__face_to_face_idx or (
                        vertex_b, vertex_a, vertex_c) in self.__face_to_face_idx
            if (were_same_halfedges_used_already or was_opposite_face_inserted):
                continue
            # Add face, if it is possible to determine its class label.
            face_idx = self.__num_faces
            assert (self.__class_label_per_face[face_idx] is None)
            class_labels_in_face = [
                self.__edge_labels[idx_halfedge_ab],
                self.__edge_labels[idx_halfedge_bc],
                self.__edge_labels[idx_halfedge_ca]
            ]
            # - Find the most frequent label in the three associated to the
            #   edges.
            if (len(set(class_labels_in_face)) == 3):
                # - Select arbitrarily the class label of the first edge.
                class_label = self.__edge_labels[idx_halfedge_ab]
                self.__class_label_per_face[face_idx] = class_label
                self.__edge_labels[idx_halfedge_ab] = class_label
                self.__edge_labels[idx_halfedge_bc] = class_label
                self.__edge_labels[idx_halfedge_ca] = class_label
                self.__is_face_ambiguously_labelled[face_idx] = True
                if (self.__print_warnings):
                    print("\033[91mNote: arbitrarily assigned class label "
                          f"of edge {idx_halfedge_ab} to face {face_idx}, as "
                          "the latter has 3 edges (indices "
                          f"{idx_halfedge_ab}, {idx_halfedge_bc} and "
                          f"{idx_halfedge_ca}) with different class labels."
                          "\033[0m")
            else:
                # - Select the class label that is shared by most edges in the
                #   face.
                for class_label in set(class_labels_in_face):
                    if (class_labels_in_face.count(class_label) > 1):
                        self.__class_label_per_face[face_idx] = class_label
                        self.__edge_labels[idx_halfedge_ab] = class_label
                        self.__edge_labels[idx_halfedge_bc] = class_label
                        self.__edge_labels[idx_halfedge_ca] = class_label
                        break

            self.__faces[face_idx] = [vertex_a, vertex_b, vertex_c]
            self.__was_halfedge_used[idx_halfedge_ab] = True
            self.__was_halfedge_used[idx_halfedge_bc] = True
            self.__was_halfedge_used[idx_halfedge_ca] = True
            self.__halfedge_to_face_idx[(vertex_a, vertex_b)] = face_idx
            self.__halfedge_to_face_idx[(vertex_b, vertex_c)] = face_idx
            self.__halfedge_to_face_idx[(vertex_c, vertex_a)] = face_idx
            self.__face_to_face_idx[(vertex_a, vertex_b, vertex_c)] = face_idx

            self.__num_faces += 1
            # Add neighboring faces.
            self.__form_faces_with_halfedge(vertex_c, vertex_b, verbose=verbose)
            self.__form_faces_with_halfedge(vertex_a, vertex_c, verbose=verbose)


class OptClass:

    def __init__(self):
        self.num_aug = 1


class FaceLabelToEdgeSoftLabelConverter:
    """Converts the face labels associated to the given input mesh to labels
    associated to the mesh faces.
    """

    def __init__(self,
                 mesh_original_filename,
                 mesh_filename,
                 num_classes,
                 edge_soft_labels_file,
                 print_warnings=False):
        self.__print_warnings = print_warnings
        # Allow to call recursive function the needed number of times.
        sys.setrecursionlimit(5000)
        self.__num_classes = num_classes
        # Read the mesh data.
        # Read mesh.
        opt = OptClass()
        mesh_pymesh = pymesh.load_mesh(mesh_filename)
        for color in ['green', 'red', 'blue']:
            if (not mesh_pymesh.has_attribute(f'face_{color}')):
                mesh_pymesh.add_attribute(f'face_{color}')
                mesh_pymesh.set_attribute(
                    f'face_{color}',
                    mesh_pymesh.get_attribute(f'vertex_{color}'))
                mesh_pymesh.remove_attribute(f'vertex_{color}')
        face_labels = np.empty(mesh_pymesh.num_faces, dtype=np.long)
        face_red = mesh_pymesh.get_attribute('face_red')
        face_green = mesh_pymesh.get_attribute('face_green')
        face_blue = mesh_pymesh.get_attribute('face_blue')
        for face_idx, (blue, green,
                       red) in enumerate(zip(face_blue, face_green, face_red)):

            assigned_label = False
            for label_idx in range(len(seg_colors)):
                if (np.all([blue, green, red] == seg_colors[label_idx])):
                    face_labels[face_idx] = label_idx
                    assigned_label = True

            assert (assigned_label), [blue, green, red]
        mesh = Mesh(file=mesh_original_filename, opt=opt, hold_history=False)
        # Map vertices in MeshCNN to pymesh vertices.
        meshcnn_vertex_idx_to_pymesh_vertex_idx = np.empty(
            [mesh_pymesh.num_vertices])
        pymesh_vertex_to_assign = [r for r in range(mesh_pymesh.num_vertices)]
        for meshcnn_vertex_idx, meshcnn_vertex in enumerate(mesh.vs):
            found_match = None
            for pymesh_vertex_idx in pymesh_vertex_to_assign:
                pymesh_vertex = mesh_pymesh.vertices[pymesh_vertex_idx]
                if (np.isclose(meshcnn_vertex[0], pymesh_vertex[0]) and
                        np.isclose(meshcnn_vertex[1], pymesh_vertex[1]) and
                        np.isclose(meshcnn_vertex[2], pymesh_vertex[2])):
                    found_match = pymesh_vertex_idx
                    pymesh_vertex_to_assign.remove(pymesh_vertex_idx)
                    meshcnn_vertex_idx_to_pymesh_vertex_idx[
                        meshcnn_vertex_idx] = pymesh_vertex_idx
                    break
            assert (found_match is not None)

        face_vertex_indices = mesh_pymesh.get_attribute(
            'face_vertex_indices').astype(np.long).reshape(-1, 3)
        vertex_to_faces = {
            vertex: [] for vertex in range(mesh_pymesh.num_vertices)
        }
        for face_idx, vertices_in_face in enumerate(face_vertex_indices):
            for pymesh_vertex_idx in vertices_in_face:
                meshcnn_vertex_idx = meshcnn_vertex_idx_to_pymesh_vertex_idx[
                    pymesh_vertex_idx]
                vertex_to_faces[meshcnn_vertex_idx].append(face_idx)

        # Read vertices.
        self.__vertices = mesh.vs
        self.__num_vertices = self.__vertices.shape[0]
        assert (self.__num_vertices == mesh_pymesh.num_vertices
               ), f"{self.__num_vertices}, {mesh_pymesh.num_vertices}"
        # Edges.
        if (hasattr(mesh, 'original_edges')):
            self.__edges = np.array(mesh.original_edges).squeeze()
        else:
            self.__edges = mesh.edges
        self.__num_edges = self.__edges.shape[0]

        self.__halfedge_to_halfedge_idx = {}
        for halfedge_idx, halfedge in enumerate(self.__edges):
            self.__halfedge_to_halfedge_idx[(halfedge[0],
                                             halfedge[1])] = halfedge_idx

        assert (num_classes in [3, 4, 8])
        edge_soft_labels = np.zeros([self.__num_edges, num_classes])

        # For each (half)-edge, see the face to which the edge belongs and form
        # edge soft-labels from the hard labels of these faces.
        for halfedge_idx, halfedge in enumerate(self.__edges):
            vertex_1, vertex_2 = halfedge
            faces_halfedge = list(
                set(vertex_to_faces[vertex_1]) & set(vertex_to_faces[vertex_2]))
            assert (len(faces_halfedge) in [1, 2])
            for face_idx in faces_halfedge:
                # Get face label.
                face_lab = face_labels[face_idx]
                # Update edge soft label.
                edge_soft_labels[halfedge_idx, face_lab] += 0.5

        # Create the output folder if non-existent.
        output_folder = os.path.dirname(edge_soft_labels_file)
        if (not os.path.exists(output_folder)):
            os.makedirs(output_folder)

        # Save soft label to file.
        np.savetxt(edge_soft_labels_file, edge_soft_labels, fmt="%.6f")
