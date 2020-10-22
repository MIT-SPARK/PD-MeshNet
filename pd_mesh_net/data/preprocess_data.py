"""
Functions in this module are modified versions of those from MeshCNN
(https://github.com/ranahanocka/MeshCNN/).
"""

import numpy as np
import pymesh

from pd_mesh_net.utils import dihedral_angle_and_local_indices_edges


# Data augmentation methods.
def augmentation(mesh,
                 vertices_scale_mean=None,
                 vertices_scale_var=None,
                 edges_flip_fraction=None):
    r"""Performs data augmentation on the mesh, according to the input options.
    Augmentation can consist in scaling the vertices of the mesh and/or flipping
    its edges.
    

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        vertices_scale_mean, vertices_scale_var (float, optional): If both are
            not None, the vertices from the input mesh are scaled by multiplying
            each of them by a scaling factor drawn from a normal distribution
            with mean `vertices_scale_mean` and variance `vertices_scale_var`.
            (default: :obj:`None`)
        edges_flip_fraction (float, optional): If not None, a fraction equal to
            `edges_flip_fraction` of edges from the input mesh are flipped if
            the dihedral angle between the two faces associated to it is within
            a certain range of values (cf. function :obj:`flip_edges`).
            (default: :obj:`None`)

    Returns:
        mesh (pymesh.Mesh.Mesh): Mesh after augmentation.
    """
    # If required, scale the vertices of the mesh.
    if (vertices_scale_mean is not None and vertices_scale_var is not None):
        mesh = scale_verts(mesh, vertices_scale_mean, vertices_scale_var)
    # If required, flip the edges of the mesh.
    if (edges_flip_fraction is not None):
        assert (0. <= edges_flip_fraction <= 1.)
        mesh = flip_edges(mesh, edges_flip_fraction)

    return mesh


def post_augmentation(mesh, slide_vertices_fraction=None):
    r"""Performs post-augmentation on the mesh, i.e., slide its vertices if
    required.

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        slide_vertices_fraction (float, optional): If not None, a fraction equal
            to `slide_vertices_fraction` of the vertices from the input mesh are
            slid.

    Returns:
        mesh_after_postaugmentation (pymesh.Mesh.Mesh): Mesh with the vertex
            positions updated after sliding (if performed, otherwise the input
            mesh).
    """
    # If required, slide the vertices of the mesh.
    if (not slide_vertices_fraction is None):
        assert (0. <= slide_vertices_fraction <= 1.)
        mesh_after_postaugmentation = slide_vertices(mesh,
                                                     slide_vertices_fraction)
    else:
        mesh_after_postaugmentation = mesh

    return mesh_after_postaugmentation


def slide_vertices(mesh, slide_vertices_fraction, min_dihedral_angle=2.65):
    r"""Slides the vertices of the mesh.

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        slide_vertices_fraction (float): Fraction of vertices to slide.
        min_dihedral_angle (float, optional): A vertex can be slid only if the
            dihedral angles between each pair of faces that share the vertex are
            between `min_dihedral_angle` and
            :math:`2\pi` - `min_dihedral_angle`. This change w.r.t. to MeshCNN
            (where it was only required that the angle be larger than
            `min_dihedral_angle`) is due to the different definition of dihedral
            angle (theirs is always smaller or equal to :math:`\pi` by
            construction, ours is between `0` and :math:`2\pi`). This value is
            to be interpreted in radians.
            (default: :obj:`2.65`)

    Returns:
        mesh_after_sliding (pymesh.Mesh.Mesh): Mesh with the vertex positions
            updated after sliding.
    """
    # Computes the dihedral angles between the faces.
    assert (0. <= min_dihedral_angle <= 2 * np.pi)
    all_edges_between_faces = pymesh.mesh_to_dual_graph(mesh)[1]
    num_edges_between_faces = all_edges_between_faces.shape[0]

    # Compute face normals.
    mesh.add_attribute("face_normal")
    face_normals = mesh.get_face_attribute("face_normal")

    dihedral_angles = np.zeros(num_edges_between_faces)
    local_indices_edges = np.zeros([num_edges_between_faces, 2], dtype=np.int)

    vertex_to_edge_idx = dict()
    vertices_after_sliding = mesh.vertices.copy()
    for edge_idx, (face_idx_1,
                   face_idx_2) in enumerate(all_edges_between_faces):
        dihedral_angles[edge_idx], local_indices_edges[
            edge_idx] = dihedral_angle_and_local_indices_edges(
                mesh=mesh,
                face_indices=[face_idx_1, face_idx_2],
                face_normals=face_normals[(face_idx_1, face_idx_2), :])
        # Save vertex-to-edge-index correspondence.
        assert (mesh.faces[face_idx_1][local_indices_edges[edge_idx][0]] ==
                mesh.faces[face_idx_2][local_indices_edges[edge_idx][1]])
        assert (mesh.faces[face_idx_1][(local_indices_edges[edge_idx][0] + 1) %
                                       3] == mesh.faces[face_idx_2]
                [local_indices_edges[edge_idx][1] - 1])
        common_vertices = tuple(
            sorted([
                mesh.faces[face_idx_1][local_indices_edges[edge_idx][0]],
                mesh.faces[face_idx_1][(local_indices_edges[edge_idx][0] + 1) %
                                       3]
            ]))
        vertex_to_edge_idx[common_vertices] = edge_idx

    vertices_to_slide = np.random.permutation(mesh.num_vertices)
    target_num_vertices_to_slide = int(slide_vertices_fraction *
                                       mesh.num_vertices)
    num_slid_vertices = 0
    # NOTE: This method, taken from MeshCNN, is not fully accurate, as dihedral
    # angles might change while sliding the vertices. Also, since it requires
    # dihedral angles, it cannot be applied on vertices that are on boundary
    # edges.
    for vertex_idx in vertices_to_slide:
        if (num_slid_vertices < target_num_vertices_to_slide):
            # - Obtain all the edges connected to the vertex to be slid.
            neighboring_vertices = mesh.get_vertex_adjacent_vertices(vertex_idx)
            connected_edges_idx_and_neighboring_vertex = np.empty(
                [len(neighboring_vertices), 2], dtype=np.int)
            num_valid_neighbors = 0
            for n in neighboring_vertices:
                vertex_to_vertex_edge = tuple(sorted([vertex_idx, n]))
                if (vertex_to_vertex_edge in vertex_to_edge_idx):
                    # NOTE: Boundary edges are ignored, as it is not possible to
                    # compute a dihedral angle associated to them.
                    connected_edges_idx_and_neighboring_vertex[
                        num_valid_neighbors] = [
                            vertex_to_edge_idx[vertex_to_vertex_edge], n
                        ]
                    num_valid_neighbors += 1
            (connected_edges_idx_and_neighboring_vertex
            ) = connected_edges_idx_and_neighboring_vertex[:num_valid_neighbors]
            if (min_dihedral_angle < min(
                    dihedral_angles[connected_edges_idx_and_neighboring_vertex]
                [:, 0]) < 2 * np.pi - min_dihedral_angle):
                # - Randomly select one of the edges connected to the vertex.
                (random_connected_edge_idx, other_vertex_idx
                ) = connected_edges_idx_and_neighboring_vertex[np.random.choice(
                    range(len(connected_edges_idx_and_neighboring_vertex)))]
                # - Take the other vertex in the edge.
                # - Slide the vertex by sampling a point between the other
                #   vertex and the vertex itself.
                slid_vertex = vertices_after_sliding[
                    vertex_idx] + np.random.uniform(
                        0.2, 0.5) * (vertices_after_sliding[other_vertex_idx] -
                                     vertices_after_sliding[vertex_idx])
                vertices_after_sliding[vertex_idx] = slid_vertex
                num_slid_vertices += 1
        else:
            break

    mesh_after_sliding = pymesh.form_mesh(vertices_after_sliding, mesh.faces)

    return mesh_after_sliding


def scale_verts(mesh, mean=1.0, var=0.1):
    r"""Scales the vertices of the input mesh, by multiplying each by a scaling
        factor drawn from a normal distribution.

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        mean (float, optional): Mean of the normal distribution from which
            the scaling factor is drawn. (default: :obj:`0.1`)
        var (float, optional): Variance of the normal distribution from which
            the scaling factor is drawn. (default: :obj:`0.1`)

    Returns:
        mesh (pymesh.Mesh.Mesh): Mesh after data augmentation.
    """
    vertices = mesh.vertices.copy()
    for vertex in vertices:
        vertex *= np.random.normal(mean, var)

    mesh = pymesh.form_mesh(vertices, mesh.faces)

    return mesh


def flip_edges(mesh, edges_flip_fraction, min_dihedral_angle=2.7):
    r"""Flips the edges in the input mesh.

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        edges_flip_fraction (float): Fraction of edges to flip.
        min_dihedral_angle (float, optional): An edge can be flipped only if all
            the following conditions are fulfilled:
            - The operation does not cause the face normal to flip around;
            - The new edge formed by edge flipping is manifold (NOTE: this is
              not the case if the two faces share a valence-3 vertex, or in
              general, when the new edge is already in the mesh, consider as an
              example the mesh with faces [0, 1, 4], [4, 1, 2], [2, 3, 4],
              [3, 0, 4] and [2, 0, 3]);
            - The dihedral angle between its two associated faces is between
              `min_dihedral_angle` and :math:`2\pi` - `min_dihedral_angle`. This
              change w.r.t. to MeshCNN (where it was only required that the
              angle be larger than `min_dihedral_angle`) is due to the different
              definition of dihedral angle (theirs is always smaller or equal to
              :math:`\pi` by construction, ours is between `0` and
              :math:`2\pi`).
            This value is to be interpreted in radians. (default: :obj:`2.7`)

    Returns:
        mesh_after_edge_flipping (pymesh.Mesh.Mesh): Mesh after edge-flipping.
    """
    assert (0. <= min_dihedral_angle <= 2 * np.pi)
    # The following contains all the edges between two faces (i.e., no boundary
    # edges), storing, for each edge, the indices of the two faces that share
    # that edge.
    all_edges_between_faces = pymesh.mesh_to_dual_graph(mesh)[1]
    # Convert to dictionary, so that it can be dynamically updated.
    faces_sharing_edge_to_edge_idx = {
        tuple(edge): edge_idx
        for edge_idx, edge in enumerate(all_edges_between_faces)
    }
    num_edges_between_faces = all_edges_between_faces.shape[0]
    # Store all edges in the mesh, in the format of the sorted indices of its
    # two endpoints.
    all_vertex_to_vertex_edges = set()
    for vertex_idx in range(len(mesh.vertices)):
        neighboring_vertices = mesh.get_vertex_adjacent_vertices(vertex_idx)
        for n_idx in neighboring_vertices:
            if (vertex_idx < n_idx):
                all_vertex_to_vertex_edges.add((vertex_idx, n_idx))

    # Compute face normals.
    mesh.add_attribute("face_normal")
    face_normals = mesh.get_face_attribute("face_normal").copy()

    edges_to_flip = np.random.permutation(num_edges_between_faces)
    target_num_edges_to_flip = int(edges_flip_fraction *
                                   num_edges_between_faces)
    num_flipped_edges = 0

    faces_after_augmentation = mesh.faces.copy()
    num_faces = len(faces_after_augmentation)
    # Stores, for each face, the faces adjacent to it.
    face_adjacent_faces = -np.ones([num_faces, 3], dtype=np.int)
    for face_idx in range(num_faces):
        adjacent_faces = mesh.get_face_adjacent_faces(face_idx)
        face_adjacent_faces[
            face_idx, :len(adjacent_faces)] = mesh.get_face_adjacent_faces(
                face_idx)
    for edge_idx in edges_to_flip:
        if (num_flipped_edges == target_num_edges_to_flip):
            break
        face_idx_1, face_idx_2 = all_edges_between_faces[edge_idx].copy()
        face_1 = faces_after_augmentation[face_idx_1].copy()
        face_2 = faces_after_augmentation[face_idx_2].copy()
        # Compute the 'local indices of the common edge'.
        dihedral_angle, local_indices = dihedral_angle_and_local_indices_edges(
            mesh=mesh,
            faces=[face_1.tolist(), face_2.tolist()],
            face_normals=face_normals[(face_idx_1, face_idx_2), :])
        assert (
            dihedral_angle is not None and local_indices is not None
        ), "Data augmentation cannot be performed on non-manifold meshes."
        (local_indices_edges_face_1, local_indices_edges_face_2) = local_indices

        if (min_dihedral_angle < dihedral_angle <
                2 * np.pi - min_dihedral_angle):
            # Form the new edge by taking the two vertices that are not shared
            # by the faces.
            assert (face_1[local_indices_edges_face_1] ==
                    face_2[local_indices_edges_face_2])
            common_vertex_1 = face_1[local_indices_edges_face_1]
            assert (face_1[(local_indices_edges_face_1 + 1) %
                           3] == face_2[local_indices_edges_face_2 - 1])
            common_vertex_2 = face_1[(local_indices_edges_face_1 + 1) % 3]
            new_edge = (face_1[local_indices_edges_face_1 - 1],
                        face_2[(local_indices_edges_face_2 + 1) % 3])
            # If the new edge is already in the mesh, a non-manifold edge would
            # be created.
            if (tuple(sorted(new_edge)) in all_vertex_to_vertex_edges):
                continue
            new_face_1 = [common_vertex_1, new_edge[1], new_edge[0]]
            new_face_2 = [common_vertex_2, new_edge[0], new_edge[1]]
            new_faces = np.array([new_face_1, new_face_2], dtype=np.int)
            # Check that the no degenerate faces are formed.
            if (not are_areas_nonnull(mesh, new_faces)):
                continue
            # Check that normals do not flip around when performing the edge
            # flip.
            new_normals_face_1 = np.cross(
                mesh.vertices[new_face_1[1]] - mesh.vertices[new_face_1[0]],
                mesh.vertices[new_face_1[2]] - mesh.vertices[new_face_1[1]])
            new_normals_face_1 /= np.linalg.norm(new_normals_face_1)
            new_normals_face_2 = np.cross(
                mesh.vertices[new_face_2[1]] - mesh.vertices[new_face_2[0]],
                mesh.vertices[new_face_2[2]] - mesh.vertices[new_face_2[1]])
            new_normals_face_2 /= np.linalg.norm(new_normals_face_2)
            normals_face_1_flip = np.arccos(
                face_normals[face_idx_1].dot(new_normals_face_1)) > (np.pi / 2)
            normals_face_2_flip = np.arccos(
                face_normals[face_idx_2].dot(new_normals_face_2)) > (np.pi / 2)
            # Only perform the flip if no face normals flip.
            if (not normals_face_1_flip and not normals_face_2_flip):
                # The faces formed by the new edge are valid.
                # - Replace the original faces with the new ones.
                faces_after_augmentation[face_idx_1] = new_faces[0]
                faces_after_augmentation[face_idx_2] = new_faces[1]
                # - Reconnect neighboring faces.\
                affected_neighbor_of_face_idx_1 = -1
                affected_neighbor_of_face_idx_2 = -1
                for neighboring_face_idx in face_adjacent_faces[face_idx_2]:
                    # - Find the neighboring face that was not previously
                    #   connected to the face but now is (it was connected with
                    #   face face_idx_2).
                    if (neighboring_face_idx == face_idx_1 or
                            neighboring_face_idx == -1):
                        continue
                    neighboring_face = faces_after_augmentation[
                        neighboring_face_idx]
                    if (common_vertex_1 in neighboring_face):
                        assert (affected_neighbor_of_face_idx_2 == -1)
                        # - This edge was shared by the face
                        #   neighboring_face_idx and the face face_idx_2. Now it
                        #   needs to become an edge between face
                        #   neighboring_face_idx and face face_idx_1.
                        faces_sharing_edge_to_change = tuple(
                            sorted([neighboring_face_idx, face_idx_2]))
                        faces_sharing_edge_changed = tuple(
                            sorted([neighboring_face_idx, face_idx_1]))
                        assert (faces_sharing_edge_to_change in
                                faces_sharing_edge_to_edge_idx)
                        edge_to_change_idx = faces_sharing_edge_to_edge_idx[
                            faces_sharing_edge_to_change]
                        (faces_sharing_edge_to_edge_idx[
                            faces_sharing_edge_changed]
                        ) = faces_sharing_edge_to_edge_idx.pop(
                            faces_sharing_edge_to_change)
                        all_edges_between_faces[
                            edge_to_change_idx] = faces_sharing_edge_changed
                        affected_neighbor_of_face_idx_2 = neighboring_face_idx

                for neighboring_face_idx in face_adjacent_faces[face_idx_1]:
                    # - Find the neighboring face that was not previously
                    #   connected to the face but now is (it was connected with
                    #   face face_idx_1).
                    if (neighboring_face_idx == face_idx_2 or
                            neighboring_face_idx == -1):
                        continue
                    neighboring_face = faces_after_augmentation[
                        neighboring_face_idx]
                    if (common_vertex_2 in neighboring_face or
                            neighboring_face_idx == -1):
                        assert (affected_neighbor_of_face_idx_1 == -1)
                        # - This edge was shared by the face
                        #   neighboring_face_idx and the face face_idx_1. Now it
                        #   needs to become an edge between face
                        #   neighboring_face_idx and face face_idx_2.
                        faces_sharing_edge_to_change = tuple(
                            sorted([neighboring_face_idx, face_idx_1]))
                        faces_sharing_edge_changed = tuple(
                            sorted([neighboring_face_idx, face_idx_2]))
                        assert (faces_sharing_edge_to_change in
                                faces_sharing_edge_to_edge_idx)
                        edge_to_change_idx = faces_sharing_edge_to_edge_idx[
                            faces_sharing_edge_to_change]
                        (faces_sharing_edge_to_edge_idx[
                            faces_sharing_edge_changed]
                        ) = faces_sharing_edge_to_edge_idx.pop(
                            faces_sharing_edge_to_change)
                        all_edges_between_faces[
                            edge_to_change_idx] = faces_sharing_edge_changed
                        affected_neighbor_of_face_idx_1 = neighboring_face_idx
                # Replace the neighboring face with the affected neighboring
                # face of face_idx_2 (which can also be -1, a boundary).
                local_idx_neighbor = np.argwhere(
                    face_adjacent_faces[face_idx_1] ==
                    affected_neighbor_of_face_idx_1)
                assert (local_idx_neighbor.shape[0] == 1)
                assert (np.argwhere(
                    face_adjacent_faces[face_idx_1] ==
                    affected_neighbor_of_face_idx_2).shape[0] == 0)
                local_idx_neighbor = local_idx_neighbor[0, 0]
                face_adjacent_faces[
                    face_idx_1,
                    local_idx_neighbor] = affected_neighbor_of_face_idx_2
                if (affected_neighbor_of_face_idx_1 != -1):
                    # Replace face face_idx_1 with face_idx_2 in the list of
                    # adjacent faces of the neighboring face.
                    idx_for_local_neighbor = np.argwhere(
                        face_adjacent_faces[affected_neighbor_of_face_idx_1] ==
                        face_idx_1)
                    assert (idx_for_local_neighbor.shape[0] == 1)
                    assert (np.argwhere(
                        face_adjacent_faces[affected_neighbor_of_face_idx_1] ==
                        face_idx_2).shape[0] == 0)
                    idx_for_local_neighbor = idx_for_local_neighbor[0, 0]
                    face_adjacent_faces[affected_neighbor_of_face_idx_1,
                                        idx_for_local_neighbor] = face_idx_2

                # Replace the neighboring face with the affected neighboring
                # face of face_idx_1 (which can also be -1, a boundary).
                local_idx_neighbor = np.argwhere(
                    face_adjacent_faces[face_idx_2] ==
                    affected_neighbor_of_face_idx_2)
                assert (local_idx_neighbor.shape[0] == 1)
                assert (np.argwhere(
                    face_adjacent_faces[face_idx_2] ==
                    affected_neighbor_of_face_idx_1).shape[0] == 0)
                local_idx_neighbor = local_idx_neighbor[0, 0]
                face_adjacent_faces[
                    face_idx_2,
                    local_idx_neighbor] = affected_neighbor_of_face_idx_1
                if (affected_neighbor_of_face_idx_2 != -1):
                    # Replace face face_idx_2 with face_idx_1 in the list of
                    # adjacent faces of the neighboring face.
                    idx_for_local_neighbor = np.argwhere(
                        face_adjacent_faces[affected_neighbor_of_face_idx_2] ==
                        face_idx_2)
                    assert (idx_for_local_neighbor.shape[0] == 1)
                    assert (np.argwhere(
                        face_adjacent_faces[affected_neighbor_of_face_idx_2] ==
                        face_idx_1).shape[0] == 0)
                    idx_for_local_neighbor = idx_for_local_neighbor[0, 0]
                    face_adjacent_faces[affected_neighbor_of_face_idx_2,
                                        idx_for_local_neighbor] = face_idx_1
                # Update the set of edges in the mesh.
                all_vertex_to_vertex_edges.remove(
                    tuple(sorted([common_vertex_1, common_vertex_2])))
                all_vertex_to_vertex_edges.add(tuple(sorted(new_edge)))

                # Update face normals.
                face_normals[face_idx_1] = new_normals_face_1
                face_normals[face_idx_2] = new_normals_face_2

                num_flipped_edges += 1

    mesh_after_edge_flipping = pymesh.form_mesh(mesh.vertices,
                                                faces_after_augmentation)

    return mesh_after_edge_flipping


def __rotate_vector_to_position(vector, final_position_first_element):
    r"""Rotates a vector left to right so that its leftmost element (index 0)
    has a certain final index in the vector after the rotation. For example, if
    the vector is `[1, 4, 7, 5]` and :obj:`final_position_first_element` is `2`,
    then the vector after rotation will be `[7, 5, 1, 4]`; likewise, if
    `final_position_first_element` is `1`, then the vector after rotation will
    be `[5, 1, 4, 7]`, and so on.

    Args:
        vector (list): Vector to rotate.
        final_position_first_element (int): Final index of the leftmost element
            of the input vector after rotation (cf. above).

    Returns:
        rotated_vector (list): Vector after rotation.
    """
    vector_length = len(vector)
    assert (0 <= final_position_first_element < vector_length)
    rotated_vector = vector[
        vector_length -
        final_position_first_element:] + vector[:vector_length -
                                                final_position_first_element]

    return rotated_vector


def are_areas_nonnull(mesh, faces):
    r"""Checks whether the areas of the input faces is not null.

    Args:
        mesh (pymesh.Mesh.Mesh): Mesh from which the vertices in the faces are
            taken.
        faces (numpy array of shape `[2, 3]`): The `i`-th element contains the
            list of the indices of the vertices in the `i`-th face, with the
            order defining the half-edges of the face.

    Returns:
        True if the areas of both the two faces is not null, False otherwise.           
    """
    assert (faces.shape[0] == 2)
    face_normals = np.cross(
        mesh.vertices[faces[:, 1]] - mesh.vertices[faces[:, 0]],
        mesh.vertices[faces[:, 2]] - mesh.vertices[faces[:, 1]])
    face_areas = np.sqrt((face_normals**2).sum(axis=1))
    face_areas *= 0.5
    # Verify also that there are no (almost) parallel edges.
    no_parallel_edges = True
    for face_idx in range(2):
        edge_1 = mesh.vertices[faces[face_idx,
                                     1]] - mesh.vertices[faces[face_idx, 0]]
        edge_direction_1 = edge_1 / (np.linalg.norm(edge_1) + 1.e-15)
        edge_2 = mesh.vertices[faces[face_idx,
                                     2]] - mesh.vertices[faces[face_idx, 1]]
        edge_direction_2 = edge_2 / (np.linalg.norm(edge_2) + 1.e-15)
        edge_3 = mesh.vertices[faces[face_idx,
                                     0]] - mesh.vertices[faces[face_idx, 2]]
        edge_direction_3 = edge_3 / (np.linalg.norm(edge_3) + 1.e-15)
        not_parallel_12 = (0.02 <= np.arccos(
            edge_direction_1.dot(edge_direction_2)) <= np.pi - 0.02)
        not_parallel_13 = (0.02 <= np.arccos(
            edge_direction_1.dot(edge_direction_3)) <= np.pi - 0.02)
        not_parallel_23 = (0.02 <= np.arccos(
            edge_direction_2.dot(edge_direction_3)) <= np.pi - 0.02)
        no_parallel_edges &= (not_parallel_12 and not_parallel_13 and
                              not_parallel_23)

    return (face_areas[0] > 0 and face_areas[1] > 0 and no_parallel_edges)
