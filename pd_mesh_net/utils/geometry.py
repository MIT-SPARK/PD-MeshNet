import numpy as np
import pymesh


def local_indices_edges(face_1, face_2):
    r"""Letting `e = (v_1, v_2)` be the edge shared by the two input faces,
    returns the local index of `v_1` and `v_2` in the face vector of the two
    input faces (cf. return value).

    Args:
        face_1, face_2 (list of int): Faces - in the format of a list containing
            the indices of the vertices in each face, ordered so as to define
            the half-edges in each face - of which to compute the 'local
            indices of the common edge'.

    Returns:
        local_indices_common_edge (list of int): If the two input faces have an
            opposite half-edge, the `i`-th element - with
            :math:`i \in \{0, 1\}`, contains the local index in `face_{i+1}` of
            the vertex shared by the two faces that is first encountered when
            trasvering the vertices in face_1 with increasing local indices,
            i.e., the two faces have mirroring half-edges `H1` and `H2`, where
            `H1 = (face_1[local_indices_first_common_vertex[0]],
                face_1[local_indices_first_common_vertex[0] + 1])` and
            `H2 = (face_2[local_indices_first_common_vertex[1]],
                face_2[local_indices_first_common_vertex[1] - 1])`.
            Example:
                `face_1 = [0, 3, 1], face_2 = [4, 1, 3]` ->
                `local_indices_common_edge[0] = 1,
                 local_indices_common_edge[1] = 2.`
            If the two input face share a half-edge with equal orientation
            (hence the edge between them is non-manifold), None is returned.
    
    Raises:
        An exception is raised if the two input faces do not have an opposite
        half-edge and do not share a half-edge with equal orientation.
    """
    # Find common edge.
    local_indices_common_edge = [None, None]
    try:
        local_indices_common_edge[1] = face_2.index(face_1[0])
        local_indices_common_edge[0] = 0
    except ValueError:
        local_indices_common_edge[0] = 1
        local_indices_common_edge[1] = face_2.index(face_1[1])
        if (face_2[local_indices_common_edge[1] - 1] != face_1[2]):
            # At this point, the two faces either share a common half-edge
            # (non-manifold edge) - in which case we return None - or they do
            # not share an edge at all.
            assert (face_2[(local_indices_common_edge[1] + 1) % 3] == face_1[2])
            return None
    else:
        if (face_2[local_indices_common_edge[1] - 1] != face_1[1]):
            if (face_2[(local_indices_common_edge[1] + 1) % 3] != face_1[2]):
                # At this point, the two faces either share a common half-edge
                # (non-manifold edge) - in which case we return None - or they
                # do not share an edge at all.
                assert (face_2[(local_indices_common_edge[1] + 1) %
                               3] == face_1[1] or
                        face_2[local_indices_common_edge[1] - 1] == face_1[2])
                return None
            local_indices_common_edge[1] = (local_indices_common_edge[1] +
                                            1) % 3
            local_indices_common_edge[0] = 2

    return local_indices_common_edge


def cross_product(vector_left, vector_right):
    r"""Returns the cross product between two input (3D) vectors. Faster than
    `numpy.cross`.

    Args:
        vector_left, vector_right (numpy array of shape :obj:`[3, ]`): Vector
            of which to compute the cross product.

    Returns
        vector_cross_product (numpy array of shape :obj:`[3, ]`): Cross product
            of the two input vectors.
    """
    vector_cross_product = np.array([
        (vector_left[1] * vector_right[2]) - (vector_left[2] * vector_right[1]),
        (vector_left[2] * vector_right[0]) - (vector_left[0] * vector_right[2]),
        (vector_left[0] * vector_right[1]) - (vector_left[1] * vector_right[0])
    ])

    return vector_cross_product


def dihedral_angle_and_local_indices_edges(mesh,
                                           face_normals,
                                           face_indices=None,
                                           faces=None):
    r"""Returns the dihedral angle (in radians) between the two faces with given
    indices in the input mesh; furthermore, returns the 'local indices of the
    common edge' (cf. function :obj:`local_indices_edges`).

    Args:
        mesh (pymesh.Mesh.Mesh): Input mesh.
        face_normals (numpy array of shape :obj:`[2, ]`): Normalized face normal
            vectors of the two faces around the input edge.
        face_indices (list of int, optional): If not None, indices in the input
            mesh of the two faces between which to compute the dihedral angle.
            One can alternatively manually specify the faces in as the list of
            vertices in it (cf. argument `faces`). (default: :obj:`None`)
        faces (list of list, optional): If not None, faces - in the format of a
            list containing two lists as element, encoding the indices of the
            vertices in the first and the second face, respectively. The
            vertices must be ordered so as to define the half-edges in each
            face. (default: :obj:`None`)
    
    Returns:
        dihedral_angle (float): Dihedral angle in radians between the two input
            faces. If the two faces share a half-edge with the same direction
            (i.e., they share a non-manifold edge), None.
        local_indices_common_edge (list of int): The `i`-th element - with
            :math:`i \in \{0, 1\}`, contains the local index in `face_{i+1}` of
            the vertex shared by the two faces that is first encountered when
            trasvering the vertices in face_1 with increasing local indices,
            i.e., the two faces have mirroring half-edges `H1` and `H2`, where
            `H1 = (face_1[local_indices_first_common_vertex[0]],
                   face_1[local_indices_first_common_vertex[0] + 1])` and
            `H2 = (face_2[local_indices_first_common_vertex[1]],
                   face_2[local_indices_first_common_vertex[1] - 1])`.
            Example:
                `face_1 = [0, 3, 1], face_2 = [4, 1, 3]` ->
                `local_indices_common_edge[0] = 1,
                local_indices_common_edge[1] = 2.`
            If the two faces share a half-edge with the same direction (i.e.,
            they share a non-manifold edge), None.
    """
    assert (len(face_normals) == 2)
    if (face_indices is not None):
        assert (faces is None)
        assert (isinstance(face_indices, list) and len(face_indices) == 2)
        face_idx_1, face_idx_2 = face_indices
        assert (all([0 <= face_idx_1, face_idx_2 < mesh.num_faces]))
        face_1 = mesh.faces[face_idx_1].tolist()
        face_2 = mesh.faces[face_idx_2].tolist()
    else:
        assert (faces is not None)
        assert (isinstance(faces, list) and len(faces) == 2)
        face_1, face_2 = faces
        assert (isinstance(face_1, list) and len(face_1) == 3)
        assert (isinstance(face_2, list) and len(face_2) == 3)

    # Find common edge.
    local_indices_common_edge = local_indices_edges(face_1, face_2)
    if (local_indices_common_edge is None):
        # The two faces share a half-edge with the same direction. One should
        # call this function again with one of the faces flipped.
        return None, None
    # - Half-edge in face 1.
    common_edge_seen_from_face_1 = mesh.vertices[face_1[
        (local_indices_common_edge[0] + 1) %
        3]] - mesh.vertices[face_1[local_indices_common_edge[0]]]
    edge_length = np.linalg.norm(common_edge_seen_from_face_1)

    # Find height vector.
    normalized_common_edge = common_edge_seen_from_face_1 / edge_length
    height_vector_1 = cross_product(face_normals[0], normalized_common_edge)
    height_vector_2 = -cross_product(face_normals[1], normalized_common_edge)
    # Project the two height vectors on the plane having normal coinciding
    # in direction with the direction of the edge between the two faces.
    # - Taking the height vector from face 2 as the x axis of the local
    #   2D coordinate frame, compute, the x and y coordinates of the
    #   projection of the height vector from face 1 on the plane.
    x_height_vector_1 = height_vector_1.dot(height_vector_2)
    y_height_vector_1 = height_vector_1.dot(
        -cross_product(normalized_common_edge, height_vector_2))
    dihedral_angle = np.arctan2(y_height_vector_1, x_height_vector_1)
    # The arctan2 function from numpy returns values in [-pi, pi].
    if (dihedral_angle < 0):
        dihedral_angle += (2 * np.pi)

    return dihedral_angle, local_indices_common_edge
