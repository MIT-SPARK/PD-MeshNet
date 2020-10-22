import numpy as np
import pymesh


def preprocess_mesh(input_mesh, prevent_nonmanifold_edges=True):
    r"""Removes duplicated vertices, duplicated faces, zero-area faces, and
    optionally faces the insertion of which would cause an edge of the mesh to
    become non-manifold. In particular, an iteration is performed over all faces
    in the mesh, keeping track of the half-edges in each face, and if a face
    contains a half-edge already found in one of the faces previously processed,
    it gets removed from the mesh.

    Args:
        input_mesh (pymesh.Mesh.Mesh): Mesh to preprocess.
        prevent_nonmanifold_edges (bool, optional): If True, faces that would
            cause an edge to become non-manifold are removed from the mesh (cf.
            above). (default: :obj:`True`)

    Returns:
        output_mesh (pymesh.Mesh.Mesh): Mesh after preprocessing.
    """
    halfedges_found = set()
    new_faces = np.empty([input_mesh.num_faces, 3])
    # Remove duplicated vertices.
    input_mesh = pymesh.remove_duplicated_vertices(input_mesh)[0]
    # Remove duplicated faces.
    input_mesh = pymesh.remove_duplicated_faces(input_mesh)[0]
    num_valid_faces = 0
    # Compute face areas so that zero-are faces can be removed.
    input_mesh.add_attribute("face_area")
    face_areas = input_mesh.get_face_attribute("face_area")
    assert (len(face_areas) == len(input_mesh.faces))
    for face, face_area in zip(input_mesh.faces, face_areas):
        # Do not include the face if it does not have three different
        # vertices.
        if (face[0] == face[1] or face[0] == face[2] or face[1] == face[2]):
            continue
        # Do not include zero-area faces.
        if (face_area == 0):
            continue
        # Optionally prevent non-manifold edges.
        if (prevent_nonmanifold_edges):
            new_halfedges_in_face = set()
            for idx in range(3):
                halfedge = (face[idx], face[(idx + 1) % 3])
                if (halfedge[0] != halfedge[1]):  # Exclude self-loops.
                    if (halfedge not in halfedges_found):
                        # Half-edge not found previously. -> The edge is
                        # manifold so far.
                        new_halfedges_in_face.add(halfedge)
            if (len(new_halfedges_in_face) == 3):
                # Face does not introduce non-manifold edges.
                halfedges_found.update(new_halfedges_in_face)
                new_faces[num_valid_faces] = face
                num_valid_faces += 1
                # Here one can compute the face features already.
        else:
            new_faces[num_valid_faces] = face
            num_valid_faces += 1

    new_faces = new_faces[:num_valid_faces]
    output_mesh = pymesh.form_mesh(input_mesh.vertices, new_faces)
    # Not including faces might have caused vertices to become isolated.
    output_mesh = pymesh.remove_isolated_vertices(output_mesh)[0]

    return output_mesh
