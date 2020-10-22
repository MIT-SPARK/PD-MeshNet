import os
import pymesh
import unittest

from pd_mesh_net.data import augmentation, post_augmentation

current_dir = os.path.dirname(__file__)


class TestPreprocessData(unittest.TestCase):

    def test_scale_vertices(self):
        # Run and check results visually.
        # - Simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh.ply'))
        mesh = augmentation(mesh=mesh,
                            vertices_scale_mean=1.0,
                            vertices_scale_var=0.1)
        if (not os.path.exists(os.path.join(current_dir, '../output_data/'))):
            os.mkdir(os.path.join(current_dir, '../output_data'))
        pymesh.save_mesh(mesh=mesh,
                         filename=os.path.join(
                             current_dir,
                             '../output_data/simple_mesh_scale_vertices.ply'),
                         ascii=True)
        # - Modified non-flat version of simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh_nonflat.ply'))
        mesh = augmentation(mesh=mesh,
                            vertices_scale_mean=1.0,
                            vertices_scale_var=0.1)
        pymesh.save_mesh(
            mesh=mesh,
            filename=os.path.join(
                current_dir,
                '../output_data/simple_mesh_nonflat_scale_vertices.ply'),
            ascii=True)

    def test_flip_edges(self):
        # Run and check results visually.
        # - Simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh.ply'))
        mesh = augmentation(mesh=mesh, edges_flip_fraction=0.5)
        if (not os.path.exists(os.path.join(current_dir, '../output_data/'))):
            os.mkdir(os.path.join(current_dir, '../output_data'))
        pymesh.save_mesh(mesh=mesh,
                         filename=os.path.join(
                             current_dir,
                             '../output_data/simple_mesh_flip_edges.ply'),
                         ascii=True)
        # - Modified non-flat version of simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh_nonflat.ply'))
        mesh = augmentation(mesh=mesh, edges_flip_fraction=0.5)
        pymesh.save_mesh(
            mesh=mesh,
            filename=os.path.join(
                current_dir,
                '../output_data/simple_mesh_nonflat_flip_edges.ply'),
            ascii=True)

    def test_slide_vertices(self):
        # Run and check results visually.
        # - Simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh.ply'))
        mesh = post_augmentation(mesh=mesh, slide_vertices_fraction=0.5)
        if (not os.path.exists(os.path.join(current_dir, '../output_data/'))):
            os.mkdir(os.path.join(current_dir, '../output_data'))
        pymesh.save_mesh(mesh=mesh,
                         filename=os.path.join(
                             current_dir,
                             '../output_data/simple_mesh_slide_vertices.ply'),
                         ascii=True)
        # - Modified non-flat version of simple mesh.
        mesh = pymesh.load_mesh(
            os.path.join(current_dir, '../common_data/simple_mesh_nonflat.ply'))
        mesh = post_augmentation(mesh=mesh, slide_vertices_fraction=0.5)
        pymesh.save_mesh(
            mesh=mesh,
            filename=os.path.join(
                current_dir,
                '../output_data/simple_mesh_nonflat_slide_vertices.ply'),
            ascii=True)
