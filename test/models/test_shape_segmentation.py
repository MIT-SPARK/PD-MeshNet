import os.path as osp
import torch
import unittest

from pd_mesh_net.data import DualPrimalDataLoader
from pd_mesh_net.datasets import Shrec2016DualPrimal, CosegDualPrimal
from pd_mesh_net.models import (DualPrimalMeshSegmenter,
                                DualPrimalUNetMeshSegmenter, DualPrimalDownConv,
                                DualPrimalUpConv)
from pd_mesh_net.nn.conv import DualPrimalConv

current_dir = osp.dirname(__file__)


class TestDualPrimalDownConv(unittest.TestCase):

    def test_forward_pass(self):
        # NOTE: this is not an actual unit test, but it is just used to verify
        # that the forward pass goes through.
        single_dual_nodes = False
        undirected_dual_edges = True
        dataset = Shrec2016DualPrimal(
            root=osp.join(current_dir, '../common_data/shrec2016'),
            categories=['shark'],
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            vertices_scale_mean=1.,
            vertices_scale_var=0.1,
            edges_flip_fraction=0.5,
            slide_vertices_fraction=0.2,
            num_augmentations=1)
        batch_size = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        # Test without pooling.
        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=1,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertEqual(log_info, None)
        # Tests with pooling.
        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=1,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            num_primal_edges_to_keep=600,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)

        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=1,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            fraction_primal_edges_to_keep=0.7,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)

        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=1,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            primal_attention_coeff_threshold=0.5,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)

        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=1,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            num_primal_edges_to_keep=600,
            allow_pooling_consecutive_edges=False,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)

        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=5,
            out_channels_dual=5,
            heads=3,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            add_self_loops_to_dual_graph=False,
            num_primal_edges_to_keep=600,
            use_decreasing_attention_coefficients=False,
            num_skips=3)
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_out, dual_graph_out, petdni_out, log_info, _,
             _) = down_conv(primal_graph_batch=primal_graph,
                            dual_graph_batch=dual_graph,
                            primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)


class TestDualPrimalUpConv(unittest.TestCase):

    def test_forward_pass(self):
        # NOTE: this is not an actual unit test, but it is just used to verify
        # that the forward pass goes through and the output features and graphs
        # have the expected format.
        single_dual_nodes = False
        undirected_dual_edges = True
        dataset = Shrec2016DualPrimal(
            root=osp.join(current_dir, '../common_data/shrec2016'),
            categories=['shark'],
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            vertices_scale_mean=1.,
            vertices_scale_var=0.1,
            edges_flip_fraction=0.5,
            slide_vertices_fraction=0.2,
            num_augmentations=1)
        batch_size = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        # Down-convolutional layer.
        out_channels_primal = 5
        out_channels_dual = 5
        out_channels_primal_after_encoder = 7
        out_channels_dual_after_encoder = 7
        heads = 1
        concat_primal = True
        concat_dual = True
        negative_slope_primal = 0.2
        negative_slope_dual = 0.2
        dropout_primal = 0
        dropout_dual = 0
        bias_primal = True
        bias_dual = True
        add_self_loops_to_dual_graph = False
        down_conv = DualPrimalDownConv(
            in_channels_primal=1,
            in_channels_dual=4,
            out_channels_primal=out_channels_primal,
            out_channels_dual=out_channels_dual,
            heads=heads,
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
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph,
            num_primal_edges_to_keep=600,
            num_skips=3,
            return_old_dual_node_to_new_dual_node=True,
            return_graphs_before_pooling=True)
        # Dual-primal convolutional layer.
        conv = DualPrimalConv(
            in_channels_primal=out_channels_primal,
            in_channels_dual=out_channels_dual,
            out_channels_primal=out_channels_primal_after_encoder,
            out_channels_dual=out_channels_dual_after_encoder,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            heads=heads,
            concat_primal=concat_primal,
            concat_dual=concat_dual,
            negative_slope_primal=negative_slope_primal,
            negative_slope_dual=negative_slope_dual,
            dropout_primal=dropout_primal,
            dropout_dual=dropout_dual,
            bias_primal=bias_primal,
            bias_dual=bias_dual,
            add_self_loops_to_dual_graph=add_self_loops_to_dual_graph)
        # Up-convolutional layer.
        up_conv = DualPrimalUpConv(
            in_channels_primal=out_channels_primal_after_encoder,
            in_channels_dual=out_channels_dual_after_encoder,
            out_channels_primal=out_channels_primal,
            out_channels_dual=out_channels_dual,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            concat_data_from_before_pooling=True)
        # Forward pass.
        for primal_graph, dual_graph, petdni in data_loader:
            (primal_graph_after_pooling, dual_graph_after_pooling,
             petdni_after_pooling, log_info, primal_graph_before_pooling,
             dual_graph_before_pooling) = down_conv(
                 primal_graph_batch=primal_graph,
                 dual_graph_batch=dual_graph,
                 primal_edge_to_dual_node_idx_batch=petdni)
            self.assertNotEqual(log_info, None)
            (primal_graph_after_pooling.x, dual_graph_after_pooling.x) = conv(
                x_primal=primal_graph_after_pooling.x,
                x_dual=dual_graph_after_pooling.x,
                edge_index_primal=primal_graph_after_pooling.edge_index,
                edge_index_dual=dual_graph_after_pooling.edge_index,
                primal_edge_to_dual_node_idx=petdni_after_pooling)
            (primal_graph_batch_out, dual_graph_batch_out,
             primal_edge_to_dual_node_idx_batch_out) = up_conv(
                 primal_graph_batch=primal_graph_after_pooling,
                 dual_graph_batch=dual_graph_after_pooling,
                 primal_edge_to_dual_node_idx_batch=petdni_after_pooling,
                 pooling_log=log_info,
                 primal_graph_batch_before_pooling=primal_graph_before_pooling,
                 dual_graph_batch_before_pooling=dual_graph_before_pooling)
        # Check that the original graphs and the 'unpooled' ones match in
        # connectivity.
        self.assertTrue(
            torch.equal(primal_graph.edge_index,
                        primal_graph_batch_out.edge_index))
        self.assertTrue(
            torch.equal(dual_graph.edge_index, dual_graph_batch_out.edge_index))
        self.assertEqual(petdni, primal_edge_to_dual_node_idx_batch_out)
        # Check that the number of output channels of the new features match the
        # one of the features in the original graphs.
        self.assertEqual(primal_graph.num_node_features,
                         primal_graph_after_pooling.num_node_features)
        self.assertEqual(dual_graph.num_node_features,
                         dual_graph_after_pooling.num_node_features)


class TestDualPrimalMeshSegmenter(unittest.TestCase):

    def test_forward_pass(self):
        # NOTE: this is not an actual unit test, but it is just used to verify
        # that the forward pass goes through and that its output are in the
        # correct format.
        single_dual_nodes = True
        undirected_dual_edges = True
        dataset = CosegDualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/coseg_config_A/')),
                                  categories=['aliens'],
                                  single_dual_nodes=single_dual_nodes,
                                  undirected_dual_edges=undirected_dual_edges,
                                  return_sample_indices=True)
        batch_size = 4
        num_classes = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        # Test without pooling.
        mesh_segmenter = DualPrimalMeshSegmenter(
            in_channels_primal=1,
            in_channels_dual=7,
            conv_primal_out_res=[64, 128],
            conv_dual_out_res=[64, 128],
            num_classes=num_classes,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            fractions_primal_edges_to_keep=[0.7, 0.7],
            num_res_blocks=2,
            heads=3,
            concat_primal=True,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            add_self_loops_to_dual_graph=False,
            return_node_to_cluster=False,
            log_ratios_new_old_primal_edges=True)

        for iteration_idx, (primal_graph, dual_graph,
                            petdni) in enumerate(data_loader):
            # - Limit test to first 5 batches.
            if (iteration_idx == 5):
                break
            output_scores, _ = mesh_segmenter(
                primal_graph_batch=primal_graph,
                dual_graph_batch=dual_graph,
                primal_edge_to_dual_node_idx_batch=petdni)
            # Verify that for each output node in the input primal graph one
            # score per each class is returned.
            self.assertEqual(output_scores.shape,
                             (primal_graph.num_nodes, num_classes))


class TestDualPrimalUNetMeshSegmenter(unittest.TestCase):

    def test_forward_pass(self):
        # NOTE: this is not an actual unit test, but it is just used to verify
        # that the forward pass goes through and that its output are in the
        # correct format.
        single_dual_nodes = True
        undirected_dual_edges = True
        dataset = CosegDualPrimal(root=osp.abspath(
            osp.join(current_dir, '../common_data/coseg_config_A/')),
                                  categories=['aliens'],
                                  single_dual_nodes=single_dual_nodes,
                                  undirected_dual_edges=undirected_dual_edges,
                                  return_sample_indices=True)
        batch_size = 4
        num_classes = 4
        data_loader = DualPrimalDataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
        # Test without pooling.
        mesh_segmenter = DualPrimalUNetMeshSegmenter(
            in_channels_primal=1,
            in_channels_dual=7,
            conv_primal_out_res=[32, 64, 128],
            conv_dual_out_res=[32, 64, 128],
            num_classes=num_classes,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            fractions_primal_edges_to_keep=[0.7, 0.7],
            num_res_blocks=2,
            heads_encoder=3,
            heads_decoder=1,
            concat_primal=False,
            concat_dual=True,
            negative_slope_primal=0.2,
            negative_slope_dual=0.2,
            dropout_primal=0,
            dropout_dual=0,
            bias_primal=True,
            bias_dual=True,
            add_self_loops_to_dual_graph=False,
            return_node_to_cluster=False,
            log_ratios_new_old_primal_edges=True)

        for iteration_idx, (primal_graph, dual_graph,
                            petdni) in enumerate(data_loader):
            # - Limit test to first 5 batches.
            if (iteration_idx == 5):
                break
            output_scores, _ = mesh_segmenter(
                primal_graph_batch=primal_graph,
                dual_graph_batch=dual_graph,
                primal_edge_to_dual_node_idx_batch=petdni)
            # Verify that for each output node in the input primal graph one
            # score per each class is returned.
            self.assertEqual(output_scores.shape,
                             (primal_graph.num_nodes, num_classes))
