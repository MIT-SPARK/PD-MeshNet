import numpy as np
import os.path as osp
import torch
import unittest

from pd_mesh_net.models import DualPrimalMeshClassifier
from pd_mesh_net.utils import create_graphs, create_dual_primal_batch

current_dir = osp.dirname(__file__)


class TestDualPrimalMeshClassifier(unittest.TestCase):

    def test_forward(self):
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        # Create a mesh classifier.
        in_channels_primal = 1
        in_channels_dual = 4
        num_groups_norm_layer = 16
        convolution_out_res = [64, 128, 256, 256]
        primal_attention_coeffs_thresholds = [None, None, None, 0.5]
        num_classes = 30
        num_output_units_fc = 100

        mesh_classifier = DualPrimalMeshClassifier(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            norm_layer_type='group_norm',
            num_groups_norm_layer=num_groups_norm_layer,
            conv_primal_out_res=convolution_out_res,
            conv_dual_out_res=convolution_out_res,
            primal_attention_coeffs_thresholds=
            primal_attention_coeffs_thresholds,
            num_classes=num_classes,
            num_output_units_fc=num_output_units_fc,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            log_ratios_new_old_primal_nodes=False)
        # Vanilla batch: simply use the same dual-primal graph pair twice.
        primal_graph_list = [primal_graph, primal_graph]
        dual_graph_list = [dual_graph, dual_graph]
        primal_edge_to_dual_node_idx_list = [
            primal_edge_to_dual_node_idx, primal_edge_to_dual_node_idx
        ]
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch) = create_dual_primal_batch(
             primal_graph_list, dual_graph_list,
             primal_edge_to_dual_node_idx_list)
        # Perform classification.
        unnormalized_scores, _ = mesh_classifier(
            primal_graph_batch=primal_graph_batch,
            dual_graph_batch=dual_graph_batch,
            primal_edge_to_dual_node_idx_batch=(
                primal_edge_to_dual_node_idx_batch))
        # Simply check that the convolution outputs tensors with the right
        # shape.
        self.assertEqual(unnormalized_scores.shape[-1], num_classes)

    def test_forward_multi_head(self):
        single_dual_nodes = False
        undirected_dual_edges = True
        graph_creator = create_graphs.GraphCreator(
            mesh_filename=osp.join(current_dir,
                                   '../common_data/simple_mesh.ply'),
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            primal_features_from_dual_features=False)
        primal_graph, dual_graph = graph_creator.create_graphs()
        (primal_edge_to_dual_node_idx
        ) = graph_creator.primal_edge_to_dual_node_idx
        # Create a mesh classifier.
        in_channels_primal = 1
        in_channels_dual = 4
        num_groups_norm_layer = 16
        convolution_out_res = [64, 128, 256, 256]
        num_heads = 2
        num_classes = 30
        num_output_units_fc = 100

        mesh_classifier = DualPrimalMeshClassifier(
            in_channels_primal=in_channels_primal,
            in_channels_dual=in_channels_dual,
            norm_layer_type=None,
            num_groups_norm_layer=num_groups_norm_layer,
            conv_primal_out_res=convolution_out_res,
            conv_dual_out_res=convolution_out_res,
            heads=num_heads,
            num_classes=num_classes,
            num_output_units_fc=num_output_units_fc,
            single_dual_nodes=single_dual_nodes,
            undirected_dual_edges=undirected_dual_edges,
            log_ratios_new_old_primal_nodes=False)
        # Vanilla batch: simply use the same dual-primal graph pair twice.
        primal_graph_list = [primal_graph, primal_graph]
        dual_graph_list = [dual_graph, dual_graph]
        primal_edge_to_dual_node_idx_list = [
            primal_edge_to_dual_node_idx, primal_edge_to_dual_node_idx
        ]
        (primal_graph_batch, dual_graph_batch,
         primal_edge_to_dual_node_idx_batch) = create_dual_primal_batch(
             primal_graph_list, dual_graph_list,
             primal_edge_to_dual_node_idx_list)
        # Perform classification.
        unnormalized_scores, _ = mesh_classifier(
            primal_graph_batch=primal_graph_batch,
            dual_graph_batch=dual_graph_batch,
            primal_edge_to_dual_node_idx_batch=(
                primal_edge_to_dual_node_idx_batch))
        # Simply check that the convolution outputs tensors with the right
        # shape.
        self.assertEqual(unnormalized_scores.shape[-1], num_classes)