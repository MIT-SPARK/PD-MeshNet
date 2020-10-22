import os.path as osp
import unittest

from pd_mesh_net.models import DualPrimalMeshClassifier
from pd_mesh_net.utils import create_model


class TestCreateModel(unittest.TestCase):

    def test_create_mesh_classifier(self):
        model_name = 'mesh_classifier'
        network_params = {
            'in_channels_primal': 1,
            'in_channels_dual': 4,
            'norm_layer_type': 'group_norm',
            'num_groups_norm_layer': 16,
            'conv_primal_out_res': [64, 128, 256, 256],
            'conv_dual_out_res': [64, 128, 256, 256],
            'num_classes': 30,
            'num_output_units_fc': 100,
            'num_res_blocks': 1,
            'single_dual_nodes': False,
            'undirected_dual_edges': True
        }
        model = create_model(model_name=model_name,
                             should_initialize_weights=True,
                             **network_params)
        self.assertTrue(isinstance(model, DualPrimalMeshClassifier))

    def test_create_nonexistent_model(self):
        model_name = 'nonexistent_model'
        self.assertRaises(KeyError,
                          create_model,
                          model_name=model_name,
                          should_initialize_weights=True)

    def test_normal_weight_initializer(self):
        model_name = 'mesh_classifier'
        network_params = {
            'in_channels_primal': 1,
            'in_channels_dual': 4,
            'norm_layer_type': 'group_norm',
            'num_groups_norm_layer': 16,
            'conv_primal_out_res': [64, 128, 256, 256],
            'conv_dual_out_res': [64, 128, 256, 256],
            'num_classes': 30,
            'num_output_units_fc': 100,
            'num_res_blocks': 1,
            'weight_initialization_type': 'normal',
            'weight_initialization_gain': 0.02,
            'single_dual_nodes': False,
            'undirected_dual_edges': True
        }
        model = create_model(model_name=model_name,
                             should_initialize_weights=True,
                             **network_params)
        # Print the weights of a fully-connected layer.
        print(
            "\nThe following weights from a fully-connected layer should look "
            "like samples drawn from a Gaussian distribution with mean 0. and "
            f"variance {network_params['weight_initialization_gain']**2}:")
        print(model.fc1.weight.data)
        # Print the weights of a batch-normalization layer.
        print("\nThe following weights from a batch-normalization layer should "
              "look like samples drawn from a Gaussian distribution with mean "
              "1.0 and variance "
              f"{network_params['weight_initialization_gain']**2}:")
        print(model.block0.conv.bn1_primal.weight.data)

    def test_nonexistent_weight_initializer(self):
        model_name = 'mesh_classifier'
        network_params = {
            'in_channels_primal': 1,
            'in_channels_dual': 4,
            'norm_layer_type': 'group_norm',
            'num_groups_norm_layer': 16,
            'conv_primal_out_res': [64, 128, 256, 256],
            'conv_dual_out_res': [64, 128, 256, 256],
            'num_classes': 30,
            'num_output_units_fc': 100,
            'num_res_blocks': 1,
            'weight_initialization_type': 'nonexistent',
            'weight_initialization_gain': 0.02,
            'single_dual_nodes': False,
            'undirected_dual_edges': True
        }
        self.assertRaises(NotImplementedError,
                          create_model,
                          model_name=model_name,
                          should_initialize_weights=True,
                          **network_params)
