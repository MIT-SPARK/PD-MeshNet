import os.path as osp
from torch.optim import Adam
import unittest

from pd_mesh_net.utils import create_model, create_optimizer


class TestCreateOptimizer(unittest.TestCase):

    def test_create_adam_optimizer(self):
        # Create a model.
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
        # Create the optimizer.
        optimizer_params = {'optimizer_type': 'adam', 'betas': (0.9, 0.999)}
        optimizer = create_optimizer(network_parameters=model.parameters(),
                                     **optimizer_params)
        self.assertTrue(isinstance(optimizer, Adam))
        # Warning for nonexistent parameter.

    def test_create_nonexistent_optimizer(self):
        # Create a model.
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
        # Create the optimizer.
        optimizer_params = {'optimizer_type': 'nonexistent_optimizer'}
        self.assertRaises(KeyError,
                          create_optimizer,
                          network_parameters=model.parameters(),
                          **optimizer_params)
