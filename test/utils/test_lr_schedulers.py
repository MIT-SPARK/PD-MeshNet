import os.path as osp
from torch.optim.lr_scheduler import LambdaLR, StepLR, ReduceLROnPlateau
import unittest

from pd_mesh_net.utils import create_model, create_optimizer, create_lr_scheduler


class TestCreateLrScheduler(unittest.TestCase):

    def test_create_lambda_scheduler(self):
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
        # Create an optimizer.
        optimizer_params = {
            'optimizer_type': 'adam',
            'betas': (0.9, 0.999),
            'lr': 0.0001
        }
        optimizer = create_optimizer(network_parameters=model.parameters(),
                                     **optimizer_params)
        # Create the scheduler.
        scheduler_params = {
            'scheduler_type': 'lambda',
            'last_epoch_constant_lr': 100,
            'last_epoch': 300
        }
        lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                           **scheduler_params)
        self.assertTrue(isinstance(lr_scheduler, LambdaLR))

        # Verify that the learning rate decays linearly over the epochs after
        # epoch 100.
        for epoch in range(1, scheduler_params['last_epoch'] + 1):
            if (epoch <= scheduler_params['last_epoch_constant_lr']):
                self.assertEqual(lr_scheduler.get_last_lr()[0],
                                 optimizer_params['lr'])
            else:
                expected_lr = optimizer_params['lr'] * (
                    ((epoch + 1) - scheduler_params['last_epoch'] - 1) /
                    (scheduler_params['last_epoch_constant_lr'] -
                     scheduler_params['last_epoch'] - 1))
                self.assertAlmostEqual(lr_scheduler.get_last_lr()[0],
                                       expected_lr, 5)
            # Verify that the learning-rate scheduler is considering the right
            # epoch. Since at the first epoch the learning-rate scheduler is
            # internally initialized to have epoch 0, the epoch "counter" in the
            # scheduler should always lag the actual epoch number by 1.
            # However, it should be noted that our LRStep scheduler internally
            # adjusts the epoch number, when using it to compute the learning
            # rate, by adding 1 to it.
            self.assertEqual(lr_scheduler.last_epoch, epoch - 1)
            # Update optimizer and learning-rate scheduler.
            optimizer.step()
            lr_scheduler.step()
        # Look at two more epochs and verify that the learning rate stays at
        # zero.
        self.assertEqual(lr_scheduler.get_last_lr()[0], 0.)
        self.assertEqual(lr_scheduler.last_epoch, 300)
        optimizer.step()
        lr_scheduler.step()
        self.assertEqual(lr_scheduler.get_last_lr()[0], 0.)
        self.assertEqual(lr_scheduler.last_epoch, 301)

    def test_create_step_scheduler(self):
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
        # Create an optimizer.
        optimizer_params = {
            'optimizer_type': 'adam',
            'betas': (0.9, 0.999),
            'lr': 0.0001
        }
        optimizer = create_optimizer(network_parameters=model.parameters(),
                                     **optimizer_params)
        # Create the scheduler.
        scheduler_params = {
            'scheduler_type': 'step',
            'step_size': 20,
            'gamma': 0.2,
        }
        lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                           **scheduler_params)
        self.assertTrue(isinstance(lr_scheduler, StepLR))

        # Verify that the learning rate is multiplied by 'gamma' every
        # 'step_size' steps of the learning-rate scheduler.
        num_epochs = 300
        last_lr = optimizer_params['lr']
        for epoch in range(1, num_epochs + 1):
            if (epoch != 1 and
                (epoch - 1) % scheduler_params['step_size'] == 0):
                # Update the learning rate.
                last_lr *= scheduler_params['gamma']
            # Verify the learning rate.
            self.assertAlmostEqual(lr_scheduler.get_last_lr()[0], last_lr, 5)
            # Verify that the learning-rate scheduler is considering the right
            # epoch. Since at the first epoch the learning-rate scheduler is
            # internally initialized to have epoch 0, the epoch "counter" in the
            # scheduler should always lag the actual epoch number by 1.
            self.assertEqual(lr_scheduler.last_epoch, epoch - 1)
            # Update optimizer and learning-rate scheduler.
            optimizer.step()
            lr_scheduler.step()

    def test_create_plateau_scheduler(self):
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
        # Create an optimizer.
        optimizer_params = {
            'optimizer_type': 'adam',
            'betas': (0.9, 0.999),
            'lr': 0.0001
        }
        optimizer = create_optimizer(network_parameters=model.parameters(),
                                     **optimizer_params)
        # Create the scheduler.
        scheduler_params = {
            'scheduler_type': 'plateau',
            'mode': 'min',
            'factor': 0.2,
            'threshold': 0.01,
            'patience': 5
        }
        lr_scheduler = create_lr_scheduler(optimizer=optimizer,
                                           **scheduler_params)
        self.assertTrue(isinstance(lr_scheduler, ReduceLROnPlateau))

        # Not implemented the actual test, cf. ReduceLROnPlateau for example
        # usage.

    def test_create_nonexistent_scheduler(self):
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
        # Create an optimizer.
        optimizer_params = {
            'optimizer_type': 'adam',
            'betas': (0.9, 0.999),
            'lr': 0.0001
        }
        optimizer = create_optimizer(network_parameters=model.parameters(),
                                     **optimizer_params)
        # Create the scheduler.
        scheduler_params = {'scheduler_type': 'nonexistent_lr_scheduler'}
        self.assertRaises(KeyError,
                          create_lr_scheduler,
                          optimizer=optimizer,
                          **scheduler_params)
