import os

from pd_mesh_net.utils import BaseTrainingJob

NETWORK_PARAMS = {
    'model_name': 'mesh_segmenter',
    'in_channels_primal': 1,
    'in_channels_dual': 7,
    'conv_primal_out_res': [64, 96],
    'conv_dual_out_res': [64, 96],
    'num_classes': 4,
    'single_dual_nodes': True,
    'undirected_dual_edges': True,
    'fractions_primal_edges_to_keep': [0.8, 0.8],
    'num_res_blocks': 1,
    'heads': 1,
    'weight_initialization_type': 'normal',
    'weight_initialization_gain': 0.02,
    'dropout_primal': 0,
    'dropout_dual': 0
}
DATA_AUGMENTATION_PARAMS = {}
DATASET_PARAMS = {
    'root': '../../../datasets/coseg_config_A_aliens/',
    'dataset_name': 'coseg',
    'node_feature_stats_filename': '../../../datasets/coseg_config_A_aliens/'
                                   'params_coseg_config_A_aliens.pkl',
    'categories': ['aliens'],
    'train': True,
    'num_augmentations': 1,
    'single_dual_nodes': True,
    'undirected_dual_edges': True
}
TEST_DATASET_PARAMS = {
    'root': '../../../datasets/coseg_config_A_aliens/',
    'dataset_name': 'coseg',
    'categories': ['aliens'],
    'train': False,
    'num_augmentations': 1,
    'single_dual_nodes': True,
    'undirected_dual_edges': True
}
DATALOADER_PARAMS = {
    'batch_size': 16,
    'shuffle': True,
    'validation_set_fraction': None
}
LOSS_PARAMS = {'task_type': 'segmentation'}
LR_SCHEDULER_PARAMS = {
    'scheduler_type': 'lambda',
    'last_epoch_constant_lr': 1000,
    'last_epoch': 1000
}
OPTIMIZER_PARAMS = {
    'optimizer_type': 'adam',
    'betas': (0.9, 0.999),
    'lr': 0.0002,  # Initial learning rate.
}
TRAINING_PARAMS = {
    'final_training_epoch': 1000,
    'log_folder': os.path.abspath('../../../training_logs/'),
    'use_gpu': True
}

if __name__ == '__main__':
    # Create training job.
    training_job = BaseTrainingJob(network_parameters=NETWORK_PARAMS,
                                   dataset_parameters={
                                       **DATASET_PARAMS,
                                       **DATA_AUGMENTATION_PARAMS
                                   },
                                   test_dataset_parameters=TEST_DATASET_PARAMS,
                                   data_loader_parameters=DATALOADER_PARAMS,
                                   optimizer_parameters=OPTIMIZER_PARAMS,
                                   lr_scheduler_parameters=LR_SCHEDULER_PARAMS,
                                   loss_parameters=LOSS_PARAMS,
                                   **TRAINING_PARAMS)
    # Run training job.
    training_job.train()
