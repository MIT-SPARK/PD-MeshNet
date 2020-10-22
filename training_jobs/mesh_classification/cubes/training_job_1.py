import os

from pd_mesh_net.utils import BaseTrainingJob

NETWORK_PARAMS = {
    'model_name': 'mesh_classifier',
    'in_channels_primal': 1,
    'in_channels_dual': 7,
    'norm_layer_type': 'group_norm',
    'num_groups_norm_layer': 16,
    'conv_primal_out_res': [64, 128],
    'conv_dual_out_res': [64, 128],
    'num_classes': 22,
    'num_output_units_fc': 100,
    'num_res_blocks': 1,
    'weight_initialization_type': 'normal',
    'weight_initialization_gain': 0.02,
    'fractions_primal_edges_to_keep': [0.8, 0.8],
    'heads': 3,
    'single_dual_nodes': True,
    'undirected_dual_edges': True,
    'dropout_dual': 0,
    'dropout_primal': 0
}
DATA_AUGMENTATION_PARAMS = {}
DATASET_PARAMS = {
    'root':
        '../../../datasets/cubes_config_A/',
    'dataset_name':
        'cubes',
    'node_feature_stats_filename':
        '../../../datasets/cubes_config_A/params_cubes_A.pkl',
    'categories': [],
    'train':
        True,
    'num_augmentations':
        1,
    'single_dual_nodes':
        True,
    'undirected_dual_edges':
        True
}
DATALOADER_PARAMS = {
    'batch_size': 16,
    'shuffle': True,
    'validation_set_fraction': None
}
LOSS_PARAMS = {'task_type': 'classification'}
LR_SCHEDULER_PARAMS = {
    'scheduler_type': 'lambda',
    'last_epoch_constant_lr': 200,
    'last_epoch': 200
}
OPTIMIZER_PARAMS = {
    'optimizer_type': 'adam',
    'betas': (0.9, 0.999),
    'lr': 0.0002,  # Initial learning rate.
}
TRAINING_PARAMS = {
    'final_training_epoch': 1000,
    'log_folder': os.path.abspath('../../../training_logs/')
}

if __name__ == '__main__':
    # Create training job.
    training_job = BaseTrainingJob(network_parameters=NETWORK_PARAMS,
                                   dataset_parameters={
                                       **DATASET_PARAMS,
                                       **DATA_AUGMENTATION_PARAMS
                                   },
                                   data_loader_parameters=DATALOADER_PARAMS,
                                   optimizer_parameters=OPTIMIZER_PARAMS,
                                   lr_scheduler_parameters=LR_SCHEDULER_PARAMS,
                                   loss_parameters=LOSS_PARAMS,
                                   **TRAINING_PARAMS)
    # Run training job.
    training_job.train()
