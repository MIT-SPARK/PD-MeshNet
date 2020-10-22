import os

from pd_mesh_net.utils import BaseTrainingJob

NETWORK_PARAMS = {
    'model_name': 'unet_mesh_segmenter',
    'in_channels_primal': 1,
    'in_channels_dual': 7,
    'conv_primal_out_res': [32, 64, 128],
    'conv_dual_out_res': [32, 64, 128],
    'num_classes': 8,
    'single_dual_nodes': True,
    'undirected_dual_edges': True,
    'fractions_primal_edges_to_keep': [0.7, 0.7],
    'num_res_blocks': 1,
    'heads_encoder': 2,
    'heads_decoder': 1,
    'weight_initialization_type': 'normal',
    'weight_initialization_gain': 0.02,
    'dropout_primal': 0,
    'dropout_dual': 0,
    'use_dual_primal_res_down_conv_blocks': True,
}
DATA_AUGMENTATION_PARAMS = {}
DATASET_PARAMS = {
    'root':
        '../../../datasets/human_seg_config_A/',
    'dataset_name':
        'human_seg',
    'node_feature_stats_filename':
        '../../../datasets/human_seg_config_A/params_human_seg_config_A.pkl',
    'train':
        True,
    'num_augmentations':
        1,
    'single_dual_nodes':
        True,
    'undirected_dual_edges':
        True
}
TEST_DATASET_PARAMS = {
    'root': '../../../datasets/human_seg_config_A/',
    'dataset_name': 'human_seg',
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
    'lr': 0.001,  # Initial learning rate.
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
