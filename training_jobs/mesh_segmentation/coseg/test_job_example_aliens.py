import os

from pd_mesh_net.utils import BaseTestJob

DATASET_PARAMS = {
    'root': '../../../datasets/coseg_config_A_aliens/',
    'dataset_name': 'coseg',
    'categories': ['aliens'],
    'train': False,
    'num_augmentations': 1,
    'single_dual_nodes': True,
    'undirected_dual_edges': True
}
DATA_AUGMENTATION_PARAMS = {}
DATALOADER_PARAMS = {'batch_size': 16}
TEST_PARAMS = {
    'log_folder': os.path.abspath('../../../training_logs/'),
    'task_type': 'segmentation',
    # Example. Replace with your job name.
    'training_job_name': 'YYYYMMDD_hhmmss',
    'save_clusterized_meshes': True
}

if __name__ == '__main__':
    # Create test job.
    test_job = BaseTestJob(dataset_parameters={
        **DATASET_PARAMS,
        **DATA_AUGMENTATION_PARAMS
    },
                           data_loader_parameters=DATALOADER_PARAMS,
                           **TEST_PARAMS)
    # Run test.
    test_job.test()
