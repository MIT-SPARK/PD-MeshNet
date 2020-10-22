import os

from pd_mesh_net.utils import BaseTestJob

DATASET_PARAMS = {
    'root': '../../../datasets/shrec_16_config_A//',
    'dataset_name': 'shrec_16',
    'compute_node_feature_stats': False,
    'categories': [],
    'train': False,
    'single_dual_nodes': True,
    'undirected_dual_edges': True
}
DATALOADER_PARAMS = {'batch_size': 16}
TEST_PARAMS = {
    'log_folder': os.path.abspath('../../../training_logs/'),
    'task_type': 'classification',
    # Example. Replace with your job name.
    'training_job_name': 'YYYYMMDD_hhmmss'
}

if __name__ == '__main__':
    # Create test job.
    test_job = BaseTestJob(dataset_parameters=DATASET_PARAMS,
                           data_loader_parameters=DATALOADER_PARAMS,
                           **TEST_PARAMS)
    # Run test.
    test_job.test()
