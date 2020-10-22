import argparse
import glob
import numpy as np
import os

from models.layers.mesh import Mesh
from label_converter import (FaceLabelToEdgeSoftLabelConverter,
                             EdgeLabelToFaceLabelConverter, OptClass)
from data.segmentation_data import read_seg


def face_labels_to_soft_edge_labels(dataset_name, meshcnn_root,
                                    root_segmented_meshes,
                                    output_soft_labels_folder):
    if (dataset_name in ['coseg_aliens', 'coseg_vases']):
        num_classes = 4
    elif (dataset_name in ['coseg_chairs']):
        num_classes = 3
    elif (dataset_name == 'human_seg'):
        num_classes = 8
    else:
        raise ValueError(
            "Please specify one of the following dataset names: "
            "['coseg_aliens', 'coseg_chairs', 'coseg_vases', 'human_seg'].")

    folder_original_meshes = os.path.join(meshcnn_root,
                                          f'datasets/{dataset_name}/test/')
    segmented_meshes = glob.glob(
        os.path.join(root_segmented_meshes, '*clusterized.ply'))

    for mesh_filename in segmented_meshes:
        mesh_name = mesh_filename.rsplit('.')[0].rpartition('/')[-1]
        short_mesh_name = mesh_name.split('_aug')[0]
        print(f"Converted labels for '{mesh_name}'.")
        mesh_original_filename = os.path.join(folder_original_meshes,
                                              f'{short_mesh_name}.obj')
        label_converter = FaceLabelToEdgeSoftLabelConverter(
            mesh_original_filename=mesh_original_filename,
            mesh_filename=mesh_filename,
            num_classes=num_classes,
            edge_soft_labels_file=os.path.join(
                output_soft_labels_folder,
                f'{dataset_name}/sseg/{mesh_name}.seseg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',
                        type=str,
                        help="Name of the dataset, one of ['coseg_aliens', "
                        "'coseg_chairs', 'coseg_vases', 'human_seg'].",
                        required=True)
    parser.add_argument('--meshcnn_root',
                        type=str,
                        help="Root folder of MeshCNN; it should contain a "
                        "`datasets` subfolder.",
                        required=True)
    parser.add_argument('--root_segmented_meshes',
                        type=str,
                        help="Folder containing the segmented meshes, e.g., "
                        "'/home/user/coseg_aliens/segmented_test_meshes'.",
                        required=True)
    parser.add_argument('--output_soft_labels_folder',
                        '--o',
                        type=str,
                        help="Folder where the output soft edge-labels should "
                        "be saved. ",
                        required=True)
    args = parser.parse_args()
    face_labels_to_soft_edge_labels(
        dataset_name=args.dataset_name,
        meshcnn_root=args.meshcnn_root,
        root_segmented_meshes=args.root_segmented_meshes,
        output_soft_labels_folder=args.output_soft_labels_folder)
