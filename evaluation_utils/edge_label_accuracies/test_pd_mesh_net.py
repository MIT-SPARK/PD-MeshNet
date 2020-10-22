import argparse
import glob
import numpy as np
import os
import torch

from data.segmentation_data import read_sseg, read_seg
from models.layers.mesh import Mesh
from util.util import seg_accuracy, pad

from label_converter import EdgeLabelToFaceLabelConverter


class PdMeshNetAccuracyCounter:
    """Based on `util.writer.Writer` by MeshCNN.
    """

    def __init__(self):
        self.nexamples = 0
        self.ncorrect = 0
        self.ncorrect_hard = 0

    def print_acc(self, acc, acc_hard):
        print(
            f'Accuracy based on ground-truth soft labels: [{acc * 100:.4}%]. '
            f'Accuracy based on ground-truth hard labels [{acc_hard * 100:.4}%]'
        )

    def reset_counter(self):
        self.ncorrect = 0

    def update_counter(self, ncorrect, ncorrect_hard, nexamples):
        self.ncorrect += ncorrect
        self.ncorrect_hard += ncorrect_hard
        self.nexamples += nexamples

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples

    @property
    def acc_hard(self):
        return float(self.ncorrect_hard) / self.nexamples


class Opt:

    def __init__(self, dataset_name):
        self.num_aug = 1
        if (dataset_name in ['coseg_aliens', 'coseg_chairs', 'coseg_vases']):
            self.nclasses = 4
        elif (dataset_name in ['human_seg']):
            self.nclasses = 8
        else:
            raise ValueError(
                "Please specify one of the following dataset names: "
                "['coseg_aliens', 'coseg_chairs', 'coseg_vases', 'human_seg'].")


def run_test(dataset_name, meshcnn_root, soft_labels_folder, verbose):
    opt = Opt(dataset_name)

    # Initialize counter.
    accuracy_counter = PdMeshNetAccuracyCounter()
    accuracy_counter.reset_counter()

    folder_original_meshes = os.path.join(meshcnn_root,
                                          f'datasets/{dataset_name}/test/')

    mesh_filenames = glob.glob(os.path.join(folder_original_meshes, '*.obj'))
    for mesh_filename in mesh_filenames:
        # Load original mesh.
        mesh = Mesh(file=mesh_filename, opt=opt, hold_history=False)
        mesh_name = mesh_filename.rsplit('.')[0].rpartition('/')[-1]
        mesh_name = f'{mesh_name}_aug_0_clusterized'
        if (verbose):
            print(f"Original mesh filename = '{mesh_filename}'.")
        # Load predicted soft labels.
        pred_soft_labels_file = os.path.join(
            soft_labels_folder, f'{dataset_name}/sseg/{mesh_name}.seseg')
        predicted_soft_labels = np.loadtxt(open(pred_soft_labels_file, 'r'),
                                           dtype='float64')
        num_edges = len(predicted_soft_labels)
        # Pad with zero the last class of coseg `chairs` (MeshCNN provides an
        # empty column for the 4th class, although the classes are 3).
        predicted_soft_labels_new = np.zeros([num_edges, opt.nclasses])
        predicted_soft_labels_new[:, :predicted_soft_labels.
                                  shape[1]] = predicted_soft_labels
        predicted_soft_labels = predicted_soft_labels_new

        assert (mesh.edges_count == num_edges)
        assert (predicted_soft_labels.ndim == 2 and
                predicted_soft_labels.shape[1] == opt.nclasses)
        num_predictions = predicted_soft_labels.shape[0]
        # Create two versions (cf. Supplementary Material `.pdf`), in each of
        # which one of the two predicted soft labels of each edge is considered
        # as 'hard label'.
        predicted_versions = np.empty([num_predictions, 2], dtype=np.long)
        for prediction_idx, prediction in enumerate(predicted_soft_labels):
            nonzero_pred = [b.item() for b in np.argwhere(prediction > 0)]
            assert (len(nonzero_pred) in [1, 2])
            if (len(nonzero_pred) == 1):
                nonzero_pred = 2 * nonzero_pred
            predicted_versions[prediction_idx] = nonzero_pred

        short_mesh_name = mesh_name.split('_aug')[0]
        # Load ground-truth softlabels.
        gt_soft_labels_file = os.path.join(
            meshcnn_root,
            f'datasets/{dataset_name}/sseg/{short_mesh_name}.seseg')
        gt_soft_labels = read_sseg(gt_soft_labels_file)
        # Load ground-truth hard labels.
        gt_hard_labels_file = os.path.join(
            meshcnn_root, f'datasets/{dataset_name}/seg/{short_mesh_name}.eseg')
        gt_hard_labels = read_seg(gt_hard_labels_file) - 1
        edge_labels = np.empty([num_edges], dtype=np.float)

        # Compute accuracy based on soft edge labels.
        correct = 0.5 * seg_accuracy(
            torch.from_numpy(predicted_versions[:, 0].reshape(1, -1)),
            torch.from_numpy(gt_soft_labels.reshape(
                1, -1, opt.nclasses)), [mesh]) + 0.5 * seg_accuracy(
                    torch.from_numpy(predicted_versions[:, 1].reshape(1, -1)),
                    torch.from_numpy(gt_soft_labels.reshape(
                        1, -1, opt.nclasses)), [mesh])
        # Compute accuracy based on hard edge labels.
        correct_hard = 0.5 * ((
            np.nonzero(predicted_versions[:, 0] == gt_hard_labels)[0].shape[0] +
            np.nonzero(predicted_versions[:, 1] == gt_hard_labels)[0].shape[0])
                             ) / num_edges
        if (verbose):
            print(
                "\tThe percentage of the mesh that was correct labelled "
                f"according to ground-truth soft labels is {correct * 100:.4}%."
            )
            print("\tThe percentage of the mesh that was correct labelled "
                  "according to ground-truth hard labels is "
                  f"{correct_hard * 100:.4}%.")
        accuracy_counter.update_counter(ncorrect=correct,
                                        ncorrect_hard=correct_hard,
                                        nexamples=1)
    accuracy_counter.print_acc(accuracy_counter.acc, accuracy_counter.acc_hard)


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
    parser.add_argument('--soft_labels_folder',
                        type=str,
                        help="Folder where the output soft edge-labels were "
                        "saved. ",
                        required=True)
    parser.add_argument('--verbose',
                        action='store_true',
                        help="If passed, displays verbose prints.")
    args = parser.parse_args()

    run_test(dataset_name=args.dataset_name,
             meshcnn_root=args.meshcnn_root,
             soft_labels_folder=args.soft_labels_folder,
             verbose=args.verbose)
