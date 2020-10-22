import inspect
import torch
import warnings


def create_loss(task_type, **loss_params):
    r"""Creates an instance of the input loss, according to the input task type
    and with the input parameters.
    Modified from MeshCNN (https://github.com/ranahanocka/MeshCNN/).

    Args:
        task_type (str): Name that identifies the task for which the loss should
            be used. Valid values are: `classification`, `segmentation`.
        ...
        Optional parameters of the loss.

    Returns:
        loss (torch.nn.modules.loss._Loss): The instance of the loss with the
            input parameters.
    """
    if (task_type in ['classification', 'segmentation']):
        loss_class = torch.nn.CrossEntropyLoss
        # Only keep the valid loss parameters.
        valid_loss_params = {}
        possible_valid_params = [
            p for p in inspect.getfullargspec(loss_class).args if p != 'self'
        ]
        for param, param_value in loss_params.items():
            if (param in possible_valid_params):
                valid_loss_params[param] = param_value
            else:
                warnings.warn(f"Ignoring parameter '{param}', invalid for loss "
                              f"'{loss_class.__name__}'.")
        loss = loss_class(**valid_loss_params)
    else:
        raise KeyError(
            f"No task is known of type '{task_type}'. Cannot create loss.")

    return loss


def compute_num_correct_predictions(task_type,
                                    outputs,
                                    targets,
                                    face_areas=None,
                                    face_to_batch_sample=None):
    r"""Computes the number of outputs predicted by a network that match the
    expected target values. Optionally, in case of segmentation task, the number
    of correct predictions is weighted by the area of the mesh faces.

    Args:
        task_type (str): Name that identifies the task for which the prediction
            was generated. Valid values are: `classification`, `segmentation`.
        outputs (torch.Tensor): Outputs of the network for the given task:
            - When `task_type` is `classification`, this tensor is of shape
                `[num_samples, num_classes]` - where `num_samples` is the number
                of samples on which evaluation is being performed and
                `num_classes` is the number of classes of the classification
                task - and its element `outputs[i, j]` represents the
                unnormalized score assigning the `i`-th sample to the `j`-th
                class;
            - When `task_type` is `segmentation`, this tensor is of shape
                `[num_faces, num_classes]` - where `num_faces` is the total
                number of (mesh) faces in all the samples on which evaluation is
                being performed and `num_classes` is the number of classes that
                can be assigned to the faces - and its element `outputs[i, j]`
                represents the unnormalized score assigning the `i`-th input
                face to the `j`-th class.
        targets (torch.Tensor): Expected target values of the samples for the
            given task:
            - When `task_type` is `classification`, this tensor is of shape
                `[num_samples,]` - where `num_samples` is the number of samples
                on which evaluation is being performed - and its element
                `targets[i]` represents the ground-truth class of the `i`-th
                sample;
            - When `task_type` is `segmentation`, this tensor is of shape
                `[num_faces,]` - where `num_faces` is the total number of (mesh)
                faces in all the samples on which evaluation is being
                performed  - and its element `targets[i]` represents the
                ground-truth class of the `i`-th input face.
        face_areas (torch.Tensor, optional): If not None, tensor of shape
            `[num_faces,]` - where `num_faces` is the total number of (mesh)
            faces in all the samples on which evaluation is being performed -
            the `i`-th element of which represents the area of the `i`-th face
            in the mesh (the absolute scale of the area is not important, but it
            is assumed that all areas from the same batch sample are in the same
            scale). Only considered if argument `task_type` is `segmentation`.
            If not None, requires the argument `face_to_batch_sample` to be also
            passed as input. (default: :obj:`None`)
        face_to_batch_sample (torch.Tensor, optional): If not None, tensor of
            shape `[num_faces,]` - where `num_faces` is the total number of
            (mesh) faces in all the samples on which evaluation is being
            performed - the `i`-th element of which represents the index of the
            sample in the batch to which the `i`-th face belongs. Only
            considered if argument `task_type` is `segmentation`. If not None,
            requires the argument `face_to_batch_sample` to be also passed as
            input. (default: :obj:`None`)
    
    Returns:
        num_correct_predictions (int/float): Number of input predictions that
            match the corresponding target value.
            - In case of classification tasks: number of input samples that are
              correctly classified;
            - In case of segmentation tasks:
              - If non-weighted accuracy is used: number of faces in the input
                samples that are assigned the correct class label;
              - If weighted accuracy is used:
                :math:`\displaystyle\sum_{i:\textrm{mesh}_i\in Meshes}
                \textrm{accuracy}_i\cdot \textrm{num_faces}_i}`,
                where :math:`\textrm{accuracy}_i = \frac{\displaystyle\sum_{
                    j: \textrm{face}_j\in Faces(\textrm{mesh}_i)} Area(\textrm{
                        face}_j)\cdot\delta(\textrm{prediction}_j = \textrm{
                            ground_truth}_j)}{\displaystyle\sum_{j: \textrm{
                                face}_j\in Faces(\textrm{mesh}_i)} Area(\textrm{
                                    face}_j)}`.

    """

    def areaweighted_accuracy_single_sample(sample_idx):
        r"""Computes the area-weighted segmentation accuracy of a single sample,
        cf. :math:`\textrm{accuracy}_i` in the docs above.

        Args:
            sample_idx (int): Index of the sample of which to compute the
                accuracy.

        Returns:
            accuracy (float): Area-weighted segmentation accuracy of the input
                sample, as defined above.
        """
        face_areas_from_sample = face_areas[face_to_batch_sample == sample_idx]
        assert (face_areas_from_sample.dim() == 1)
        outputs_from_sample = outputs[face_to_batch_sample == sample_idx]
        targets_from_sample = targets[face_to_batch_sample == sample_idx]

        accuracy = (face_areas_from_sample *
                    (outputs_from_sample.argmax(axis=1) == targets_from_sample)
                   ).sum() / face_areas_from_sample.sum()
        num_faces_in_sample = face_areas_from_sample.shape[0]

        return accuracy, num_faces_in_sample

    if (task_type in ['classification', 'segmentation']):
        assert (isinstance(outputs, torch.Tensor) and
                isinstance(targets, torch.Tensor))
        assert ((face_areas is None) == (face_to_batch_sample is None)), (
            "When one of the arguments `face_areas`, `face_to_batch_sample`, "
            "also the other one needs to be not None.")
        # The number of predictions corresponds to the number of samples in the
        # batch in case of mesh classification (in which a single label is
        # assigned to each shape) and to the number of total mesh faces in the
        # batch in case of mesh segmentation (in which a label is assigned to
        # each face).
        assert (outputs.shape[0] == targets.shape[0])
        assert (targets.dim() == 1)
        num_classes = outputs.shape[1]
        assert (0 <= targets.min().item() <= targets.max().item() <=
                num_classes - 1)
        if (face_areas is not None and task_type == 'segmentation'):
            # Area-weighted accuracy.
            assert (isinstance(face_areas, torch.Tensor) and
                    isinstance(face_to_batch_sample, torch.Tensor))
            assert (face_areas.shape == (outputs.shape[0],) and
                    face_to_batch_sample.shape == (outputs.shape[0],))
            # - Find the indices of all the samples in the batch.
            sample_indices = face_to_batch_sample.unique()
            # - Compute the accuracy for each sample.
            sum_accuracies = 0
            total_num_faces = 0
            for sample_idx in sample_indices:
                (accuracy_curr_sample,
                 num_faces_curr_sample) = areaweighted_accuracy_single_sample(
                     sample_idx=sample_idx)
                sum_accuracies += (accuracy_curr_sample * num_faces_curr_sample)
                total_num_faces += num_faces_curr_sample
            assert (total_num_faces == targets.shape[0])

            # - Compute the total accuracy.
            num_correct_predictions = sum_accuracies
        else:
            # Accuracy without area weighting.
            num_correct_predictions = (outputs.argmax(
                axis=1) == targets).sum().item()
    else:
        raise KeyError(
            f"No task is known of type '{task_type}'. Cannot compute number of "
            "correct predictions.")

    return num_correct_predictions
