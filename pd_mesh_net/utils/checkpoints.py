import glob
import heapq
import os


def get_epoch_most_recent_checkpoint(checkpoint_subfolder):
    r"""Finds the most recent checkpoint in the log folder expected to contain
    the checkpoints and returns its epoch number.

    Args:
        checkpoint_subfolder (str): Path of the (sub)folder containing the
            checkpoints.

    Returns:
        most_recent_checkpoint (int or None): The epoch index of the most recent
            checkpoint found in the log folder - if one is found; None
            otherwise.
    """
    checkpoints_found = [
        f for f in glob.glob(
            os.path.join(checkpoint_subfolder, 'checkpoint_*.pth'))
    ]
    # Found maximum epoch index.
    most_recent_checkpoint = -1
    for f in checkpoints_found:
        epoch_number = os.path.basename(f).split('.')[0].split(
            'checkpoint_')[-1]
        if (epoch_number.isdigit()):
            epoch_number = int(epoch_number)
            if (epoch_number > most_recent_checkpoint):
                most_recent_checkpoint = epoch_number

    if (most_recent_checkpoint == -1):
        most_recent_checkpoint = None

    return most_recent_checkpoint


def get_epoch_and_batch_most_recent_checkpoint(checkpoint_subfolder):
    r"""Finds the most recent checkpoint in the log folder expected to contain
    the checkpoints and returns its epoch number and batch index.
    Args:
        checkpoint_subfolder (str): Path of the (sub)folder containing the
            checkpoints.
    Returns:
        epoch_most_recent_checkpoint (int or None): The epoch index of the most
            recent checkpoint found in the log folder - if one is found; None
            otherwise.
        batch_most_recent_checkpoint (int or None): The batch index of the most
            recent checkpoint found in the log folder - if one is found; None
            otherwise.
    """
    checkpoints_found = [
        f for f in glob.glob(
            os.path.join(checkpoint_subfolder, 'checkpoint_*.pth'))
    ]
    # Found maximum epoch index.
    epoch_most_recent_checkpoint = -1
    batch_index_most_recent_checkpoint = -1
    for f in checkpoints_found:
        epoch_number_and_batch_index = os.path.basename(f).split('.')[0].split(
            'checkpoint_')[-1]
        epoch_number, batch_index = epoch_number_and_batch_index.split(
            '_batch_')
        if (epoch_number.isdigit() and batch_index.isdigit()):
            epoch_number = int(epoch_number)
            batch_index = int(batch_index)
            if (epoch_number > epoch_most_recent_checkpoint):
                epoch_most_recent_checkpoint = epoch_number
                batch_index_most_recent_checkpoint = batch_index
            elif (epoch_number == epoch_most_recent_checkpoint):
                if (batch_index > batch_index_most_recent_checkpoint):
                    batch_index_most_recent_checkpoint = batch_index

    assert ((epoch_most_recent_checkpoint == -1) == (
        batch_index_most_recent_checkpoint == -1))

    if (epoch_most_recent_checkpoint == -1):
        epoch_most_recent_checkpoint = None
    if (batch_index_most_recent_checkpoint == -1):
        batch_index_most_recent_checkpoint = None

    return epoch_most_recent_checkpoint, batch_index_most_recent_checkpoint


def find_epoch_and_batch_all_checkpoints(checkpoint_subfolder):
    r"""Finds the all the checkpoints in the log folder expected to contain
    the checkpoints and returns a sorted list of all the epoch numbers or of all
    the epoch numbers and batch indices, depending on whether checkpoints are
    saved only at the end of the epochs or also at the end of batches.

    Args:
        checkpoint_subfolder (str): Path of the (sub)folder containing the
            checkpoints.
    Returns:
        epochs_andor_batches_checkpoints (list): Empty list if no checkpoints
            are found in the log folder; otherwise:
            - List, sorted by increasing epoch number, of the epochs of all the
              checkpoints found in the folder, if checkpoints are in the
              epoch-only format;
            - List of tuples, if checkpoints are in the epoch-and-batch format;
              each tuple is associated to a checkpoint found in the folder, with
              the first and second element in each tuple representing
              respectively. The tuples are sorted by increasing epoch number and
              increasing batch index.
    """
    checkpoints_found = [
        f for f in glob.glob(
            os.path.join(checkpoint_subfolder, 'checkpoint_*.pth'))
    ]

    found_epochonly_checkpoint = False
    found_epochandbatch_checkpoint = False
    epochs_andor_batches_checkpoints = []
    # Find epoch and/or batch index for each checkpoint.
    for f in checkpoints_found:
        epoch_number_andor_batch_index = os.path.basename(f).split(
            '.')[0].split('checkpoint_')[-1]
        if (epoch_number_andor_batch_index.isdigit()):
            # Epoch-only format.
            epoch_number = int(epoch_number)
            found_epochonly_checkpoint = True
            assert (not found_epochandbatch_checkpoint), (
                "Found checkpoints of incompatible format: both epoch-only and "
                "epoch-and-batch checkpoints were found.")
            # Add the element to the list of epochs.
            heapq.heappush(epochs_andor_batches_checkpoints, epoch_number)
        else:
            epoch_number, batch_index = epoch_number_andor_batch_index.split(
                '_batch_')
            assert (epoch_number.isdigit() and batch_index.isdigit()
                   ), f"Checkpoint {f} is in unrecognized format."
            # Epoch-and-batch format.
            epoch_number = int(epoch_number)
            batch_index = int(batch_index)
            found_epochandbatch_checkpoint = True
            assert (not found_epochonly_checkpoint), (
                "Found checkpoints of incompatible format: both epoch-only and "
                "epoch-and-batch checkpoints were found.")
            # Add the element to the list of epochs.
            heapq.heappush(epochs_andor_batches_checkpoints,
                           (epoch_number, batch_index))

    # Convert the list of epochs/epoch-and-batch into the sorted format.
    if (len(epochs_andor_batches_checkpoints) > 0):
        epochs_andor_batches_checkpoints = [
            heapq.heappop(epochs_andor_batches_checkpoints)
            for _ in range(len(epochs_andor_batches_checkpoints))
        ]

    return epochs_andor_batches_checkpoints


def get_generic_checkpoint(checkpoint_subfolder):
    r"""Searches for a checkpoint with a single generic filename (but ".pth"
    extension) in the given subfolder.

    Args:
        checkpoint_subfolder (str): Path of the (sub)folder containing the
            checkpoints.
    Returns:
        checkpoint_filename (str or None): If a single checkpoint with valid
            extension is found, complete filename of the checkpoint. Otherwise,
            None.
    """
    checkpoints_found = [
        f for f in glob.glob(os.path.join(checkpoint_subfolder, '*.pth'))
    ]

    if (len(checkpoints_found) == 1):
        checkpoint_filename = checkpoints_found[0]
    else:
        checkpoint_filename = None

    return checkpoint_filename