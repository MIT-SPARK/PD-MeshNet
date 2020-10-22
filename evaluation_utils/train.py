import argparse
import os

from pd_mesh_net.utils import BaseTrainingJob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--f',
        type=str,
        help="Path to the folder containing the pretrained model to evaluate "
        "and the training parameters.",
        required=True)
    parser.add_argument(
        '--checkpoint_batch_frequency',
        type=int,
        help=
        "Frequency (in batches) of checkpoint saving; if passed, overrides the "
        "argument `checkpoint_epoch_frequency`.")
    parser.add_argument('--checkpoint_epoch_frequency',
                        type=int,
                        help="Frequency (in epochs) of checkpoint saving.",
                        default=1)
    parser.add_argument('--last_epoch',
                        type=int,
                        help="Last training epoch.",
                        required=True)
    parser.add_argument('--verbose',
                        help="If passed, will display verbose prints.",
                        action='store_true')
    args = parser.parse_args()

    training_job_folder = os.path.abspath(args.f)
    assert (os.path.exists(training_job_folder)
           ), f"Could not find the training job folder {training_job_folder}."
    log_folder = os.path.dirname(training_job_folder)
    training_job_name = os.path.basename(training_job_folder)

    checkpoint_batch_frequency = None
    checkpoint_epoch_frequency = args.checkpoint_epoch_frequency

    if (args.checkpoint_batch_frequency):
        checkpoint_batch_frequency = args.checkpoint_batch_frequency
        checkpoint_epoch_frequency = None
        print("Saving checkpoints in epoch-and-batch format, with a checkpoint "
              "saved every {checkpoint_batch_frequency} checkpoints.")

    # Create training job.
    training_job = BaseTrainingJob(
        final_training_epoch=args.last_epoch,
        log_folder=log_folder,
        checkpoint_batch_frequency=checkpoint_batch_frequency,
        checkpoint_epoch_frequency=checkpoint_epoch_frequency,
        training_job_name=training_job_name,
        verbose=args.verbose)
    # Run training job.
    training_job.train()