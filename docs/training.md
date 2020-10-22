## Training
An example training job is provided in [this file](../training_jobs/mesh_classification/shrec_16/training_job_1.py). To run it, simply navigate to the folder and execute this command in the `virtualenv` created upon installation:
```bash
python training_job_1.py
```
### Further info on setting up a training job
More in general, you can run a training job by using the helper class [`BaseTrainingJob`](../pd_mesh_net/utils/base_training_job.py). As done in the above example script, you can simply define the parameters of the network, dataset, data loader, optimizer, learning-rate scheduler (optional) and loss that you with to use for training, as well as the parameters of the training job (the only required ones are the folder where to store the training log, argument `log_folder`, and the final epoch of training, argument `final_training_epoch`), and pass them to instance of `BaseTrainingJob` upon initialization. Then you can simply run
```python
training_job.train()
```
(if `training_job` is the instance of `BaseTrainingJob`). The helper class will automatically take care of:
- Initializing the training job, by internally calling factory methods to instantiate the network (cf. [`create_model`](../pd_mesh_net/utils/models.py)), the dataset and data loader (cf. [`DualPrimalDataLoader`](../pd_mesh_net/data/data_loader.py) and [`create_dataset`](../pd_mesh_net/utils/datasets.py)), the optimizer (cf. [`create_optimizer`](../pd_mesh_net/utils/optimizers.py)), the learning-rate scheduler (cf. [`create_lr_scheduler`](../pd_mesh_net/utils/lr_schedulers.py)) and the loss (cf. [`create_loss`](../pd_mesh_net/utils/losses.py)).
- Saving the training parameters to disk, automatically filling the optional arguments of the factory methods with the default values, as well as the training checkpoints, for resuming training and performing evaluation on a pretrained model.
- Automatically checking the compatibility of the parameters used when resuming a training job or evaluating a model with those used when first launching the training job.
- Saving training logs that can be visualized in TensorBoard.

The structure of a folder created to store all the data from a training job is the following:
```.
<training_job_name>
└───checkpoints/
|   |   checkpoint_0001.pth
|   |   ...
|
└───tensorboard/
|   |   <TensorBoard event files>
|
└───training_parameters.yml
```
- The training job name can be defined by passing the argument `training_job_name` to the `BaseTrainingJob`. If this is not done, a name will be automatically assigned to the job based on the time at which the job is executed, in the format `YYYYMMDD_hhmmss`.
- The folder `checkpoints/` contains the PyTorch model files containing the weights of the network, the optimizer (and learning-rate scheduler-, if present) parameters, and additional data for TensorBoard logging.
- The folder `tensorboard/` contains the training logs that can be visualized in TensorBoard. A data point is added to the training logs with a frequency defined by the argument `minibatch_summary_writer_frequency` of the `BaseTrainingJob` (in terms of number of mini-batches, with default value 10). To visualize the training logs in TensorBoard, navigate to the `tensorboard/` folder and run the following command within the `virtualenv` created upon installation:

    ```bash
    tensorboard --logdir .
    ```
- The file `training_parameters.yml` contains the parameters of the training job in YAML format. These are used to resume a training job, ensure compatibility of the parameters, and recreate the components used in the training job for evaluation with a pretrained model.

### Resuming a training job
The easiest way to resume a training job is to pass the argument `training_job_name` to a `BaseTrainingJob` and run `train()` on it, as done in the example below:
```python
import os

from pd_mesh_net.utils import BaseTrainingJob

TRAINING_PARAMS = {
    'final_training_epoch': 200,
    'log_folder': os.path.abspath('.'),     # Change the log folder if necessary.
    'training_job_name': 'YYYYMMDD_hhmmss'  # Set your training-job name here.
}

if __name__ == '__main__':
    # Create training job.
    training_job = BaseTrainingJob(**TRAINING_PARAMS)
    # Run training job.
    training_job.train()
```
The `BaseTrainingJob` will automatically load the training parameters and the latest available checkpoint in the folder `checkpoints/` and resume the training job.