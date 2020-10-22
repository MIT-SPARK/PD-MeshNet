## Evaluation
An example script to perform evaluation from a pretrained model is provided in [this file](../training_jobs/mesh_classification/shrec_16/test_job_1.py). To run it, simply navigate to the folder and execute this command in the `virtualenv` created upon installation:
```bash
python test_job_1.py
```

Similarly to `BaseTrainingJob`, a helper class for test jobs is provided: [`BaseTestJob`](../pd_mesh_net/utils/base_test_job.py).
Evaluation can be performed by:
- Initializing the instance of `BaseTestJob` with:
    - The name of the training job to use for evaluation (argument `training_job_name`);
    - The path to the root folder containing the folder `<training_job_name>` as a subfolder (argument `log_folder`);
    - The type of task for which the model was training and that should be used to compute the test accuracy (argument `task_type`, cf. factory method [`compute_num_correct_predictions`](../pd_mesh_net/utils/losses.py));
    - Optionally, the parameters of a different dataset and/or data loader than the ones used for training (same type of parameters as those used in `BaseTrainingJob`).
- Running the test job:
    ```python
    test_job.test()
    ```
    (if `test_job` is the instance of `BaseTestJob` used to perform evaluation).
