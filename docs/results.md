## Results
In the following, we assume always to work on the activated virtualenv `pd_mesh_net` previously created, and we denote the main root of the repo as `$PD_MESH_NET_ROOT` (cf. [install.md](./install.md)).
### Data preparation
- Please download the pretrained models and the training parameters to reproduce
the paper results from https://www.dropbox.com/s/rip0mhsytvt14mh/pretrained_models_pd_mesh_net.zip.
- Extract the archive in a folder of your choice, which we refer to as `$UNZIP_FOLDER`.
- Choose a local folder `$DATASET_FOLDER` that will contain the datasets, which
  will be automatically downloaded and processed, e.g.,
  ```bash
    export DATASET_FOLDER=/home/user/datasets
  ```
- The training-parameter files contain a placeholder for the dataset folder.
Replace it with the folder defined above:
  ```bash
    cd $UNZIP_FOLDER
    find . -type f -name '*.yml' -exec sed -i "s@DATASET_FOLDER@${DATASET_FOLDER}@g" {} +
  ```
### Reproducing the accuracies from the pretrained models
- Choose the experiment of which to reproduce the accuracy and store the path to the associated folder in an enviromental variable `$JOB_FOLDER`. For instance, to reproduce the experiment on mesh segmentation on the COSEG `aliens` dataset, set:
    ```bash
    export JOB_FOLDER=$UNZIP_FOLDER/coseg/aliens
    ```
- Generate the dataset to run the evaluation. Also the training dataset will automatically be generated, because it will be used to compute mean and standard deviation of the features, used to standardize the test dataset:
    ```bash
    cd $PD_MESH_NET_ROOT/evaluation_utils
    python generate_dataset.py --f $JOB_FOLDER
    ```
- Run the evaluation:
    ```bash
    cd $PD_MESH_NET_ROOT/evaluation_utils
    python test.py --f $JOB_FOLDER
    ```

For reference, we report below the results from the paper and the variable `$JOB_FOLDER` that needs to be set to reproduce them:

| Experiment name           | Classification accuracy | Face-label accuracy | `$JOB_FOLDER` |
|---------------------------|:-----------------------:|:-------------------:|:-------------:|
|shrec16, config A, split 1 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/shrec16/split1` |
|shrec16, config A, split 2 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/shrec16/split2` |
|shrec16, config A, split 3 |           99.17%_*_    |         -           | `$UNZIP_FOLDER/shrec16/split3` |
|shrec16, config B, split 1 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_B/split1` |
|shrec16, config B, split 2 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_B/split2` |
|shrec16, config B, split 3 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_B/split3` |
|shrec16, config C, split 1 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_C/split1` |
|shrec16, config C, split 2 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_C/split2` |
|shrec16, config C, split 3 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec16/config_C/split3` |
|shrec10, config A, split 1 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/shrec10/split1` |
|shrec10, config A, split 2 |           98.67%_*_    |         -           | `$UNZIP_FOLDER/shrec10/split2` |
|shrec10, config A, split 3 |           99.00%_*_    |         -           | `$UNZIP_FOLDER/shrec10/split3` |
|shrec10, config B, split 1 |           100.00%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_B/split1` |
|shrec10, config B, split 2 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_B/split2` |
|shrec10, config B, split 3 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_B/split3` |
|shrec10, config C, split 1 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_C/split1` |
|shrec10, config C, split 2 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_C/split2` |
|shrec10, config C, split 3 |           99.67%_*_    |         -           | `$UNZIP_FOLDER/ablation_configs/shrec10/config_C/split3` |
|cube engraving, config A |           94.39%    |         -           | `$UNZIP_FOLDER/cube_engraving/` |
|coseg `aliens`, config A, U-Net, `mean` pooling aggregation |           -    |      97.63\%              | `$UNZIP_FOLDER/coseg/aliens_mean_aggr/` |
|coseg `aliens`, config A, U-Net, `add` pooling aggregation |           -    |      98.18\%              | `$UNZIP_FOLDER/coseg/aliens_add_aggr/` |
|coseg `chairs`, config A, U-Net, `mean` pooling aggregation  |           -    |      97.08\%              | `$UNZIP_FOLDER/coseg/chairs_mean_aggr/` |
|coseg `chairs`, config A, U-Net, `add` pooling aggregation |           -    |      97.23\%              | `$UNZIP_FOLDER/coseg/chairs_add_aggr/` |
|coseg `vases`, config A, U-Net, `mean` pooling aggregation  |           -    |      95.47\%              | `$UNZIP_FOLDER/coseg/vases_mean_aggr/` |
|coseg `vases`, config A, U-Net, `add` pooling aggregation |           -    |      95.36\%              | `$UNZIP_FOLDER/coseg/vases_add_aggr/` |
|coseg `aliens`, config A, superpixel-like |           -    |      94.75\%              | `$UNZIP_FOLDER/ablation_superpixels/coseg_aliens/` |
|coseg `chairs`, config A, superpixel-like |           -    |      93.74\%              | `$UNZIP_FOLDER/ablation_superpixels/coseg_chairs/` |
|coseg `vases`, config A, superpixel-like |           -    |      92.79\%              | `$UNZIP_FOLDER/ablation_superpixels/coseg_vases/` |
|human body, config A, U-Net, `mean` pooling aggregation |           -    |      84.78\%              | `$UNZIP_FOLDER/human_seg_mean_aggr/` |
|human body, config A, U-Net, `add` pooling aggregation |           -    |      85.61\%              | `$UNZIP_FOLDER/human_seg_add_aggr/` |

*Both for shrec16 and shrec10, the results reported in the paper are averaged over the 3 splits (e.g., for shrec10, config B, (100. + 99.67 + 99.67) / 3 = 99.78\%).

### Retraining
To rerun a training job with the same parameters that allowed to obtain the
pretrained models above:
- Navigate to the folder `$JOB_FOLDER` corresponding to the job;
- If you would like to start the training process from scratch using the same
parameters that allowed to obtain the results above, move or delete the
checkpoints in the `checkpoints/` folder. Optionally, the checkpoint can be left
in the folder, but when using the training script, the training will resume from
the epoch of the saved pretrained model;
- Start the training job, specifying at which epoch the training job should
terminate, as shown below:
  ```bash
  cd $PD_MESH_NET_ROOT/evaluation_utils
  python train.py --f $JOB_FOLDER --last_epoch 200   #e.g., final epoch 200
  ```

### Reproducing the segmentation accuracies that require predicted edge labels (cf. Sec. H of the Supplementary Material `.pdf`)
#### Setting up MeshCNN
Since we compute the accuracy defined by MeshCNN (cf. Supplementary Material
`.pdf`), we use their implementation to compute the accuracy, which relies on
internal data representations of their code. It is therefore required to set up
MeshCNN as follows:
- Please download MeshCNN in a folder `$DOWNLOAD_FOLDER` of you choice, using
the [official code provided](https://github.com/ranahanocka/MeshCNN) (last
tested with commit `15b83cc`):
  ```bash
  cd $DOWNLOAD_FOLDER
  git clone https://github.com/ranahanocka/MeshCNN
  export MESHCNN_ROOT=$DOWNLOAD_FOLDER/MeshCNN
  cd $MESHCNN_ROOT
  git checkout 15b83cc
  ```
- Download the (segmentation) datasets in the version of MeshCNN:
  ```bash
  cd $MESHCNN_ROOT
  bash ./scripts/coseg_seg/get_data.sh
  bash ./scripts/human_seg/get_data.sh
  ```
- Move the label-conversion scripts in the  `edge_label_accuracies` folder to
  the MeshCNN folder:
  ```bash
  cp $PD_MESH_NET_ROOT/evaluation_utils/edge_label_accuracies/* $MESHCNN_ROOT
  ```
#### Converting our face labels to soft edge labels
- For any of the segmentation experiments above (valid `$JOB_FOLDER`s are `$UNZIP_FOLDER/coseg/aliens/`, `$UNZIP_FOLDER/coseg/chairs/`, `$UNZIP_FOLDER/coseg/vases/`, `$UNZIP_FOLDER/human_seg/`)
generate the segmented meshes by running the same evaluation script shown in the
section above, but with the `--save_clusters` flag. For instance, for the COSEG
`aliens` experiment (assuming the dataset was already generated, cf. above):
  ```bash
    export JOB_FOLDER=$UNZIP_FOLDER/coseg/aliens/
    cd $PD_MESH_NET_ROOT/evaluation_utils
    python test.py --f $JOB_FOLDER --save_clusters
  ```
- Choose a folder `$OUTPUT_SOFT_LABELS_FOLDER` where the converted edge
soft-labels should be saved, and convert the predicted face labels to edge
soft-labels, e.g.,:
  ```bash
  export OUTPUT_SOFT_LABELS_FOLDER=~/output_soft_labels
  mkdir $OUTPUT_SOFT_LABELS_FOLDER
  export DATASET_NAME=coseg_aliens
  # Remember to update JOB_FOLDER accordingly, cf. above.
  workon pd_mesh_net
  cd $MESHCNN_ROOT
  python obtain_soft_edge_labels.py \
    --dataset_name $DATASET_NAME \
    --meshcnn_root $MESHCNN_ROOT \
    --root_segmented_meshes $JOB_FOLDER/segmented_test_meshes \
    --o $OUTPUT_SOFT_LABELS_FOLDER
  ```
- Evaluate the accuracies using the script `test_pd_mesh_net.py`, as follows:
    ```bash
    python test_pd_mesh_net.py   \
      --dataset_name $DATASET_NAME \
      --meshcnn_root $MESHCNN_ROOT \
      --soft_labels_folder $OUTPUT_SOFT_LABELS_FOLDER
    ```

  For reference, we report below the results from the paper and the variables that need to be set to reproduce them:

  | Experiment name           | Acc. ground-truth hard edge labels | Acc. ground-truth soft edge labels  | `$JOB_FOLDER` | `$DATASET_NAME` |
  |---------------------------|:-----------------------:|:-------------------:|:-------------:|:-------------:|
  |coseg `aliens`, config A, U-Net, `mean` pooling aggregation |           96.51%    |      98.53%              |  `$UNZIP_FOLDER/coseg/aliens_mean_aggr/` | `coseg_aliens` |
  |coseg `aliens`, config A, U-Net, `add` pooling aggregation |           97.06%    |      99.03%              |  `$UNZIP_FOLDER/coseg/aliens_add_aggr/` | `coseg_aliens` |
  |coseg `chairs`, config A, U-Net, `mean` pooling aggregation |           96.26%    |      97.93%              |  `$UNZIP_FOLDER/coseg/chairs_mean_aggr/` | `coseg_chairs` |
  |coseg `chairs`, config A, U-Net, `add` pooling aggregation |           96.44%    |      98.21%              |  `$UNZIP_FOLDER/coseg/chairs_add_aggr/` | `coseg_chairs` |
  |coseg `vases`, config A, U-Net, `mean` pooling aggregation |           94.70%    |      97.96%              |  `$UNZIP_FOLDER/coseg/vases_mean_aggr/` | `coseg_vases` |
  |coseg `vases`, config A, U-Net, `add` pooling aggregation |           94.57%    |      97.83%              |  `$UNZIP_FOLDER/coseg/vases_add_aggr/` | `coseg_vases` |
  |human body, config A, U-Net, `mean` pooling aggregation |           84.13%    |      90.44%              |  `$UNZIP_FOLDER/human_seg_mean_aggr/` | `human_seg` |
  |human body, config A, U-Net, `add` pooling aggregation |           85.09%    |      91.11%              |  `$UNZIP_FOLDER/human_seg_add_aggr/` | `human_seg` |
