# Primal-Dual Mesh Convolutional Neural Networks

**Authors:** Francesco Milano, Antonio Loquercio, Antoni Rosinol, Davide Scaramuzza, Luca Carlone

## Installation
Please follow the instructions in the file [install.md](./docs/install.md).

## Training and evaluation
- For a detailed explanation on how to run a training job and perform evaluation
on the test set, please see [training.md](./docs/training.md) and [evaluation.md](./docs/evaluation.md), respectively.
- To reproduce the paper results, either by retraining a network or by using a
pretrained model, please see [results.md](./docs/results.md).

## Datasets
The code provided automatically downloads the datasets SHREC and Cube Engraving
in the downsampled version provided by
[MeshCNN](https://github.com/ranahanocka/MeshCNN). For the SHREC dataset, we
randomly generated two new sets for Split 16 and three sets for Split 10 (cf.
main paper). We also automatically download the COSEG and Human Body datasets,
which we preprocessed from the version provided by
[MeshCNN](https://github.com/ranahanocka/MeshCNN) so as to convert ground-truth
labels on the edges to ground-truth labels on the faces (cf., e.g., [`pd_mesh_net.datasets.coseg_dual_primal.py`](./pd_mesh_net/datasets/coseg_dual_primal.py)).

## Unit tests
For details on how to run unit tests, please see [unit_tests.md](./docs/unit_tests.md).


## Publications

If you find this work useful for your research, please cite:

- F. Milano, A. Loquercio, A. Rosinol, D. Scaramuzza, L. Carlone, [**Primal-Dual Mesh Convolutional Neural Networks**](http://rpg.ifi.uzh.ch/docs/NeurIPS20_Milano.pdf). 34th _Conference on Neural Information Processing Systems (NeurIPS)_, 2020.
 
 ```bibtex
 @InProceedings{Milano20NeurIPS-PDMeshNet,
   title = {Primal-Dual Mesh Convolutional Neural Networks},
   author = {Milano, Francesco and Loquercio, Antonio and Rosinol, Antoni and Scaramuzza, Davide and Carlone, Luca},
   year = {2020},
   booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
   url = {https://github.com/MIT-SPARK/PD-MeshNet}
 }
```

## Acknowledgements

The structure of parts of the code is based on similar code from [MeshCNN](https://github.com/ranahanocka/MeshCNN/) and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).